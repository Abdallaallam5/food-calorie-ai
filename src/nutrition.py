"""حساب السعرات من نتيجة التصنيف.

المعادلة الأساسية:
    السعرات = (سعرات كل 100 جرام) × (الكمية بالجرام) ÷ 100

الكمية بتتحدد بالترتيب ده:
    1. المستخدم كتب الوزن بالجرام ← بنستخدمه.
    2. المستخدم اختار small / medium / large ← الحصة المعتادة × معامل.
    3. اتقاست مساحة الأكل من الصورة ← المساحة × (جرام لكل سم²) حسب شكل الأكل.
    4. غير كده ← الحصة المعتادة.
"""
import csv
import os

DEFAULT_TABLE = os.path.join(os.path.dirname(__file__), "..", "data", "nutrition",
                             "food101_nutrition.csv")

SIZE_MULTIPLIER = {"small": 0.7, "medium": 1.0, "large": 1.4}

# جرام لكل سم² من مساحة الطبق حسب شكل الأكل (تقريبي: السُمك × الكثافة)
GRAMS_PER_CM2 = {
    "flat": 0.9,     # بيتزا، بان كيك، أومليت
    "piece": 1.3,    # لحمة، سمك، ساندويتشات، قطع
    "mound": 1.6,    # رز، مكرونة، كاري
    "salad": 0.6,    # سلطات (فيها هوا كتير)
    "dessert": 1.5,  # قطع كيك وحلويات
    "bowl": None,    # شوربة: العمق مش باين في الصورة، فبنستخدم الحصة المعتادة
}
# حدود منطقية حوالين الحصة المعتادة عشان غلطة في الصورة ما تطلعش رقم مجنون
MIN_FACTOR, MAX_FACTOR = 0.4, 2.5

# لو ثقة الموديل في أول اختيار أقل من كده، بناخد متوسط موزون لأول 3 اختيارات
CONFIDENCE_THRESHOLD = 0.6
# أقل احتمال نعرضه كبديل
MIN_ALTERNATIVE_CONF = 0.03

# هامش الخطأ المتوقع (الزيت والسمنة المخفيين، اختلاف الوصفات...)
ERROR_MARGIN = 0.25
ERROR_MARGIN_IMAGE = 0.30  # تقدير الكمية من الصورة بيزوّد عدم اليقين شوية


def load_table(path: str = DEFAULT_TABLE) -> dict:
    with open(path, encoding="utf-8") as f:
        return {r["food"]: {"name_ar": r["name_ar"],
                            "kcal_per_100g": float(r["kcal_per_100g"]),
                            "serving_g": float(r["serving_g"]),
                            "shape": r["shape"]}
                for r in csv.DictReader(f)}


class CalorieCalculator:
    def __init__(self, table_path: str = DEFAULT_TABLE):
        self.table = load_table(table_path)

    def portion_grams(self, food: str, size: str | None = None, grams: float | None = None,
                      food_area_cm2: float | None = None) -> tuple[float, str]:
        """يرجّع (الجرامات، مصدر التقدير)."""
        info = self.table[food]
        if grams:
            return float(grams), "user"
        if size in SIZE_MULTIPLIER:
            return info["serving_g"] * SIZE_MULTIPLIER[size], "size"
        density = GRAMS_PER_CM2.get(info["shape"])
        if food_area_cm2 and density:
            g = food_area_cm2 * density
            g = min(max(g, info["serving_g"] * MIN_FACTOR), info["serving_g"] * MAX_FACTOR)
            return g, "image"
        if food_area_cm2 and density is None:
            return info["serving_g"], "bowl"
        return info["serving_g"], "default"

    def estimate(self, predictions: list[dict], size: str | None = None,
                 grams: float | None = None, food_area_cm2: float | None = None) -> dict:
        """predictions: ناتج FoodClassifier.predict، مرتبة من الأعلى ثقة للأقل."""
        top = predictions[0]
        if top["confidence"] >= CONFIDENCE_THRESHOLD:
            candidates = [top]
        else:
            candidates = [p for p in predictions[:3] if p["food"] in self.table]

        total_conf = sum(p["confidence"] for p in candidates) or 1.0
        kcal = 0.0
        for p in candidates:
            g, _ = self.portion_grams(p["food"], size, grams, food_area_cm2)
            kcal += (p["confidence"] / total_conf) * self.table[p["food"]]["kcal_per_100g"] * g / 100

        info = self.table[top["food"]]
        portion, source = self.portion_grams(top["food"], size, grams, food_area_cm2)
        margin = ERROR_MARGIN_IMAGE if source == "image" else ERROR_MARGIN
        return {
            "food": top["food"],
            "name_ar": info["name_ar"],
            "confidence": top["confidence"],
            "portion_g": round(portion),
            "portion_source": source,
            "food_area_cm2": round(food_area_cm2) if food_area_cm2 else None,
            "kcal_per_100g": info["kcal_per_100g"],
            "calories": round(kcal),
            "calories_range": [round(kcal * (1 - margin)), round(kcal * (1 + margin))],
            "uncertain": len(candidates) > 1,
            "alternatives": [{"food": p["food"], "name_ar": self.table[p["food"]]["name_ar"],
                              "confidence": p["confidence"]} for p in predictions[1:3]
                             if p["food"] in self.table and p["confidence"] >= MIN_ALTERNATIVE_CONF],
        }
