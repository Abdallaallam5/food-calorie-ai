"""حساب السعرات من نتيجة التصنيف.

المعادلة الأساسية:
    السعرات = (سعرات كل 100 جرام) × (الكمية بالجرام) ÷ 100

الكمية دلوقتي بتيجي من "الحصة المعتادة" للصنف، والمستخدم يقدر يقول
الطبق صغير/وسط/كبير أو يدخل الوزن بالجرام. في خطوة لاحقة هنقدّر الكمية من الصورة نفسها.
"""
import csv
import os

DEFAULT_TABLE = os.path.join(os.path.dirname(__file__), "..", "data", "nutrition",
                             "food101_nutrition.csv")

SIZE_MULTIPLIER = {"small": 0.7, "medium": 1.0, "large": 1.4}

# لو ثقة الموديل في أول اختيار أقل من كده، بناخد متوسط موزون لأول 3 اختيارات
CONFIDENCE_THRESHOLD = 0.6

# هامش الخطأ المتوقع (الزيت والسمنة المخفيين، اختلاف الوصفات...)
ERROR_MARGIN = 0.25


def load_table(path: str = DEFAULT_TABLE) -> dict:
    with open(path, encoding="utf-8") as f:
        return {r["food"]: {"name_ar": r["name_ar"],
                            "kcal_per_100g": float(r["kcal_per_100g"]),
                            "serving_g": float(r["serving_g"])}
                for r in csv.DictReader(f)}


class CalorieCalculator:
    def __init__(self, table_path: str = DEFAULT_TABLE):
        self.table = load_table(table_path)

    def portion_grams(self, food: str, size: str = "medium", grams: float | None = None) -> float:
        if grams:
            return float(grams)
        return self.table[food]["serving_g"] * SIZE_MULTIPLIER.get(size, 1.0)

    def estimate(self, predictions: list[dict], size: str = "medium",
                 grams: float | None = None) -> dict:
        """predictions: ناتج FoodClassifier.predict، مرتبة من الأعلى ثقة للأقل."""
        top = predictions[0]
        if top["confidence"] >= CONFIDENCE_THRESHOLD:
            candidates = [top]
        else:
            candidates = [p for p in predictions[:3] if p["food"] in self.table]

        total_conf = sum(p["confidence"] for p in candidates) or 1.0
        kcal = 0.0
        for p in candidates:
            info = self.table[p["food"]]
            g = self.portion_grams(p["food"], size, grams)
            kcal += (p["confidence"] / total_conf) * info["kcal_per_100g"] * g / 100

        info = self.table[top["food"]]
        portion = self.portion_grams(top["food"], size, grams)
        return {
            "food": top["food"],
            "name_ar": info["name_ar"],
            "confidence": top["confidence"],
            "portion_g": round(portion),
            "kcal_per_100g": info["kcal_per_100g"],
            "calories": round(kcal),
            "calories_range": [round(kcal * (1 - ERROR_MARGIN)), round(kcal * (1 + ERROR_MARGIN))],
            "uncertain": len(candidates) > 1,
            "alternatives": [{"food": p["food"], "name_ar": self.table[p["food"]]["name_ar"],
                              "confidence": p["confidence"]} for p in predictions[1:3]
                             if p["food"] in self.table],
        }
