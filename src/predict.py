"""التنبؤ على صورة.

التشغيل:
    python -m src.predict path/to/image.jpg
"""
import os
import sys

import torch
from PIL import Image

from . import config as C
from .data import get_transforms
from .model import load_trained
from .nutrition import CalorieCalculator
from .portion import estimate_food_area


def default_model_dir() -> str:
    """لو في موديل متدرب في SAVE_DIR (مثلًا على Drive) نستخدمه، غير كده الموديل الجاهز اللي في الريبو."""
    if os.path.exists(os.path.join(C.SAVE_DIR, "best_model.pth")):
        return C.SAVE_DIR
    return C.WEIGHTS_DIR


class FoodClassifier:
    def __init__(self, save_dir: str | None = None, device: str | None = None):
        save_dir = save_dir or default_model_dir()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.classes = load_trained(save_dir, self.device)
        self.tf = get_transforms(train=False)

    @torch.no_grad()
    def predict(self, image: Image.Image, top_k: int = 5):
        x = self.tf(image.convert("RGB")).unsqueeze(0).to(self.device)
        probs = self.model(x).softmax(1)[0]
        top = probs.topk(top_k)
        return [{"food": self.classes[i], "confidence": round(p.item(), 4)}
                for p, i in zip(top.values, top.indices)]


class CalorieEstimator:
    """صورة ← نوع الأكل ← السعرات."""

    def __init__(self, save_dir: str | None = None, device: str | None = None):
        self.classifier = FoodClassifier(save_dir, device)
        self.calculator = CalorieCalculator()

    def estimate(self, image: Image.Image, size: str | None = None, grams: float | None = None,
                 return_portion: bool = False):
        """size=None و grams=None ← الكمية بتتقدّر من الصورة تلقائي."""
        preds = self.classifier.predict(image, top_k=5)
        portion = None
        if not grams and size is None:
            portion = estimate_food_area(image)
        result = self.calculator.estimate(preds, size=size, grams=grams,
                                          food_area_cm2=portion.food_area_cm2 if portion else None)
        return (result, portion) if return_portion else result


SOURCE_LABEL = {
    "user": "الوزن اللي انت كتبته",
    "size": "حجم الطبق اللي اخترته",
    "image": "متقدّرة من الصورة",
    "bowl": "الحصة المعتادة، لأن عمق الشوربة مش باين في الصورة",
    "default": "الحصة المعتادة، لأن مفيش طبق واضح في الصورة",
}


def format_result(r: dict) -> str:
    area = f"، مساحة الأكل ~{r['food_area_cm2']} سم²" if r.get("food_area_cm2") and r["portion_source"] == "image" else ""
    lines = [
        f"🍽️  {r['name_ar']} ({r['food']}) — ثقة {r['confidence']:.0%}",
        f"⚖️  الكمية: ~{r['portion_g']} جرام ({SOURCE_LABEL[r['portion_source']]}{area})",
        f"🔥 السعرات: ~{r['calories']} kcal  (بين {r['calories_range'][0]} و {r['calories_range'][1]})",
    ]
    if r["uncertain"]:
        lines.append("⚠️  الموديل مش متأكد، والرقم متوسط موزون بين أكتر من صنف")
    if r["alternatives"]:
        alts = "، ".join(f"{a['name_ar']} ({a['confidence']:.0%})" for a in r["alternatives"])
        lines.append(f"🤔 احتمالات تانية: {alts}")
    return "\n".join(lines)


if __name__ == "__main__":
    # python -m src.predict image.jpg [small|medium|large]   (من غير حجم ← تقدير من الصورة)
    size = sys.argv[2] if len(sys.argv) > 2 else None
    est = CalorieEstimator()
    print(format_result(est.estimate(Image.open(sys.argv[1]), size=size)))
