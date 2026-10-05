"""التنبؤ على صورة.

التشغيل:
    python -m src.predict path/to/image.jpg
"""
import sys

import torch
from PIL import Image

from . import config as C
from .data import get_transforms
from .model import load_trained
from .nutrition import CalorieCalculator


class FoodClassifier:
    def __init__(self, save_dir: str = C.SAVE_DIR, device: str | None = None):
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

    def __init__(self, save_dir: str = C.SAVE_DIR, device: str | None = None):
        self.classifier = FoodClassifier(save_dir, device)
        self.calculator = CalorieCalculator()

    def estimate(self, image: Image.Image, size: str = "medium", grams: float | None = None):
        preds = self.classifier.predict(image, top_k=5)
        return self.calculator.estimate(preds, size=size, grams=grams)


def format_result(r: dict) -> str:
    lines = [
        f"🍽️  {r['name_ar']} ({r['food']}) — ثقة {r['confidence']:.0%}",
        f"⚖️  الكمية: ~{r['portion_g']} جرام",
        f"🔥 السعرات: ~{r['calories']} kcal  (بين {r['calories_range'][0]} و {r['calories_range'][1]})",
    ]
    if r["uncertain"]:
        lines.append("⚠️  الموديل مش متأكد، والرقم متوسط موزون بين أكتر من صنف")
    if r["alternatives"]:
        alts = "، ".join(f"{a['name_ar']} ({a['confidence']:.0%})" for a in r["alternatives"])
        lines.append(f"🤔 احتمالات تانية: {alts}")
    return "\n".join(lines)


if __name__ == "__main__":
    # python -m src.predict image.jpg [small|medium|large]
    size = sys.argv[2] if len(sys.argv) > 2 else "medium"
    est = CalorieEstimator()
    print(format_result(est.estimate(Image.open(sys.argv[1]), size=size)))
