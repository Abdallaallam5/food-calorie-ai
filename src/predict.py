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


if __name__ == "__main__":
    clf = FoodClassifier()
    for r in clf.predict(Image.open(sys.argv[1])):
        print(f"{r['food']:25s} {r['confidence']:.1%}")
