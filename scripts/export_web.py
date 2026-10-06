"""يجهّز ملفات الموقع اللي بيشتغل في المتصفح (docs/).

التشغيل:
    python -m scripts.export_web
بيعمل:
    docs/model/model.onnx   ← الموديل بصيغة ONNX
    docs/model/foods.json   ← أسماء الأصناف + جدول السعرات + ثوابت الحساب
لازم يتشغّل تاني بعد أي تدريب جديد أو تعديل في جدول السعرات.
"""
import json
import os

import numpy as np
import onnxruntime as ort
import torch

from src import config as C
from src import nutrition as N
from src import portion as P
from src.model import load_trained

OUT = os.path.join(C.ROOT, "docs", "model")


def export_onnx(path: str):
    model, classes = load_trained(C.WEIGHTS_DIR, "cpu")
    dummy = torch.randn(1, 3, C.IMG_SIZE, C.IMG_SIZE)
    torch.onnx.export(model, dummy, path, input_names=["image"], output_names=["logits"],
                      opset_version=17, dynamo=False)

    # نتأكد إن ONNX بيطلع نفس نتيجة PyTorch
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    x = torch.randn(4, 3, C.IMG_SIZE, C.IMG_SIZE)
    with torch.no_grad():
        ref = torch.cat([model(x[i:i + 1]) for i in range(4)]).numpy()
    got = np.concatenate([sess.run(None, {"image": x[i:i + 1].numpy()})[0] for i in range(4)])
    diff = np.abs(ref - got).max()
    assert diff < 1e-3 and (ref.argmax(1) == got.argmax(1)).all(), f"ONNX mismatch: {diff}"
    print(f"✅ model.onnx ({os.path.getsize(path) / 1e6:.1f} MB) | max diff vs PyTorch: {diff:.2e}")
    return classes


def export_foods(path: str, classes: list[str]):
    table = N.load_table()
    data = {
        "classes": classes,
        "foods": table,
        "config": {
            "img_size": C.IMG_SIZE, "resize": 256, "mean": C.MEAN, "std": C.STD,
            "size_multiplier": N.SIZE_MULTIPLIER, "grams_per_cm2": N.GRAMS_PER_CM2,
            "min_factor": N.MIN_FACTOR, "max_factor": N.MAX_FACTOR,
            "confidence_threshold": N.CONFIDENCE_THRESHOLD,
            "min_alternative_conf": N.MIN_ALTERNATIVE_CONF,
            "error_margin": N.ERROR_MARGIN, "error_margin_image": N.ERROR_MARGIN_IMAGE,
            "portion": {
                "plate_diameter_cm": P.PLATE_DIAMETER_CM, "max_side": P.MAX_SIDE,
                "color_diff_threshold": P.COLOR_DIFF_THRESHOLD,
                "min_coverage": P.MIN_COVERAGE, "max_coverage": P.MAX_COVERAGE,
                "min_rim_uniformity": P.MIN_RIM_UNIFORMITY,
            },
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    print(f"✅ foods.json ({len(table)} أصناف)")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    classes = export_onnx(os.path.join(OUT, "model.onnx"))
    export_foods(os.path.join(OUT, "foods.json"), classes)
