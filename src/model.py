"""بناء الموديل وتحميله."""
import json
import os

import timm
import torch

from . import config as C


def build_model(num_classes: int, pretrained: bool = True):
    """EfficientNet متدرب على ImageNet، وآخر طبقة بعدد أصناف الأكل (Transfer Learning)."""
    return timm.create_model(C.MODEL_NAME, pretrained=pretrained, num_classes=num_classes)


def load_trained(save_dir: str = C.SAVE_DIR, device: str = "cpu"):
    """يحمّل أحسن موديل متدرب وأسماء الأصناف."""
    with open(os.path.join(save_dir, "classes.json")) as f:
        classes = json.load(f)
    model = build_model(len(classes), pretrained=False)
    state = torch.load(os.path.join(save_dir, "best_model.pth"), map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval(), classes
