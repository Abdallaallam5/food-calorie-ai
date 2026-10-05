"""كل إعدادات المشروع في مكان واحد."""
import os

# المسارات
DATA_DIR = os.environ.get("DATA_DIR", "./data")
SAVE_DIR = os.environ.get("SAVE_DIR", "./checkpoints")

# الموديل
MODEL_NAME = "efficientnet_b0"
IMG_SIZE = 224

# التدريب
BATCH_SIZE = 64
EPOCHS = 6
LR = 1e-3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
NUM_WORKERS = 2

# Normalization بتاع ImageNet
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
