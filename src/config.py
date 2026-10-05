"""كل إعدادات المشروع في مكان واحد."""
import os

# المسارات
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(ROOT, "data"))
SAVE_DIR = os.environ.get("SAVE_DIR", os.path.join(ROOT, "checkpoints"))  # التدريب بيحفظ هنا
WEIGHTS_DIR = os.path.join(ROOT, "weights")  # الموديل المتدرب الجاهز اللي جوه الريبو

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
