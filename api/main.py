"""Food Calorie AI — الباك إند.

التشغيل محليًا:
    uvicorn api.main:app --reload --port 7860
وبعدين افتح http://localhost:7860
"""
import base64
import io
import os

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel

from src.portion import visualize
from src.predict import CalorieEstimator

# نفس الموقع اللي على GitHub Pages (بيشتغل في المتصفح، والـ API متاح جنبه)
WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")
MAX_UPLOAD_MB = 10
SIZES = {"small", "medium", "large"}

app = FastAPI(title="Food Calorie AI", version="1.0")
estimator = CalorieEstimator()  # بيتحمّل مرة واحدة لما السيرفر يقوم
table = estimator.calculator.table


def _clean_size(size: str | None) -> str | None:
    if size in (None, "", "auto"):
        return None
    if size not in SIZES:
        raise HTTPException(400, "size لازم يكون auto أو small أو medium أو large")
    return size


def _encode_jpeg(rgb) -> str:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else None


@app.get("/api/health")
def health():
    return {"status": "ok", "classes": len(table)}


@app.get("/api/foods")
def foods():
    """كل الأصناف اللي الموديل يعرفها (عشان المستخدم يصحّح لو الموديل غلط)."""
    return sorted(({"food": k, "name_ar": v["name_ar"]} for k, v in table.items()),
                  key=lambda f: f["name_ar"])


@app.post("/api/predict")
async def predict(image: UploadFile = File(...), size: str | None = Form(None),
                  grams: float | None = Form(None)):
    data = await image.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"الصورة أكبر من {MAX_UPLOAD_MB}MB")
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "الملف ده مش صورة")
    if grams is not None and not (1 <= grams <= 5000):
        raise HTTPException(400, "الوزن لازم يكون بين 1 و 5000 جرام")

    result, portion = estimator.estimate(img, size=_clean_size(size), grams=grams,
                                         return_portion=True)
    result["visualization"] = _encode_jpeg(visualize(portion)) if portion else None
    result["plate_coverage"] = round(portion.coverage, 3) if portion else None
    return result


class CaloriesIn(BaseModel):
    food: str
    size: str | None = None
    grams: float | None = None
    food_area_cm2: float | None = None


@app.post("/api/calories")
def calories(body: CaloriesIn):
    """إعادة الحساب لما المستخدم يختار الصنف الصح بنفسه."""
    if body.food not in table:
        raise HTTPException(404, "الصنف ده مش موجود")
    result = estimator.calculator.estimate([{"food": body.food, "confidence": 1.0}],
                                           size=_clean_size(body.size), grams=body.grams,
                                           food_area_cm2=body.food_area_cm2)
    result["confidence"] = None  # المستخدم اللي اختاره، مش الموديل
    return result


# الواجهة (لازم تتسجّل في الآخر عشان ما تغطيش على /api)
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
