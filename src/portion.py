"""تقدير مساحة الأكل من الصورة باستخدام الطبق كمسطرة.

الفكرة:
1. نلاقي دايرة الطبق في الصورة (Hough Circles).
2. نعرف لون الطبق من الحلقة اللي على حافته، وأي بكسل جوه الطبق لونه مختلف يبقى أكل.
3. الطبق العادي قطره ~26 سم، فنعرف كل بكسل يساوي كام سم²، ونحسب مساحة الأكل الحقيقية.

لو مفيش طبق باين أو النتيجة مش منطقية، بنرجع None والحساب بيستخدم الحصة المعتادة.
"""
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

PLATE_DIAMETER_CM = 26.0
MAX_SIDE = 512            # بنصغّر الصورة عشان السرعة
COLOR_DIFF_THRESHOLD = 22  # قد إيه لازم لون البكسل يختلف عن الطبق عشان يتحسب أكل (LAB)
MIN_COVERAGE, MAX_COVERAGE = 0.05, 0.95
MIN_RIM_UNIFORMITY = 0.5   # أقل نسبة من حافة الطبق لازم تكون بلون الطبق


@dataclass
class PortionResult:
    food_area_cm2: float
    coverage: float           # نسبة الأكل من مساحة الطبق
    plate_circle: tuple       # (x, y, r) على الصورة المصغّرة
    food_mask: np.ndarray     # ماسك الأكل على الصورة المصغّرة
    image: np.ndarray         # الصورة المصغّرة (RGB)


def _resize(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    s = MAX_SIDE / max(h, w)
    return cv2.resize(img, (int(w * s), int(h * s))) if s < 1 else img


def find_plate(rgb: np.ndarray):
    """يرجّع أكبر دايرة منطقية (x, y, r) أو None."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 5)
    short = min(gray.shape)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=short,
                               param1=100, param2=40,
                               minRadius=int(short * 0.25), maxRadius=int(short * 0.62))
    if circles is None:
        return None
    x, y, r = max(np.round(circles[0]).astype(int), key=lambda c: c[2])
    return int(x), int(y), int(r)


def segment_food(rgb: np.ndarray, circle) -> np.ndarray | None:
    """ماسك للبكسلات اللي لونها مختلف عن لون الطبق جوه الدايرة، أو None لو ده مش طبق."""
    x, y, r = circle
    h, w = rgb.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
    inside = dist < r * 0.97
    rim = (dist > r * 0.90) & (dist < r * 0.98)

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    plate_color = np.median(lab[rim], axis=0)   # الميديان بيتجاهل الأكل اللي داخل على الحافة
    diff = np.linalg.norm(lab - plate_color, axis=2)

    # تأكيد إنه طبق فعلًا: أغلب حافته لازم تكون لون واحد تقريبًا
    if (diff[rim] < COLOR_DIFF_THRESHOLD).mean() < MIN_RIM_UNIFORMITY:
        return None

    mask = ((diff > COLOR_DIFF_THRESHOLD) & inside).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # نشيل النقط الصغيرة
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # نسد الفراغات جوه الأكل
    return mask.astype(bool)


def estimate_food_area(image: Image.Image) -> PortionResult | None:
    rgb = _resize(np.array(image.convert("RGB")))
    circle = find_plate(rgb)
    if circle is None:
        return None

    mask = segment_food(rgb, circle)
    if mask is None:
        return None
    r = circle[2]
    plate_px = np.pi * r ** 2
    coverage = mask.sum() / plate_px
    if not (MIN_COVERAGE <= coverage <= MAX_COVERAGE):
        return None

    cm2_per_px = (np.pi * (PLATE_DIAMETER_CM / 2) ** 2) / plate_px
    return PortionResult(food_area_cm2=float(mask.sum() * cm2_per_px), coverage=float(coverage),
                         plate_circle=circle, food_mask=mask, image=rgb)


def visualize(res: PortionResult) -> np.ndarray:
    """صورة توضيحية: الطبق بالأخضر والأكل متلوّن بالبرتقالي."""
    out = res.image.copy()
    out[res.food_mask] = (0.5 * out[res.food_mask] + 0.5 * np.array([255, 140, 0])).astype(np.uint8)
    x, y, r = res.plate_circle
    cv2.circle(out, (x, y), r, (0, 200, 0), 3)
    return out
