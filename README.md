# Food Calorie AI 🍽️

موديل ذكاء اصطناعي بيشوف صورة طبق أكل ويقدّر **نوعه** و**كميته** و**سعراته الحرارية**.

## 🌐 الموقع

**https://abdallaallam5.github.io/food-calorie-ai/**

ارفع صورة أو صوّر الطبق من موبايلك، وهتاخد نوع الأكل وكميته وسعراته. ولو الموديل غلط في الصنف، تقدر تختار الصح والسعرات تتحسب تاني.

**الموديل بيشتغل جوه المتصفح نفسه** (ONNX Runtime Web + OpenCV.js)، فالصورة مش بتتبعت لأي سيرفر، والموقع static ومجاني بالكامل على GitHub Pages. نتايج المتصفح متطابقة مع نسخة بايثون بالظبط.

## 🧪 أو جرّبه على Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Abdallaallam5/food-calorie-ai/blob/main/notebooks/colab_runner.ipynb)

1. اضغط على زرار **Open in Colab**.
2. شغّل أول خليتين في **الجزء الأول** (`Shift + Enter`).
3. ارفع صورة طبق أكل متصوّر من فوق.

مش محتاج GPU ولا تدريب، لأن الموديل المتدرب موجود في الريبو (`weights/`).

### شكل النتيجة

```
🍽️  بيتزا (pizza) — ثقة 91%
⚖️  الكمية: ~250 جرام (متقدّرة من الصورة، مساحة الأكل ~280 سم²)
🔥 السعرات: ~675 kcal  (بين 473 و 878)
```

## إزاي بيشتغل

1. **التعرف على الأكل:** موديل EfficientNet-B0 متدرب بـ Transfer Learning على [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) (101 صنف، 101 ألف صورة).
2. **تقدير الكمية:** الطبق بيتستخدم كمسطرة (قطره ~26 سم)، فبنقيس مساحة الأكل بالسم² ونحوّلها لجرامات حسب شكل الأكل. لو مفيش طبق واضح، بنستخدم الحصة المعتادة.
3. **حساب السعرات:** من جدول تغذية لكل 100 جرام (`data/nutrition/food101_nutrition.csv`).

النتيجة بتطلع **مدى** (±25–30%) مش رقم واحد بس، لأن الزيت والسمنة والوصفات بتختلف ومش باينة في الصورة.

> القيم في جدول التغذية تقريبية (متوسطات لوصفات شائعة)، ويُفضل مراجعتها مع [USDA FoodData Central](https://fdc.nal.usda.gov/).

### الحدود الحالية
- بيتعرف على الـ 101 صنف الموجودين في Food-101 بس، والأكل المصري لسه مش مضاف.
- بيتعامل مع الطبق كأنه صنف واحد.
- تقدير الكمية بيفترض طبق مقاسه عادي ومتصوّر من فوق.

## هيكل المشروع

```
food-calorie-ai/
├── src/
│   ├── config.py     # كل الإعدادات
│   ├── data.py       # تحميل الداتا والـ transforms
│   ├── model.py      # بناء الموديل وتحميله
│   ├── train.py      # التدريب
│   ├── portion.py    # تقدير مساحة الأكل من الصورة
│   ├── nutrition.py  # حساب السعرات من جدول التغذية
│   └── predict.py    # صورة ← صنف ← كمية ← سعرات
├── docs/             # الموقع (GitHub Pages)
│   ├── index.html, app.js   # الواجهة
│   ├── engine.js     # نفس منطق src/ بس في المتصفح
│   └── model/        # model.onnx + جدول السعرات
├── scripts/export_web.py    # بيحوّل الموديل لـ ONNX ويجهّز docs/model
├── api/main.py       # API بـ FastAPI (اختياري، لو عايز سيرفر)
├── weights/          # الموديل المتدرب الجاهز
├── notebooks/
│   └── colab_runner.ipynb   # تجربة + تدريب على Colab
├── data/nutrition/   # جدول السعرات للـ 101 صنف
└── Dockerfile        # تشغيل الـ API على أي سيرفر
```

## التشغيل على جهازك

```bash
pip install -r requirements.txt fastapi "uvicorn[standard]" python-multipart
python -m http.server -d docs 8000       # الموقع بس على http://localhost:8000
uvicorn api.main:app --port 7860         # أو الموقع + API على http://localhost:7860
python -m src.predict pizza.jpg          # الكمية متقدّرة من الصورة
python -m src.predict pizza.jpg large    # أو حدد الحجم بنفسك
python -m src.train                      # تدريب من الأول (محتاج GPU)
```

## الـ API

| Endpoint | الوظيفة |
|---|---|
| `POST /api/predict` | بياخد `image` (و`size` أو `grams` اختياري) ويرجّع الصنف والكمية والسعرات |
| `POST /api/calories` | إعادة الحساب لصنف يختاره المستخدم: `{"food": "pizza", "food_area_cm2": 200}` |
| `GET /api/foods` | كل الأصناف اللي الموديل يعرفها |
| `GET /docs` | توثيق تفاعلي (Swagger) |

> الـ API ده للي عايز يستخدم الموديل من تطبيق تاني (موبايل مثلًا). الموقع نفسه مش محتاجه.

## النشر

الموقع بيتنشر من فولدر `docs/` على GitHub Pages (Settings ← Pages ← Branch: `main` / `docs`)، وأي push على `main` بيحدّثه لوحده.

**بعد أي تدريب جديد أو تعديل في جدول السعرات**، شغّل:
```bash
python -m scripts.export_web   # بيعمل docs/model/model.onnx و foods.json ويتأكد إن ONNX مطابق لـ PyTorch
```

الـ API (`api/main.py`) اختياري، ولو عايز تشغّله على سيرفر فيه `Dockerfile` جاهز (محتاج 1GB رامات على الأقل).

## خطة العمل

- [x] تدريب موديل التصنيف
- [x] جدول السعرات وحساب الكالوريز
- [x] تقدير الكمية من الصورة
- [ ] إضافة الأكل المصري
- [x] API بـ FastAPI
- [x] واجهة ويب + Deploy
