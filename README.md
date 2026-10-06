# Food Calorie AI 🍽️

موديل ذكاء اصطناعي بيشوف صورة طبق أكل ويقدّر **نوعه** و**كميته** و**سعراته الحرارية**.

## 🌐 الموقع

**https://abdallaallam5-food-calorie-ai.hf.space**

ارفع صورة أو صوّر الطبق من موبايلك، وهتاخد نوع الأكل وكميته وسعراته. ولو الموديل غلط في الصنف، تقدر تختار الصح والسعرات تتحسب تاني.

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
├── api/main.py       # الباك إند (FastAPI)
├── web/index.html    # واجهة الموقع
├── weights/          # الموديل المتدرب الجاهز
├── notebooks/
│   └── colab_runner.ipynb   # تجربة + تدريب على Colab
├── data/nutrition/   # جدول السعرات للـ 101 صنف
├── Dockerfile        # تشغيل الموقع على أي سيرفر
└── .github/workflows/deploy-hf.yml  # نشر تلقائي على Hugging Face
```

## التشغيل على جهازك

```bash
pip install -r requirements.txt fastapi "uvicorn[standard]" python-multipart
uvicorn api.main:app --port 7860         # الموقع على http://localhost:7860
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

## النشر

كل push على `main` بيتنشر لوحده على Hugging Face Spaces عن طريق GitHub Actions. محتاج بس secret اسمه `HF_TOKEN` في إعدادات الريبو.

الـ `Dockerfile` كمان بيشتغل على أي منصة بتدعم Docker (Railway, Render, Fly.io...) بشرط يكون فيها 1GB رامات على الأقل.

## خطة العمل

- [x] تدريب موديل التصنيف
- [x] جدول السعرات وحساب الكالوريز
- [x] تقدير الكمية من الصورة
- [ ] إضافة الأكل المصري
- [x] API بـ FastAPI
- [x] واجهة ويب + Deploy
