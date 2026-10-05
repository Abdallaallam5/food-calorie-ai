# Food Calorie AI 🍽️

موديل ذكاء اصطناعي بيشوف صورة طبق أكل ويقدّر عدد السعرات الحرارية.

## الفكرة

1. **التعرف على الأكل:** موديل EfficientNet متدرب بـ Transfer Learning على Food-101.
2. **تقدير الكمية:** الطبق بيتستخدم كمسطرة (قطره ~26 سم)، فبنقيس مساحة الأكل بالسم² ونحوّلها لجرامات حسب شكل الأكل.
3. **حساب السعرات:** من جدول تغذية لكل 100 جرام (`data/nutrition/food101_nutrition.csv`).

> القيم في الجدول تقريبية (متوسطات لوصفات شائعة)، ويُفضل مراجعتها مع [USDA FoodData Central](https://fdc.nal.usda.gov/).

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
│   └── predict.py    # صورة ← صنف ← سعرات
├── notebooks/
│   └── colab_runner.ipynb   # تشغيل المشروع على Colab
├── api/              # (قريبًا) FastAPI
├── data/nutrition/   # جدول السعرات للـ 101 صنف
└── requirements.txt
```

## التشغيل

### على Colab (مُوصى بيه، فيه GPU مجاني)
1. ارفع المشروع على GitHub.
2. افتح `notebooks/colab_runner.ipynb` على Colab وغيّر `REPO_URL`.
3. اختار T4 GPU وشغّل الخلايا بالترتيب.

### على جهازك
```bash
pip install -r requirements.txt
python -m src.train                  # التدريب
python -m src.predict pizza.jpg          # الكمية متقدّرة من الصورة
python -m src.predict pizza.jpg large    # أو حدد الحجم بنفسك
```

## خطة العمل

- [x] تجهيز البيئة واستكشاف الداتا
- [x] تدريب موديل التصنيف
- [x] جدول السعرات وحساب الكالوريز
- [x] تقدير الكمية من الصورة
- [ ] إضافة الأكل المصري
- [ ] API بـ FastAPI
- [ ] واجهة ويب + Deploy
