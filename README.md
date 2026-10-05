# Food Calorie AI 🍽️

موديل ذكاء اصطناعي بيشوف صورة طبق أكل ويقدّر عدد السعرات الحرارية.

## الفكرة

1. **التعرف على الأكل:** موديل EfficientNet متدرب بـ Transfer Learning على Food-101.
2. **تقدير الكمية:** *(قريبًا)*
3. **حساب السعرات:** *(قريبًا)* من جدول تغذية لكل 100 جرام.

## هيكل المشروع

```
food-calorie-ai/
├── src/
│   ├── config.py     # كل الإعدادات
│   ├── data.py       # تحميل الداتا والـ transforms
│   ├── model.py      # بناء الموديل وتحميله
│   ├── train.py      # التدريب
│   └── predict.py    # التنبؤ على صورة
├── notebooks/
│   └── colab_runner.ipynb   # تشغيل المشروع على Colab
├── api/              # (قريبًا) FastAPI
├── data/nutrition/   # (قريبًا) جدول السعرات
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
python -m src.predict pizza.jpg      # التنبؤ
```

## خطة العمل

- [x] تجهيز البيئة واستكشاف الداتا
- [x] تدريب موديل التصنيف
- [ ] جدول السعرات وتقدير الكمية
- [ ] إضافة الأكل المصري
- [ ] API بـ FastAPI
- [ ] واجهة ويب + Deploy
