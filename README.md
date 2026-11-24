# פרויקט זיהוי זיופים קוליים – קוד

## מה יש בתיקייה

| קובץ | מה הוא עושה |
| --- | --- |
| `build_dataset.py` | בונה את מאגר האימון (1,520 קטעים של 4 שניות, 16kHz מונו) מהמאגרים הציבוריים, וכותב `data/train.csv` |
| `download_models.py` | מוריד את שני המודלים המאומנים מראש לגישה B (wav2vec 2.0, WavLM) |
| `stage1_data_analysis.ipynb` | שלב 1: סקירת הנתונים, Baseline ובדיקות קיצורי דרך |
| `features.py` | חילוץ פיצ'רים: MFCC, LFCC, log-spectrogram, ייצוגי SSL |
| `run_experiments.py` | כל הניסויים של תכנית שלב 2, כותב תוצאות ל-`results/` |

## שלב 1: לבנות את הנתונים (במחשב שלך, ב-Terminal)

```bash
cd ~/Documents/uni/HIT/deepfake-project
pip install numpy pandas soundfile soxr requests pyarrow huggingface_hub remotezip
python3 build_dataset.py --out data
python3 download_models.py
```

הסקריפט לא שומר את המאגרים המלאים (יותר מ-100GB), אלא רק את הקטעים שנבחרו (כ-200MB בסוף).
הוא קורא את הארכיונים ברצף ועוצר כשיש מספיק קטעים, ולכן התעבורה בפועל היא בערך 3–10GB, והריצה יכולה לקחת שעה–שעתיים.

| מקור | נלקח | מאיפה |
| --- | --- | --- |
| ASVspoof 5, dev | 300 אמיתי + 300 מזויף (A09–A16, חלוקה שווה) | Zenodo, רשומה 14498691 |
| CodecFake | 300 מזויף, מ-10 קודקים | Hugging Face: rogertseng/CodecFake |
| VCTK 0.92 | 300 אמיתי – ההקלטות המקוריות ש-CodecFake נוצר מהן | Edinburgh DataShare |
| M-AILABS, en_US | 320 אמיתי, עד 10 קטעים לספר | data.solak.de |

אם מקור אחד נכשל, אפשר להריץ רק אותו שוב, למשל: `python3 build_dataset.py --out data --only mailabs`.
אם הורדת ארכיון בדפדפן, אפשר לתת נתיב מקומי במקום כתובת (`--help` מראה איך).

## שלב 2: להריץ את הניסויים

```bash
pip install librosa scikit-learn torch transformers
python3 run_experiments.py --data data --ssl-models data/models/wav2vec2-base data/models/wavlm-base-plus
```

## החלטות בבניית הנתונים

- **אורך:** קטעים קצרים מ-3 שניות לא נלקחים, וקטעים ארוכים נחתכים בחלון אקראי של 4 שניות. אחרת, ריפוד ארוך באפסים היה רמז קל לזיהוי המאגר.
- **CodecFake ו-VCTK:** בתיאור הפרויקט ל-CodecFake יש 300 אמיתיים ו-300 מזויפים. האמיתיים הם הקלטות VCTK של אותם דוברים, ולכן ב-`train.csv` שניהם מסומנים `source=CodecFake`, ועמודת `corpus` שומרת את המקור האמיתי.
- **דובר:** עמודת `speaker` כוללת את שם המאגר (`ASVspoof5:D_0123`), כדי שדוברים ממאגרים שונים לא יתערבבו בפיצול.
- **מקורות:** ASVspoof 5, CodecFake ו-VCTK תחת CC BY 4.0. M-AILABS תחת הרישיון החופשי של M-AILABS.
