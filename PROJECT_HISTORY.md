# היסטוריית הפרויקט וסיכום עבודה

פרויקט: סיווג EEG של דמיון תנועתי על בסיס BCI Competition IV Dataset 2a

תאריך עדכון: 2026-07-07

## מטרת הפרויקט

המטרה המרכזית הייתה לבנות מערכת למידת מכונה שמסווגת אותות EEG לפי דמיון תנועתי:

- דמיון תנועה של יד שמאל
- דמיון תנועה של יד ימין
- דמיון תנועה של שתי הרגליים
- דמיון תנועה של הלשון

בנוסף למטרה המחקרית של השוואת אלגוריתמים, רצינו להתקדם לכיוון מערכת BCI בזמן אמת: מערכת שבעתיד תוכל לקבל זרם EEG חי ולתרגם כוונה מנטלית לפקודת כיוון, למשל עבור שליטה בסמן.

## תיאור המאגר

השתמשנו במאגר BCI Competition IV Dataset 2a.

מבנה המאגר:

- 9 נבדקים: A01 עד A09
- לכל נבדק קובץ אימון מסומן: `A0XT.gdf`
- לכל נבדק קובץ הערכה: `A0XE.gdf`
- קיימים קבצי תוויות רשמיים עבור קבצי ההערכה בתיקייה `true_labels/`
- 22 ערוצי EEG ועוד 3 ערוצי EOG
- 4 מחלקות דמיון תנועתי

קודי האירועים המרכזיים:

```text
769 = left hand
770 = right hand
771 = both feet
772 = tongue
783 = unknown cue in evaluation files
```

בשלב ה-benchmark השתמשנו בפרוטוקול התחרות:

```text
train on A0XT.gdf
test on A0XE.gdf
score using true_labels/A0XE.mat
```

המדד המרכזי הוא Cohen's kappa, בדומה לתחרות המקורית. המדד הזה מתקן עבור הצלחה מקרית. בבעיה עם 4 מחלקות, ניחוש אקראי נותן בערך 25% accuracy, ולכן kappa מתאים יותר להשוואה בין מודלים.

## שאלות המחקר

במהלך העבודה התמקדנו בכמה שאלות:

1. האם אפשר לשחזר או לעבור את תוצאת הייחוס של הזוכה בתחרות 2008?
2. האם שיטות קלאסיות כמו CSP/FBCSP ו-Riemannian geometry עדיפות על רשתות נוירונים במאגר קטן יחסית?
3. האם מודלים מודרניים יותר כמו ShallowConvNet, EEG-TCNet או ResNet קצר משפרים ביצועים?
4. האם שינוי תחומי תדר, חלון זמן או מסווג אחרי tangent space יכול להעלות את kappa?
5. איזו גישה מתאימה יותר למערכת BCI אישית בזמן אמת?

## מבנה הקוד

נקודת הכניסה הראשית:

```text
pipeline.py
```

מודולים מרכזיים:

```text
eeg_project/data.py        טעינה, ניקוי בסיסי, epoching, שמירת dataset מוכן
eeg_project/decoders.py    CSP, FBCSP, Riemannian, וריאנטים ו-ensemble
eeg_project/cnn.py         מודלים נוירוניים ב-PyTorch
eeg_project/benchmark.py   פרוטוקול train-T/test-E הרשמי
eeg_project/cli.py         ממשק שורת פקודה
eeg_project/reporting.py   שמירת טבלאות, גרפים וסיכומים
eeg_project/live_demo.py   הכנה לדמו חי דרך LSL
eeg_project/cursor_demo.py דמו offline של שליטה בסמן
```

פקודות חשובות:

```powershell
.\.venv\Scripts\python.exe pipeline.py inspect BCICIV_2a_gdf/A01T.gdf
.\.venv\Scripts\python.exe pipeline.py prepare
.\.venv\Scripts\python.exe pipeline.py benchmark --models riemann fbcsp
.\.venv\Scripts\python.exe pipeline.py demo --subject A03 --model riemann
```

## עיבוד מקדים

העיבוד המקדים הבסיסי:

1. טעינת קובצי GDF עם MNE
2. סימון ערוצי EOG
3. שמירת 22 ערוצי EEG בלבד כקלט למודלים
4. סינון בתחום 8-30 Hz, בהתאם לתיאור Dataset 2a ולתחומי mu/beta
5. resampling ל-125 Hz
6. חיתוך epochs מ-0.5 עד 4.0 שניות אחרי הופעת cue
7. שמירת מטריצה במבנה:

```text
trials x channels x samples
```

בדקנו גם תיקון EOG ברגרסיה, אך הוא כמעט לא שיפר את התוצאות ולכן נשאר אופציונלי ולא ברירת מחדל.

## אלגוריתמים שנבדקו

### מודלים קלאסיים

- CSP + Logistic Regression
- CSP + SVM
- CSP + Random Forest
- CSP + LDA
- FBCSP + Mutual Information feature selection + LDA
- Riemannian tangent-space decoder

### מודלים נוירוניים

- CNN קטן בסגנון EEGNet
- Raw short ResNet עם temporal dilation
- ShallowConvNet
- EEG-TCNet

### שדרוגים ו-ablation

בדקנו גם:

- `riemann_wide`: שימוש בבנק תדרים רחב יותר, 4-40 Hz
- `riemann_lr`: tangent-space features עם Logistic Regression
- `riemann_wide_lr`: בנק תדרים רחב + Logistic Regression
- `riemann_fbcsp_vote`: soft-vote ensemble של Riemannian + FBCSP
- חלונות זמן שונים:

```text
0.5-4.0s
1.0-4.0s
0.5-3.5s
1.0-3.5s
```

על A01, חלון הזמן המקורי `0.5-4.0s` נשאר הטוב ביותר.

## תוצאות מרכזיות

תוצאת הייחוס של הזוכה בתחרות 2008, FBCSP של Ang et al., היא בערך:

```text
mean kappa ~= 0.57
```

### השוואת שלושת המודלים המרכזיים

תוצאות train-T/test-E על כל 9 הנבדקים:

```text
model             mean accuracy   mean kappa
riemann           0.712           0.616
fbcsp             0.675           0.566
shallow_convnet   0.627           0.503
```

מסקנה: Riemannian היה המודל היחיד שעבר בבירור את תוצאת הייחוס של 2008.

### תוצאת ensemble

לאחר מכן יצרנו ensemble מסוג soft-vote בין Riemannian לבין FBCSP:

```text
model                 mean accuracy   mean kappa
riemann_fbcsp_vote    0.735           0.647
```

זוהי התוצאה הטובה ביותר בפרויקט עד כה.

קובץ תוצאות:

```text
results/benchmark_all_subjects_riemann_fbcsp_vote.csv
```

תוצאות לפי נבדק:

```text
subject   accuracy   kappa
A01       0.865      0.819
A02       0.562      0.417
A03       0.858      0.810
A04       0.736      0.648
A05       0.590      0.454
A06       0.552      0.403
A07       0.816      0.755
A08       0.837      0.782
A09       0.799      0.731
```

### וריאנט Riemannian רחב

בדקנו גם `riemann_wide_lr`:

```text
model             mean accuracy   mean kappa
riemann_wide_lr   0.709           0.612
```

הוא שיפר חלק מהנבדקים אך לא שיפר את הממוצע לעומת Riemannian המקורי. לכן הוא מעניין כאבלציה, אך לא נבחר כמודל הסופי.

## מה עבד טוב

### Riemannian tangent space

המודל הזה עבד טוב במיוחד כי הוא מתאים מאוד ל-EEG עם מעט דוגמאות. במקום ללמוד מיליוני פרמטרים, הוא משתמש במבנה covariance בין ערוצים וממפה אותו ל-tangent space. זה מתאים לדמיון תנועתי, שבו המידע החשוב הוא שינויי כוח ותבניות מרחביות מעל הקורטקס המוטורי.

### FBCSP

FBCSP עבד טוב וחזר כמעט בדיוק לאזור תוצאת הזוכה המקורי של 2008. זה הגיוני כי הוא נבנה במיוחד למטלה הזאת: בחירת תחומי תדר, CSP בכל תחום, ובחירת פיצ'רים.

### Ensemble של Riemannian + FBCSP

זה היה השיפור המשמעותי ביותר. שני המודלים עושים טעויות שונות, ולכן soft-vote ביניהם העלה את הממוצע:

```text
riemann kappa           0.616
fbcsp kappa             0.566
riemann_fbcsp_vote      0.647
```

## מה עבד פחות טוב

### מודלים נוירוניים

הוספנו כמה מודלים נוירוניים:

- CNN קטן
- Raw ResNet
- ShallowConvNet
- EEG-TCNet

ShallowConvNet היה הטוב מביניהם, אך עדיין לא עבר את השיטות הקלאסיות בממוצע. EEG-TCNet עבד טכנית, אבל בגרסה הלא מכווננת שלו על A01 קיבל:

```text
eeg_tcnet on A01: accuracy=0.608, kappa=0.477
```

הסיבה הסבירה: Dataset 2a קטן יחסית לרשתות נוירונים. לכל נבדק יש רק 288 trials באימון, כלומר בערך 72 trials לכל מחלקה. זה מספיק לשיטות קלאסיות יעילות, אבל מעט עבור deep learning.

### חלונות זמן אחרים

חשבנו שאולי שינוי חלון הזמן ישפר את הביצועים, אבל על A01 החלון המקורי היה עדיף:

```text
0.5-4.0s  best
1.0-4.0s  worse
0.5-3.5s  worse
1.0-3.5s  worse
```

## אתגרים ואיך התגברנו עליהם

### חוסר נתונים עבור deep learning

האתגר המרכזי היה שמאגר Dataset 2a אינו גדול מספיק כדי להבטיח יתרון לרשתות נוירונים. ניסינו להוסיף מודלים מודרניים, אבל התוצאות הראו ששיטות קלאסיות עדיין חזקות יותר במצב low-data.

הפתרון: לא להכריח deep learning לנצח, אלא לבצע השוואה הוגנת ולהציג את המסקנה: במאגר קטן ומובנה, Riemannian/FBCSP עדיפים.

### פרוטוקול הערכה

בתחילת העבודה היו גם ניסויים של train/test split פנימי על קובצי T. בהמשך עברנו לפרוטוקול אמין יותר:

```text
train on T
test on E
official labels
mean kappa across 9 subjects
```

זה מאפשר השוואה ישירה לתחרות המקורית.

### בעיות זיכרון בהרצות ארוכות

כאשר הרצנו את כל הנבדקים ברצף, היו תקלות זיכרון עם MNE/OpenBLAS. התגברנו על זה בעזרת:

- הרצה של כל נבדק בתהליך Python חדש
- הגבלת thread count:

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
```

### בחירת מדד

Accuracy לבדו לא מספיק, כי בבעיה עם 4 מחלקות יש הצלחה מקרית של כ-25%. לכן השתמשנו ב-Cohen's kappa, המדד המקובל בתחרות.

## קשר למערכת BCI בזמן אמת

הפרויקט כולל גם כיוון מעשי למערכת live:

- `cursor_demo.py`: דמו offline שבו פלט המודל מזיז סמן
- `live_demo.py`: חיבור לזרם EEG חי דרך LSL
- `calibrate-gui`: ממשק כיול עם cues דמויי חיצים, בדומה לפרוטוקול motor imagery
- שימוש בחלונות החלקה בזמן אמת
- מיפוי:

```text
left hand  -> move cursor left
right hand -> move cursor right
feet       -> move cursor down
tongue     -> move cursor up
```

למערכת אישית אמיתית נדרש שלב calibration אישי:

1. הקלטת EEG של המשתמש עם cues מסומנים
2. אימון decoder על אותו משתמש ואותה קסדה
3. הוספת מחלקת rest/no-control
4. החלקת תחזיות בזמן אמת כדי למנוע קפיצות

כרגע המועמד הטוב ביותר ל-live BCI הוא `riemann_fbcsp_vote`, ואם רוצים מערכת פשוטה ומהירה יותר אז `riemann`.

לאחר עדכון נוסף הוספנו גם workflow מעשי לכיול אישי:

```text
calibrate-gui    -> הצגת cues ויזואליים והקלטת EEG מסומן
calibrate-train  -> אימון decoder אישי ושמירת joblib
live-demo        -> טעינת המודל האישי עם החלקת הסתברויות וסף ביטחון
```

מחלקת `rest` נשמרת כברירת מחדל, ולכן המערכת יכולה ללמוד גם מצב של "אין פקודה" ולא רק ארבע פקודות תנועה.

## מסקנות

המסקנה המרכזית היא שבמאגר BCI IV 2a, שיטות קלאסיות מותאמות EEG עדיין חזקות מאוד. רשתות נוירונים הן כיוון מעניין, אבל בתנאי low-data הן לא בהכרח מנצחות.

התוצאה הטובה ביותר שלנו:

```text
riemann_fbcsp_vote
mean accuracy = 0.735
mean kappa    = 0.647
```

זה גבוה מתוצאת הייחוס של הזוכה המקורי בתחרות:

```text
2008 FBCSP winner kappa ~= 0.57
```

לכן הפרויקט מראה גם שחזור של baseline קלאסי חזק, גם שיפור בעזרת Riemannian geometry, וגם שיפור נוסף בעזרת ensemble.

## קבצים חשובים להגשה

```text
README.md
PROJECT_HISTORY.md
pipeline.py
eeg_project/
results/benchmark_all_subjects_riemann_fbcsp_vote.csv
results/benchmark_all_subjects_riemann_fbcsp_shallow_convnet.csv
results/benchmark_all_subjects_riemann_wide_lr.csv
results/benchmark_kappa.png
results/benchmark_summary.md
```

## המשך אפשרי

אם נמשיך לפתח את הפרויקט:

1. לאסוף calibration אישי ל-EEG חי
2. להוסיף מחלקת rest
3. לבצע subject-specific model selection
4. לבדוק transfer learning ממאגר PhysioNet Motor Movement/Imagery
5. לשפר את ה-live demo לממשק אינטראקטיבי מלא
