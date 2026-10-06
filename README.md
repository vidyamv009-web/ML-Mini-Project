# Cardiovascular Disease (CVD) Prediction Using Machine Learning

-----------------------------------------------------------------------------------------------------------

**Name:** Vidya  
**Organization:** Entri Elevate  
**Date:** 06-Oct-2026  
**Notebook:** `Cardiovascular_Disease_Prediction.ipynb`  
**Problem type:** Binary Classification  
**Best model:** Gradient Boosting Classifier  

-----------------------------------------------------------------------------------------------------------

## 1. Overview of Problem Statement

Cardiovascular disease (CVD) is one of the leading causes of death worldwide. Early identification of people at higher risk can support timely medical checks and lifestyle changes. Predicting CVD from routine clinical measurements is challenging because several factors such as age, blood pressure, cholesterol, glucose, body mass index, and lifestyle habits interact in complex ways. Therefore, developing an effective machine learning classification model can help flag higher-risk profiles and support preventive healthcare decisions. This model is decision support tool, not as a medical diagnosis system.

## 2. Objective

To develop the best binary classification model that predicts the presence or absence of cardiovascular disease (`cardio` = 0 or 1) using patient clinical and lifestyle features, and to save the final model for reuse (including web-app inference).

## 3. Data Description

- **Source:** Kaggle – [Cardio Vascular Disease Detection](https://www.kaggle.com/datasets/bhadaneeraj/cardio-vascular-disease-detection) (`cardio_train.csv`)
- **Raw shape used in this project:** 69,301 rows × 13 columns
- **Target column:** `cardio` (0 = no CVD, 1 = CVD present)
- **Task:** Binary Classification

### Features (original dataset)

| Feature | Description |
|---|---|
| `id` | Unique patient identifier (dropped; not predictive) |
| `age` | Age in days (converted to years) |
| `gender` | Gender code (1 = female, 2 = male in this dataset coding) |
| `height` | Height in cm (used to create BMI, then dropped) |
| `weight` | Weight in kg (used to create BMI, then dropped) |
| `ap_hi` | Systolic blood pressure |
| `ap_lo` | Diastolic blood pressure |
| `cholesterol` | 1 = normal, 2 = above normal, 3 = well above normal |
| `gluc` | 1 = normal, 2 = above normal, 3 = well above normal |
| `smoke` | Smoking status (0/1) |
| `alco` | Alcohol intake (0/1) |
| `active` | Physical activity (0/1) |
| `cardio` | Target – presence of cardiovascular disease (0/1) |

### Final modeling features (after engineering)

`age`, `gender`, `ap_hi`, `ap_lo`, `cholesterol`, `gluc`, `smoke`, `alco`, `active`, `bmi`

## 4. Data Collection

- Imported `cardio_train.csv` with semicolon separator (`sep=';'`).
- Inspected shape, column names, dtypes, missing values, and sample rows.
- Confirmed target balance was nearly equal (~50% / 50%).

## 5. Data Preprocessing – Data Cleaning

- Dropped non-predictive `id` column.
- Converted `age` from days to years (observed range about 30–65 years).
- Removed invalid records:
  - unrealistic blood pressure, height, and weight
  - cases where systolic BP was not greater than diastolic BP  
  - 1,482 rows removed in this filter step
- Checked missing values after cleaning (nothing).
- Removed duplicate clinical profiles after dropping `id` (3,762 duplicates removed).
- Final cleaned working set before modeling filters: 64,057 rows.

## 6. Exploratory Data Analysis (EDA)

Gained insights into distributions and relationships with CVD using:

- Histogram  
- Boxplot  
- Count Plot  
- Heatmap Correlation  
- Pie Diagram  
- Kernel Density Estimation (KDE)

### Key EDA findings

- Strongest correlations with `cardio`: ap_hi, ap_lo, age, cholesterol, then weight/BMI.
- Lifestyle fields (`smoke`, `alco`) were relatively weak in this dataset.
- Target remained roughly balanced after cleaning.

## 7. Feature Engineering

- Created BMI = weight / (height/100)².
- Briefly computed pulse pressure for inspection, then avoided keeping redundant BP combinations.
- Filtered extreme BMI values.
- Dropped redundant columns after BMI creation (`height`, `weight`, and pulse pressure when used as a linear duplicate of BP).
- Final modeling shape: 64,038 rows × 11 columns (10 features + target).

## 8. Feature Selection

- Used SelectKBest (ANOVA F-test) and Random Forest feature importance.
- Exploratory ranking on full data was treated as insight only.
- Final ranking for modeling decisions used training data only (to avoid test leakage).
- Top contributors: ap_hi, ap_lo, age, cholesterol, bmi.
- Weaker contributors: gender, alcohol, smoking.
- Kept the clinically relevant original fields for the final model.

## 9. Split Data into Training and Testing Sets

- Stratified 80/20 train-test split (`random_state=42`).
- Training set: 51,230 rows  
- Testing set: 12,808 rows  
- Class ratio preserved in both sets (~51% CVD / ~49% no CVD).

## 10. Feature Scaling

- Applied StandardScaler so numerical features have comparable magnitude.
- Important for Logistic Regression, SVM, KNN, etc.
- Final deployment used a scikit-learn Pipeline so scaling is applied automatically at prediction time.

-----------------------------------------------------------------------------------------------------------

## 11. Build the ML Model (at least 5)

Implemented 8 classification algorithms:

1. Logistic Regression  
2. Decision Tree  
3. Random Forest  
4. K-Nearest Neighbors  
5. Support Vector Machine (SVM)  
6. Naive Bayes  
7. Gradient Boosting  
8. AdaBoost  

Best model by F1-Score: Gradient Boosting Classifier

## 12. Model Evaluation

Used classification metrics:

- Confusion Matrix  
- Accuracy  
- Precision  
- Recall  
- F1-Score  
- ROC Curve / AUC  

### Best untuned model (Gradient Boosting) – test set

| Metric | Approx. result |
|---|---|
| Accuracy | ~0.73 |
| Precision (CVD) | ~0.74 |
| Recall (CVD) | ~0.70 |
| F1-Score | 0.7226 |
| Overall report accuracy | 0.73 |

## 13. Hyperparameter Tuning and Pipeline

- Built Pipeline: `StandardScaler` + `GradientBoostingClassifier`
- Tuned with `GridSearchCV` (3-fold CV, scoring = F1)
- Best parameters found:  
  `learning_rate=0.1`, `max_depth=2`, `n_estimators=150`
- Best CV F1: 0.7289

### Tuned Gradient Boosting – test set

| Metric | Result |
|---|---|
| Accuracy | 0.7239 |
| Precision | 0.7421 |
| Recall | 0.7011 |
| F1-Score | 0.7210 |
| ROC-AUC | 0.7879 |

Final selected model: untuned Gradient Boosting pipeline (slightly higher test F1 than the tuned grid search result in this run).

## 14. Save the Model

Saved artifacts with `joblib`:

- `cvd_prediction_model.joblib` – final Pipeline (scaler + classifier)
- `cvd_feature_columns.joblib` – expected inference column order  

Expected columns:  
`['age', 'gender', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'bmi']`

## 15. Test with Unseen Data

Tested two synthetic patient profiles:

| Profile | Key traits | Prediction | CVD probability |
|---|---|---|---|
| Patient 1 | Age 55, higher BP (150/95), high cholesterol, smokes, not active | CVD Risk (Positive) | 0.890 |
| Patient 2 | Age 35, normal BP (115/75), normal labs, active | No CVD Risk (Negative) | 0.084 |

Results are clinically directionally sensible and confirm the saved model loads and predicts correctly.

## 16. Interpretation of Results (Conclusion)

- An end-to-end binary CVD classification pipeline was built successfully.
- Gradient Boosting was the best performing model among the eight classifiers compared.
- Blood pressure, age, cholesterol, and BMI were the strongest predictive signals.
- About 72–73% accuracy and ROC-AUC ≈ 0.79 are realistic for this public tabular dataset with routine check-up features.
- Limitations: self-reported lifestyle fields may be noisy; dataset may not generalize to all populations; no ECG/family history; model is a screening aid, not a clinical diagnosis tool.

## 17. Future Work

- Recalibrate predicted probabilities for more interpretable risk scores.
- Add richer clinical features if available (family history, ECG, smoking duration, diet).
- Deploy/maintain the Flask web demo (`cvd_webapp`) for model loading using docker.

-----------------------------------------------------------------------------------------------------------

## How to run the notebook

1. Place `cardio_train.csv` in the same folder as the notebook.  
2. Open `Cardiovascular_Disease_Prediction.ipynb`.  
3. Run all cells from top to bottom.  
4. After training, `cvd_prediction_model.joblib` and `cvd_feature_columns.joblib` will be saved.

## Project files (related)

| File | Purpose |
|---|---|
| `Cardiovascular_Disease_Prediction.ipynb` | Full ML workflow |
| `cardio_train.csv` | Dataset |
| `cvd_prediction_model.joblib` | Saved final model pipeline |
| `cvd_feature_columns.joblib` | Feature column list for inference |
| `cvd_webapp/` | Flask UI for live prediction demo |

-----------------------------------------------------------------------------------------------------------

