# AI-and-ML-based-model-prediction
Built and deployed production ML models for health outcome prediction and real estate pricing — from raw data to live inference pipelines using Scikit-learn, TensorFlow, and Power BI.

# NeuralEdge — AI & Data Science Internship Projects

> AI & Data Science Internship · Elewayte · Sep–Nov 2025

---

## Overview

Two production ML models built and deployed during my internship at Elewayte — a **Health Outcome Predictor** (classification) and a **House Price Estimator** (regression). Both were developed end-to-end: from exploratory data analysis and feature engineering through training, hyperparameter tuning, evaluation, and live pipeline integration.

---

## Models

### Model 1 — Health Outcome Predictor

Predicts patient health outcomes (high risk / low risk) using clinical features. Combines a Random Forest baseline with a TensorFlow deep classifier. Applied SMOTE for class imbalance and feature importance ranking to select the top 18 predictive variables.

| Metric | Score |
|---|---|
| Accuracy | 94.3% |
| F1 Score | 0.912 |
| AUC-ROC | 0.97 |
| Precision | 93.7% |
| Recall (TPR) | 96.0% |

**Key Features Used:**
- `glucose_normalized` (log-transformed)
- `age_bp_interaction` (engineered interaction term)
- `bmi_risk` (binary threshold feature)
- `cholesterol`, `systolic_bp`, `age`

---

### Model 2 — House Price Estimator

Estimates residential property prices from structural and location features. Stacked ensemble combining Gradient Boosting and XGBoost with a Ridge regression meta-learner. Extensive feature engineering including interaction terms, log transforms, and neighborhood clustering.

| Metric | Score |
|---|---|
| R² Score | 0.924 |
| MAPE | 11.2% |
| Mean Absolute Error | ₹18,000 |

**Baseline vs Final:**
| Model | R² |
|---|---|
| Linear Regression (baseline) | 0.74 |
| Gradient Boosting (tuned) | 0.91 |
| Stacked Ensemble (final) | 0.924 |

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.11 |
| ML Framework | Scikit-learn |
| Deep Learning | TensorFlow 2.x |
| Gradient Boosting | XGBoost |
| Data Wrangling | Pandas, NumPy |
| Imbalanced Data | imbalanced-learn (SMOTE) |
| Hyperparameter Tuning | GridSearchCV, RandomizedSearchCV |
| Visualization | Matplotlib, Seaborn |
| BI Dashboards | Power BI |
| Model Persistence | joblib |
| Inference API | FastAPI |
| Prototyping | Jupyter Notebook |

---

## Project Structure

```
neuralEdge/
├── data/
│   ├── raw/                        # Original datasets
│   ├── processed/                  # Cleaned and engineered datasets
│   └── external/                   # Neighborhood clustering data
│
├── notebooks/
│   ├── 01_eda_health.ipynb         # Exploratory data analysis — health
│   ├── 02_eda_house.ipynb          # Exploratory data analysis — house
│   ├── 03_health_model.ipynb       # Health model training and eval
│   └── 04_house_model.ipynb        # House model training and eval
│
├── src/
│   ├── health_model.py             # Health outcome classifier pipeline
│   ├── house_model.py              # House price regression pipeline
│   ├── hyperparameter_tuning.py    # GridSearchCV + RandomizedSearchCV
│   ├── inference_pipeline.py       # FastAPI inference endpoints
│   └── utils/
│       ├── feature_engineering.py  # Feature creation functions
│       ├── preprocessing.py        # Scaling, encoding, imputation
│       └── evaluation.py           # Metrics, confusion matrix, plots
│
├── models/
│   ├── health_tf_pipeline.pkl      # Saved health model (joblib)
│   └── house_stack_pipeline.pkl    # Saved house model (joblib)
│
├── dashboards/
│   └── stakeholder_report.pbix     # Power BI dashboard file
│
├── requirements.txt
└── README.md
```

---

## ML Pipeline

```
Raw Data
    │
    ▼
Data Ingestion ──── Pandas · CSV / SQL · Missing value imputation · Outlier detection (IQR)
    │
    ▼
Feature Engineering ──── Interaction terms · Log transforms · Encoding · Scaling
    │
    ▼
Class Balancing ──── SMOTE oversampling · Stratified splits · Class weighting
    │
    ▼
Hyperparameter Tuning ──── GridSearchCV (RF) · RandomizedSearchCV (XGBoost) · 5-fold CV
    │
    ▼
Evaluation ──── Accuracy / F1 / AUC-ROC / R² / MAPE · Confusion matrix · Feature importance
    │
    ▼
Deployment ──── FastAPI endpoint · joblib persistence · Live data workflow integration
```

---

## Hyperparameter Tuning

### GridSearchCV — Random Forest (Health Model)
```python
rf_param_grid = {
    'n_estimators':      [100, 200, 300],
    'max_depth':         [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'max_features':      ['sqrt', 'log2'],
}
# 5-fold CV, scoring='f1', n_jobs=-1
```

### RandomizedSearchCV — XGBoost (House Model)
```python
xgb_param_dist = {
    'n_estimators':     randint(100, 800),
    'learning_rate':    uniform(0.01, 0.2),
    'max_depth':        randint(3, 10),
    'subsample':        uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.5, 0.5),
    'reg_alpha':        uniform(0, 1),
}
# n_iter=60, 5-fold CV, n_jobs=-1
```

---

## Inference API

Both models are served via a FastAPI application with POST endpoints that accept structured input and return predictions in real time.

**Health Endpoint:** `POST /predict/health`
```json
Request:  { "age": 52, "bmi": 31.4, "glucose": 148, "systolic_bp": 142, "cholesterol": 220 }
Response: { "risk_probability": 0.83, "outcome": "high" }
```

**House Price Endpoint:** `POST /predict/house`
```json
Request:  { "total_sqft": 1850, "overall_qual": 7, "neighborhood": "Saidapet", "year_built": 2010, "garage_area": 400 }
Response: { "estimated_price": 4820000.0, "currency": "INR" }
```

---

## Feature Engineering Highlights

### Health Model
- **`glucose_normalized`** — Log1p transform to reduce right skew in glucose distribution
- **`bmi_risk`** — Binary flag for BMI > 30 (clinical obesity threshold)
- **`age_bp_interaction`** — Multiplicative interaction between age and systolic blood pressure
- Top 18 features selected from Random Forest feature importance ranking

### House Model
- **Log transforms** on `lot_area`, `total_sqft`, `garage_area` (skewed distributions)
- **`quality_x_area`** — Overall quality score multiplied by total square footage
- **`age_remod`** — Years since last renovation (yr_sold − year_remod_add)
- **Neighborhood clustering** — KMeans (k=8) to group neighborhoods by price band

---

## Visualizations & Dashboards

- **Matplotlib** — Feature importance bar charts, ROC curves, confusion matrices, predicted vs actual scatter plots, learning curves, cross-validation score distributions
- **Power BI** — Stakeholder-facing dashboards with interactive filters for model performance metrics, prediction distributions, and feature contribution breakdowns

---

## Internship Context

**Company:** Elewayte  
**Role:** AI & Data Science Intern  
**Duration:** September 2025 – November 2025  
**Focus:** ML model development, hyperparameter optimization, data visualization, pipeline integration

---

## Requirements

```
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
tensorflow>=2.13.0
xgboost>=1.7.0
imbalanced-learn>=0.11.0
matplotlib>=3.7.0
seaborn>=0.12.0
fastapi>=0.100.0
uvicorn>=0.23.0
joblib>=1.3.0
scipy>=1.11.0
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## Showcase

Open `ai-datascience-internship.html` in any browser to view the interactive project showcase — no setup required.

---

## License

This project was developed during an internship at Elewayte. All code is for portfolio demonstration purposes.
