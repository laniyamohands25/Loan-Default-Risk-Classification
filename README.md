# 🏦 Loan Default Risk Classification

> **Project #11 — Predictive Analytics (Group Project3)**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://loan-default-risk-classification-project3.streamlit.app/)

---
---

# 📋 Project Details

| Field | Details |
|---|---|
| **Course** | Predictive Analytics |
| **Institution** | Kerala Digital University |
| **Project Type** | Machine Learning & Predictive Analytics |
| **Deployment** | Streamlit Web Application |
| **Models Used** | Logistic Regression, Random Forest, LightGBM |

---
## 👥 Team Members & Course Details


| Name |  |
|---|---|
| Project3 | Group 8 |
| Sreekutty Santhosh | Bio-AI (253213)|
| Laniya Mohan |Bio-AI (253218) |
| Theertha Vijyachandran | Bio-AI (253223|


---
| Field | Details |
|---|---|
| **Project Title** | Loan Default Risk Classification for Banking |
| **Dataset** | Home Credit Default Risk (`application_train.csv`) |
| **Course** | Data Science Project (Machine Learning / AI) |


| **Live Deployment** | *(https://loan-default-risk-classification-project3.streamlit.app/)* |

---

## 📌 Problem Statement & Motivation

Many individuals lack sufficient credit history, making it difficult for banks to assess loan eligibility fairly. Traditional credit scoring systems often exclude legitimate borrowers, while approving high-risk applicants can lead to significant financial losses.

This project addresses the challenge of binary loan default classification — predicting whether an applicant is likely to default on a loan. The goal is to build a model that:

- Minimizes risky approvals (optimizes for recall on the default class)
- Ensures fairness using income, employment, and credit history features
- Provides regulatory-compliant explainability using SHAP values
- Deploys as an interactive web application for real-world use

---

## 📊 Dataset Description

| Property | Details |
|---|---|
| **Source** | Home Credit Default Risk — Kaggle |
| **File Used** | `application_train.csv` |
| **Total Records** | 307,511 rows (approx.) |
| **Original Features** | 122 columns |
| **Target Variable** | `TARGET` — `0` = No Default, `1` = Default |
| **Class Distribution** | Highly imbalanced (~92% No Default / ~8% Default) |

### Key Features Used

| Feature | Description |
|---|---|
| `AMT_INCOME_TOTAL` | Applicant annual income |
| `AMT_CREDIT` | Total loan amount requested |
| `AMT_ANNUITY` | Monthly loan repayment |
| `AMT_GOODS_PRICE` | Price of financed goods |
| `EXT_SOURCE_2` | External credit score |
| `DAYS_BIRTH` | Applicant age (days) |
| `DAYS_EMPLOYED` | Employment duration |
| `LOAN_TO_INCOME_RATIO` | Engineered financial ratio |

---

## 🔬 Methodology Overview

The project follows the complete Data Science Project Life Cycle.

### Stage 1 — Problem Definition & Literature Review

Defined the binary classification problem to predict loan default risk. Studied literature related to credit risk modelling, class imbalance handling, explainable AI, and regulatory compliance in banking systems.

### Stage 2 — Data Collection & Understanding

- Loaded dataset using Pandas
- Explored shape, datatypes, missing values, and target distribution
- Identified severe class imbalance in the dataset

### Stage 3 — Data Preprocessing & Cleaning

- Removed columns with more than 50% missing values
- Applied median imputation for numerical features
- Applied mode imputation for categorical features
- Performed log transformation using `np.log1p()` for income distribution normalization

### Stage 4 — Exploratory Data Analysis (EDA)

Performed visual analysis using:

- Count plots for loan default distribution
- Boxplots for income comparison
- Correlation heatmaps
- Credit score vs default analysis
- Feature relationship visualization

### Stage 5 — Feature Engineering & Selection

Created engineered features such as:

```python
LOAN_TO_INCOME_RATIO = AMT_CREDIT / AMT_INCOME_TOTAL
```

Selected important features based on:

- Correlation analysis
- Domain relevance
- Model importance

Dataset was split using stratified train-test splitting (80/20).

### Stage 6 — Model Building & Training

Three machine learning models were trained and compared.

| Model | Configuration |
|---|---|
| Logistic Regression | `class_weight='balanced'` |
| Random Forest | 300 trees with max depth |
| LightGBM | Gradient boosting with imbalance handling |

#### Handling Class Imbalance

- SMOTE applied on training data
- Balanced class weights used
- `scale_pos_weight` used in LightGBM

#### Data Scaling

`StandardScaler` applied after SMOTE processing.

### Stage 7 — Model Evaluation & Comparison

Models evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

Primary optimization target: **Recall**, to reduce false negatives and risky loan approvals.

#### Evaluation Visualizations
# -correlation heatmap
-<img width="898" height="657" alt="heatmap" src="https://github.com/user-attachments/assets/172ad8e6-9e6a-494d-9e1f-eb2014e15f25" />

# -loan to income ratio vs default plot
- <img width="686" height="470" alt="loan to income ratio vs default" src="https://github.com/user-attachments/assets/49d78d29-5b71-415a-93cc-bfbe991f8a7d" />

# - Confusion matrices
- <img width="1388" height="431" alt="Screenshot 2026-05-16 171832" src="https://github.com/user-attachments/assets/84234d0d-f9c5-4bc6-8dfd-c9a10137c276" />

# - ROC curves
- <img width="889" height="590" alt="ROC_curve" src="https://github.com/user-attachments/assets/6c19f3e9-7214-4198-bc18-5f2522ea7d1c" />

# - Precision-Recall curves
- <img width="889" height="590" alt="precison Recall curves" src="https://github.com/user-attachments/assets/5c6c2f8f-2e69-4769-a0e7-63d6baabadf9" />

# -5 fold cross validation
- <img width="690" height="390" alt="5 fold cross validation" src="https://github.com/user-attachments/assets/8e0fa7a1-5605-4906-8141-01d272f09c4b" />



### Stage 8 — Explainability Using SHAP

Implemented SHAP explainability for model transparency.

- Used `shap.TreeExplainer`
- Generated SHAP summary plots
- Identified important predictors affecting loan default

Top impactful features:

- `EXT_SOURCE_2`
- `LOAN_TO_INCOME_RATIO`
- `AMT_CREDIT`
- `AMT_INCOME_TOTAL`

### Stage 9 — Streamlit Deployment

Built a complete interactive Streamlit application with:

- Applicant input forms
- Real-time prediction
- Risk classification
- Probability gauge charts
- Model comparison tables
- Recommendation system
- Live metrics display

### Stage 10 — Documentation

Project includes:

- Jupyter notebook documentation
- Streamlit deployment app
- Requirements file
- Complete GitHub README

---

## 📈 Results Summary

### Model Comparison

| Model | Strength | Weakness |
|---|---|---|
| Logistic Regression | Fast and interpretable | Lower recall |
| Random Forest | Good precision | Slower inference |
| LightGBM | Best recall and ROC-AUC | Slightly less interpretable |

### Key Findings

- `EXT_SOURCE_2` is the strongest predictor of default
- Engineered features improved prediction performance
- SMOTE significantly improved minority class recall
- Threshold tuning improved high-risk detection

---

---

## ⚙️ Local Setup & Running Instructions

### Prerequisites

- Python 3.8+
- pip

### 1. Clone Repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### requirements.txt

```txt
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
lightgbm>=4.0.0
plotly>=5.17.0
matplotlib>=3.7.0
joblib>=1.3.0
imbalanced-learn
shap
```

### 3. Download Dataset

Download the Home Credit Default Risk dataset from Kaggle and place:

```text
application_train.csv
```

inside the project directory.

### 4. Run Jupyter Notebook

```bash
jupyter notebook project3.ipynb
```

### 5. Launch Streamlit App

```bash
streamlit run app.py
```

Open browser:

```text
http://localhost:8501
```

---

## 📂 Project Structure

```text
.
├── data/
│   ├── train_data.csv
│   └── test_data.csv
│
├── images/
│   ├── eda/
│   ├── evaluation/
│   ├── shap/
│   └── app/
│
├── individual_profiles/
│
├── models/
│   ├── lgb_model.pkl
│   ├── lr_model.pkl
│   ├── rf_model_compressed.pkl.gz
│   ├── scaler_compressed.pkl.gz
│   └── meta.json
│
├── notebooks/
│   └── project3 (Workflow).ipynb
│
├── .gitignore
├── LICENSE
├── Loan Default Risk Classification.pptx
├── README.md
├── app.py
├── model_evaluation.ipynb
└── requirements.txt
```

---

## 🔗 Live Deployment

🌐 Streamlit App: *https://loan-default-risk-classification-project3.streamlit.app/*

---
# 🖼️ Application Screenshots

---

## 🏠 Home Screen

<img width="1920" height="1200" alt="home_app" src="https://github.com/user-attachments/assets/13777081-a35a-4379-a7b4-c9650c0ea8a8" />


---

## 🟢 Low Risk Prediction

<img width="1920" height="2392" alt="low_risk" src="https://github.com/user-attachments/assets/fb8350a5-808e-447d-945c-d7fb88772b1c" />


---

## 🟡 Medium Risk Prediction

<img width="1920" height="2392" alt="medium_risk" src="https://github.com/user-attachments/assets/1393c20a-8b6f-4d71-a145-6d706a55d226" />


---

## 🔴 High Risk Prediction

<img width="1920" height="2392" alt="high_risk" src="https://github.com/user-attachments/assets/7c5e9cf9-af5c-4e8d-855f-fa09b0a391c3" />


---
## 🛡️ Regulatory Compliance Note

This project uses SHAP explainability to support transparent AI-assisted credit risk decisions. Each prediction can be interpreted using feature contribution analysis, supporting fairness and compliance requirements in banking and fintech systems.

---

## 📚 References

- Home Credit Default Risk — Kaggle
- LightGBM Documentation
- SHAP Documentation
- Imbalanced-learn Documentation
- Streamlit Documentation
