# Loan Default Risk Classification for Banking

> Predictive Analytics Course Project | 

---

# 📋 Project Details

| Field | Details |
|---|---|
| Course | Predictive Analytics |
| Institution | Kerala Digital University |
| Project Type | Banking Risk Analytics & Machine Learning |
| Dataset | Home Credit Default Risk Dataset |

---

# 👥 Team Members

| Name | Role |
|---|---|
| Member 1 | Data Preprocessing, EDA, Feature Engineering |
| Member 2 | Model Building & Evaluation |
| Member 3 | Deployment & Documentation |

---

# 📚 Stage 1: Problem Definition & Literature Review

Loan default prediction is an important problem in banking and financial risk management. Financial institutions use predictive analytics and machine learning techniques to identify high-risk borrowers before approving loans.

Traditional machine learning approaches such as Logistic Regression are widely used for credit risk analysis because of their interpretability and simplicity. Advanced ensemble learning methods such as Random Forest and LightGBM improve prediction performance by capturing complex nonlinear financial patterns.

Recent banking analytics research highlights the importance of handling class imbalance because loan default datasets usually contain significantly fewer default cases than non-default cases. Therefore, recall becomes an important evaluation metric for identifying risky borrowers correctly.

Explainable AI techniques such as SHAP are increasingly used in banking systems to improve transparency and interpretability.

This project applies these concepts to the Home Credit loan default dataset.

---

# 📌 Problem Statement

The objective of this project is to predict whether a loan applicant is likely to default using machine learning techniques on banking and credit-related data.

The project focuses on:
- identifying risky borrowers
- handling imbalanced financial datasets
- improving recall
- reducing risky loan approvals

---

# 📊 Stage 2: Data Collection & Understanding

## Dataset Used

Modified Home Credit Default Risk Dataset

### Primary Source

Mendeley Data Repository

https://data.mendeley.com/datasets/c8fnb7tgb3/1

### Original Source

Kaggle — Home Credit Default Risk Dataset

---

## Dataset Characteristics

- Binary classification dataset
- Real-world banking risk dataset
- Highly imbalanced target variable
- Contains missing values and outliers
- Financial and demographic features included

---

## Important Features

- Applicant Income (`AMT_INCOME_TOTAL`)
- Loan Amount (`AMT_CREDIT`)
- Loan Annuity (`AMT_ANNUITY`)
- Goods Price (`AMT_GOODS_PRICE`)
- External Credit Scores (`EXT_SOURCE_*`)
- Employment Information
- Age Information

---

## Target Variable

| Label | Meaning |
|---|---|
| 0 | Non-default |
| 1 | Loan Default |

---

## Class Distribution

| Class | Percentage |
|---|---|
| Non-default | ~91.9% |
| Default | ~8.1% |

The dataset is highly imbalanced, making recall an important metric for identifying risky loan applicants.

---

# ⚙️ Stage 3: Data Preprocessing & Cleaning

The following preprocessing techniques were applied:

- Missing value analysis
- Removal of highly sparse columns
- Median imputation for numerical columns
- Mode imputation for categorical columns
- Handling missing values
- Preparing data for machine learning workflows

---

# 📊 Stage 4: Exploratory Data Analysis (EDA)

EDA techniques completed in the project include:

- Loan default class distribution analysis
- Income distribution analysis
- Income vs default analysis
- Credit-related feature analysis
- Correlation heatmap analysis
- Outlier analysis

---

## Key EDA Observations

- The dataset is highly imbalanced with significantly fewer default cases.
- Income-related features contain extreme outliers.
- Credit-related external score features strongly influence default risk.
- Loan amount and annuity features show strong positive correlation.

---

# 🧩 Stage 5: Feature Engineering & Selection

## Engineered Features

| Feature | Description |
|---|---|
| Loan-to-Income Ratio | Measures loan burden relative to applicant income |

This engineered feature helps capture financial repayment burden and risk exposure.

---

## Selected Features for Modeling

The following important financial features were selected:

- AMT_INCOME_TOTAL
- AMT_CREDIT
- AMT_ANNUITY
- AMT_GOODS_PRICE
- EXT_SOURCE_2
- DAYS_BIRTH
- DAYS_EMPLOYED
- LOAN_TO_INCOME_RATIO

---

# 🔀 Train-Test Split Preparation

The dataset was split into training and testing sets using stratified sampling to preserve class imbalance distribution.

- Training Set: 80%
- Testing Set: 20%

Stratified sampling was used because the dataset is highly imbalanced.

---

# 🚧 Upcoming Stages

The following stages will be completed in upcoming project phases:

- Model Building & Training
- Model Evaluation & Comparison
- SHAP Explainability
- Streamlit Deployment
- Final Documentation & PPT

---

# 📁 Current Project Structure

```bash
Loan-Default-Risk-Classification/
│
├── notebooks/
│   └── 1_Data_Cleaning_and_EDA.ipynb
│
├── data/
│   ├── train_data.csv
│   ├── test_data.csv
│   └── dataset_info.txt
│
├── requirements.txt
├── README.md
│
├── individual_profiles/
│
└── screenshots/
```

---

# 🚀 How to Run the Notebook

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/Loan-Default-Risk-Classification.git

# Install dependencies
pip install -r requirements.txt

# Open notebook using Google Colab or Jupyter Notebook
```

---

# 📚 References

- Home Credit Default Risk Dataset
- LightGBM Documentation
- SHAP Documentation
- Scikit-learn Documentation
- Streamlit Documentation

---

# 📜 License

This project was developed as part of the Predictive Analytics course project.
