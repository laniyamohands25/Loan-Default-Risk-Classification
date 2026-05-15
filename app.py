
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gzip
import json
import plotly.graph_objects as go
import warnings
import os

warnings.filterwarnings('ignore')

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Loan Default Risk Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 1rem;
}

.risk-high {
    background-color: #ff6b6b;
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
}

.risk-low {
    background-color: #51cf66;
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
}

.risk-medium {
    background-color: #ffd43b;
    padding: 1rem;
    border-radius: 10px;
    color: #333;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD MODELS
# ==========================================

def load_model(filepath):

    if filepath.endswith(".gz"):

        with gzip.open(filepath, "rb") as f:
            return pickle.load(f)

    else:

        with open(filepath, "rb") as f:
            return pickle.load(f)

@st.cache_resource
def load_all_models():

    models_dir = "models"

    # LightGBM
    lgb_model = load_model(f"{models_dir}/lgb_model.pkl")

    # Logistic Regression
    lr_model = load_model(f"{models_dir}/lr_model.pkl")

    # Random Forest (compressed)
    rf_model = load_model(f"{models_dir}/rf_model_compressed.pkl.gz")

    # Metadata
    with open(f"{models_dir}/meta.json", "r") as f:
        meta = json.load(f)

    return lgb_model, rf_model, lr_model, meta

# Load models
lgb_model, rf_model, lr_model, meta = load_all_models()

FEATURE_NAMES = meta['feature_names']
OPTIMAL_THRESHOLD = meta['optimal_threshold']
LGB_METRICS = meta['lgb_metrics']

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def calculate_loan_to_income_ratio(credit, income):

    if income > 0:
        return credit / income

    return 0

def get_risk_level(probability):

    if probability < 0.3:
        return "Low Risk", "risk-low", "✅"

    elif probability < 0.6:
        return "Medium Risk", "risk-medium", "⚠️"

    else:
        return "High Risk", "risk-high", "🔴"

def create_gauge_chart(probability):

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={'text': "Default Probability (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 60], 'color': "yellow"},
                {'range': [60, 100], 'color': "salmon"}
            ]
        }
    ))

    fig.update_layout(height=300)

    return fig

# ==========================================
# APP HEADER
# ==========================================

st.markdown(
    '<div class="main-header">🏦 Loan Default Risk Predictor</div>',
    unsafe_allow_html=True
)

st.markdown("---")

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.header("📋 Model Information")

st.sidebar.info(f"""
Accuracy : {LGB_METRICS['accuracy']:.3f}

Recall : {LGB_METRICS['recall']:.3f}

Precision : {LGB_METRICS['precision']:.3f}

F1 Score : {LGB_METRICS['f1_score']:.3f}
""")

# ==========================================
# INPUT SECTION
# ==========================================

st.subheader("📝 Applicant Information")

col1, col2 = st.columns(2)

with col1:

    amt_income_total = st.number_input(
        "Annual Income",
        value=50000.0
    )

    amt_credit = st.number_input(
        "Loan Amount",
        value=20000.0
    )

    amt_annuity = st.number_input(
        "Loan Annuity",
        value=5000.0
    )

    amt_goods_price = st.number_input(
        "Goods Price",
        value=25000.0
    )

with col2:

    ext_source_2 = st.slider(
        "Credit Score",
        0.0,
        1.0,
        0.5
    )

    age = st.number_input(
        "Age",
        value=35
    )

    employment_years = st.number_input(
        "Employment Duration (Years)",
        value=5
    )

# ==========================================
# FEATURE ENGINEERING
# ==========================================

loan_to_income_ratio = calculate_loan_to_income_ratio(
    amt_credit,
    amt_income_total
)

# ==========================================
# PREDICTION
# ==========================================

if st.button("🔮 Predict Default Risk"):

    st.write("Prediction processing...")

    try:

        # Convert to training format
        days_birth = -age * 365
        days_employed = -employment_years * 365

        # Features
        features = np.array([[
            amt_income_total,
            amt_credit,
            amt_annuity,
            amt_goods_price,
            ext_source_2,
            days_birth,
            days_employed,
            loan_to_income_ratio
        ]])

        # ==========================================
        # MODEL PREDICTIONS
        # ==========================================

        probability = lgb_model.predict_proba(features)[0][1]

        prediction = (
            1 if probability >= OPTIMAL_THRESHOLD else 0
        )

        rf_proba = rf_model.predict_proba(features)[0][1]

        lr_proba = lr_model.predict_proba(features)[0][1]

        st.markdown("---")

        # ==========================================
        # RISK DISPLAY
        # ==========================================

        risk_text, risk_class, risk_icon = get_risk_level(probability)

        st.markdown(f"""
        <div class="{risk_class}">
            <h2>{risk_icon} {risk_text}</h2>
            <h3>Default Probability: {probability:.1%}</h3>
        </div>
        """, unsafe_allow_html=True)

        # ==========================================
        # GAUGE CHART
        # ==========================================

        fig = create_gauge_chart(probability)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ==========================================
        # MODEL COMPARISON
        # ==========================================

        st.subheader("📊 Model Comparison")

        comparison_df = pd.DataFrame({

            "Model": [
                "LightGBM",
                "Random Forest",
                "Logistic Regression"
            ],

            "Probability": [
                f"{probability:.1%}",
                f"{rf_proba:.1%}",
                f"{lr_proba:.1%}"
            ]

        })

        st.dataframe(
            comparison_df,
            use_container_width=True
        )

        # ==========================================
        # FINAL RECOMMENDATION
        # ==========================================

        st.subheader("📝 Final Recommendation")

        if prediction == 1:

            st.warning(
                "⚠️ High default risk detected. Manual review recommended."
            )

        else:

            st.success(
                "✅ Applicant appears creditworthy."
            )

    except Exception as e:

        st.error(f"Prediction Error: {str(e)}")

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.markdown(
    "<p style='text-align:center;color:gray;'>Powered by LightGBM | Banking Risk Analytics</p>",
    unsafe_allow_html=True
)
