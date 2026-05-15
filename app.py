
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Loan Default Risk Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Load models and artifacts
@st.cache_resource
def load_models():
    """Load all model files and metadata"""
    with open("models/lgb_model.pkl", "rb") as f:
        lgb_model = pickle.load(f)
    with open("models/rf_model.pkl", "rb") as f:
        rf_model = pickle.load(f)
    with open("models/lr_model.pkl", "rb") as f:
        lr_model = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("models/meta.json", "r") as f:
        meta = json.load(f)
    return lgb_model, rf_model, lr_model, scaler, meta

lgb_model, rf_model, lr_model, scaler, meta = load_models()

# Extract metadata
FEATURE_NAMES = meta['feature_names']
OPTIMAL_THRESHOLD = meta['optimal_threshold']
LGB_METRICS = meta['lgb_metrics']
FEATURE_IMPORTANCE = meta['feature_importance']

def calculate_loan_to_income_ratio(amt_credit, amt_income_total):
    """Calculate loan-to-income ratio"""
    if amt_income_total > 0:
        return amt_credit / amt_income_total
    return 0.0

def get_risk_level(probability):
    """Determine risk level based on probability"""
    if probability < 0.3:
        return "Low Risk", "risk-low", "✅"
    elif probability < 0.6:
        return "Medium Risk", "risk-medium", "⚠️"
    else:
        return "High Risk", "risk-high", "🔴"

def create_gauge_chart(probability):
    """Create a gauge chart for probability visualization"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        title={'text': "Default Probability (%)"},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 60], 'color': "yellow"},
                {'range': [60, 100], 'color': "salmon"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def create_feature_importance_chart():
    """Create feature importance bar chart from saved importance"""
    df_importance = pd.DataFrame({
        'Feature': list(FEATURE_IMPORTANCE.keys()),
        'Importance': list(FEATURE_IMPORTANCE.values())
    }).sort_values('Importance', ascending=True)
    
    fig = px.bar(df_importance, 
                 x='Importance', 
                 y='Feature',
                 orientation='h',
                 color='Importance',
                 color_continuous_scale='Blues',
                 title="Feature Importance (LightGBM)")
    fig.update_layout(height=400)
    return fig

# Header
st.markdown('<div class="main-header">🏦 Loan Default Risk Predictor</div>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.header("📋 About")
st.sidebar.info(
    f"""
    **Model Performance (LightGBM)**
    - Accuracy : {LGB_METRICS['accuracy']*100:.1f}%
    - Recall   : {LGB_METRICS['recall']:.3f}
    - Precision: {LGB_METRICS['precision']:.3f}
    - F1-Score : {LGB_METRICS['f1_score']:.3f}
    
    **Optimal Threshold**: {OPTIMAL_THRESHOLD:.3f}
    
    **Features Used:**
    {', '.join(FEATURE_NAMES)}
    """
)

st.sidebar.markdown("---")
st.sidebar.markdown("**⚠️ Disclaimer**")
st.sidebar.caption(
    "This is a predictive model. Final loan decisions should consider additional factors and human review."
)

# Input Method
st.subheader("📝 Enter Loan Applicant Information")

col1, col2 = st.columns(2)

with col1:
    amt_income_total = st.number_input(
        "💰 Annual Income ($)",
        min_value=0.0,
        max_value=10_000_000.0,
        value=50000.0,
        step=1000.0,
        format="%.0f",
        help="Total annual income of the applicant"
    )
    
    amt_credit = st.number_input(
        "💳 Loan Amount Requested ($)",
        min_value=0.0,
        max_value=10_000_000.0,
        value=20000.0,
        step=1000.0,
        format="%.0f",
        help="Total loan amount requested"
    )
    
    amt_annuity = st.number_input(
        "📅 Annual Loan Payment ($)",
        min_value=0.0,
        max_value=1_000_000.0,
        value=5000.0,
        step=500.0,
        format="%.0f",
        help="Annual installment payment"
    )

with col2:
    amt_goods_price = st.number_input(
        "🛒 Goods Price ($)",
        min_value=0.0,
        max_value=10_000_000.0,
        value=25000.0,
        step=1000.0,
        format="%.0f",
        help="Price of the goods being purchased"
    )
    
    ext_source_2 = st.slider(
        "📊 Credit Score (EXT_SOURCE_2)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        help="External credit score (higher is better)"
    )
    
    days_birth = st.number_input(
        "🎂 Age (years)",
        min_value=18,
        max_value=100,
        value=35,
        step=1,
        help="Applicant's age"
    )
    
    days_employed = st.number_input(
        "💼 Employment Duration (years)",
        min_value=0,
        max_value=50,
        value=5,
        step=1,
        help="Years employed"
    )

# Calculate loan-to-income ratio
loan_to_income_ratio = calculate_loan_to_income_ratio(amt_credit, amt_income_total)

# Display calculated ratio
st.metric(
    "📈 Calculated Loan-to-Income Ratio",
    f"{loan_to_income_ratio:.2f}",
    help="Higher ratio indicates higher risk (above 0.5 is concerning)"
)

# Predict button
if st.button("🔮 Predict Default Risk", type="primary", use_container_width=True):
    # Convert days to appropriate format (negative as in original data)
    days_birth_days = -days_birth * 365
    days_employed_days = -days_employed * 365 if days_employed > 0 else 0
    
    # Prepare features
    features = np.array([[
        amt_income_total,
        amt_credit,
        amt_annuity,
        amt_goods_price,
        ext_source_2,
        days_birth_days,
        days_employed_days,
        loan_to_income_ratio
    ]])
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict with LightGBM (best model)
    probability = lgb_model.predict_proba(features_scaled)[0][1]
    prediction = 1 if probability >= OPTIMAL_THRESHOLD else 0
    
    # Get predictions from other models for comparison
    rf_proba = rf_model.predict_proba(features_scaled)[0][1]
    lr_proba = lr_model.predict_proba(features_scaled)[0][1]
    
    st.markdown("---")
    st.subheader("📊 Prediction Results")
    
    # Main results row
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        risk_text, risk_class, risk_icon = get_risk_level(probability)
        st.markdown(f"""
        <div class='{risk_class}'>
            <h2>{risk_icon} {risk_text}</h2>
            <p style='font-size: 1.2rem; margin: 0;'>
                Default Probability: {probability:.1%}
            </p>
            <p style='margin-top: 10px;'>
                {'⚠️ Recommend: Review application carefully' if prediction == 1 else '✅ Recommend: Approve loan'}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        fig = create_gauge_chart(probability)
        st.plotly_chart(fig, use_container_width=True)
    
    # Model comparison
    st.subheader("🔄 Model Comparison")
    comparison_df = pd.DataFrame({
        'Model': ['LightGBM (Primary)', 'Random Forest', 'Logistic Regression'],
        'Default Probability': [f"{probability:.1%}", f"{rf_proba:.1%}", f"{lr_proba:.1%}"],
        'Prediction': [
            'Default' if prediction == 1 else 'No Default',
            'Default' if rf_proba >= OPTIMAL_THRESHOLD else 'No Default',
            'Default' if lr_proba >= OPTIMAL_THRESHOLD else 'No Default'
        ]
    })
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # Feature importance visualization
    st.subheader("🔍 Feature Impact Analysis")
    fig_importance = create_feature_importance_chart()
    st.plotly_chart(fig_importance, use_container_width=True)
    
    # Risk factors explanation
    st.subheader("📝 Risk Assessment Explanation")
    
    risk_factors = []
    protective_factors = []
    
    if loan_to_income_ratio > 0.5:
        risk_factors.append(f"• **High loan-to-income ratio** ({loan_to_income_ratio:.2f}) - Loan amount is high relative to income")
    
    if ext_source_2 < 0.3:
        risk_factors.append(f"• **Low credit score** ({ext_source_2:.2f}) - Poor credit history indicated")
    
    if days_employed < 2:
        risk_factors.append(f"• **Short employment duration** ({days_employed} years) - Less job stability")
    
    if amt_credit > amt_income_total * 0.5:
        risk_factors.append(f"• **Large loan relative to income** - Loan exceeds 50% of annual income")
    
    if ext_source_2 > 0.7:
        protective_factors.append(f"• **Good credit score** ({ext_source_2:.2f}) - Strong credit history")
    
    if days_employed > 5:
        protective_factors.append(f"• **Stable employment** ({days_employed} years)")
    
    if loan_to_income_ratio < 0.3:
        protective_factors.append(f"• **Low loan-to-income ratio** ({loan_to_income_ratio:.2f})")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if risk_factors:
            st.markdown("**⚠️ Risk Factors Identified:**")
            for factor in risk_factors:
                st.write(factor)
        else:
            st.success("✅ No major risk factors identified")
    
    with col2:
        if protective_factors:
            st.markdown("**✅ Protective Factors:**")
            for factor in protective_factors:
                st.write(factor)
    
    # Final recommendation
    st.markdown("---")
    if probability >= OPTIMAL_THRESHOLD:
        st.warning("⚠️ **Final Recommendation: Review Required** - High default probability detected. Consider additional verification or lower loan amount.")
    else:
        st.success("✅ **Final Recommendation: Approve** - Applicant appears creditworthy based on the provided information.")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Powered by LightGBM | Optimized for Recall | Regulatory Compliant</p>",
    unsafe_allow_html=True
)
