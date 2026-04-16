from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model


ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "churn_ann_model.keras"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"
FEATURE_COLUMNS_PATH = ARTIFACTS_DIR / "feature_columns.json"


# Page config
st.set_page_config(
    page_title="AI Customer Risk Analyzer",
    page_icon="🚀",
    layout="wide"
)


# 🎨 Enhanced UI Styling
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }
    .hero {
        background: linear-gradient(135deg, #020617 0%, #0f172a 40%, #1e293b 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 24px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 20px 50px rgba(0,0,0,0.3);
    }
    .hero h1 {
        margin: 0;
        font-size: 2.5rem;
        letter-spacing: 1px;
    }
    .hero p {
        margin-top: 0.7rem;
        opacity: 0.85;
        font-size: 1.1rem;
    }
    .result-card {
        background: #ffffff;
        border-radius: 18px;
        padding: 1.5rem;
        margin-top: 1.5rem;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    .stButton>button {
        height: 3.2em;
        border-radius: 12px;
        font-size: 16px;
        font-weight: 600;
        background: linear-gradient(90deg, #22c55e, #16a34a);
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# 🎯 Hero Section
st.markdown(
    """
    <div class="hero">
        <h1>🚀 AI Customer Risk Analyzer</h1>
        <p>Smart prediction system to identify high-risk customers using Neural Networks</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# Load artifacts
@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing model file: {MODEL_PATH}. Run model.ipynb first."
        )

    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Missing scaler file: {SCALER_PATH}. Run model.ipynb first."
        )

    if not FEATURE_COLUMNS_PATH.exists():
        raise FileNotFoundError(
            f"Missing feature columns file: {FEATURE_COLUMNS_PATH}. Run model.ipynb first."
        )

    model = load_model(MODEL_PATH)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
        feature_columns = json.load(f)

    return model, scaler, feature_columns


try:
    model, scaler, feature_columns = load_artifacts()
except Exception as error:
    st.error(str(error))
    st.stop()


# 📊 Input Section
st.subheader("📥 Enter Customer Details")

with st.form("churn_form"):
    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.number_input("💳 Credit Score", 300, 850, 600)
        gender = st.selectbox("👤 Gender", ["Female", "Male"])
        age = st.number_input("🎂 Age", 18, 100, 40)
        tenure = st.number_input("📅 Tenure", 0, 10, 3)
        balance = st.number_input("💰 Balance", 0.0, 250000.0, 60000.0)

    with col2:
        num_products = st.number_input("📦 Products", 1, 4, 2)
        has_cr_card = st.selectbox("💳 Credit Card", ["Yes", "No"])
        is_active_member = st.selectbox("⚡ Active Member", ["Yes", "No"])
        estimated_salary = st.number_input("💵 Salary", 0.0, 500000.0, 80000.0)
        geography = st.selectbox("🌍 Location", ["France", "Germany", "Spain"])

    submitted = st.form_submit_button("🚀 Analyze Risk")


# 🔮 Prediction Section
if submitted:
    raw_row = pd.DataFrame(
        [
            {
                "CreditScore": credit_score,
                "Geography": geography,
                "Gender": gender,
                "Age": age,
                "Tenure": tenure,
                "Balance": balance,
                "NumOfProducts": num_products,
                "HasCrCard": 1 if has_cr_card == "Yes" else 0,
                "IsActiveMember": 1 if is_active_member == "Yes" else 0,
                "EstimatedSalary": estimated_salary,
            }
        ]
    )

    processed_row = raw_row.copy()
    processed_row["Gender"] = processed_row["Gender"].map({"Female": 0, "Male": 1})
    processed_row = pd.get_dummies(processed_row, columns=["Geography"], drop_first=True)
    processed_row = processed_row.reindex(columns=feature_columns, fill_value=0)

    scaled_row = scaler.transform(processed_row)
    churn_probability = float(model.predict(scaled_row, verbose=0)[0][0])
    stay_probability = 1.0 - churn_probability
    churn_label = "⚠️ HIGH RISK" if churn_probability >= 0.5 else "✅ LOW RISK"

    st.markdown('<div class="result-card">', unsafe_allow_html=True)

    st.subheader("📊 Risk Analysis Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("⚠️ Churn Risk", f"{churn_probability * 100:.2f}%")

    with col2:
        st.metric("✅ Retention Chance", f"{stay_probability * 100:.2f}%")

    st.write(f"### Final Decision: {churn_label}")

    st.progress(churn_probability)

    st.markdown("</div>", unsafe_allow_html=True)

    st.caption("Model uses ANN with preprocessing: encoding + scaling")