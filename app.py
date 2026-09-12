# ============================================================
# PPD Risk Calculator - Streamlit Cloud Deployment Version
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="PPD Risk Calculator",
    page_icon="🍼",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------- Load assets (relative path for cloud) ----------
@st.cache_resource
def load_assets():
    import pickle
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("cat_en_to_cat.pkl", "rb") as f:
        cat_en_to_cat = pickle.load(f)
    with open("scaler_params.pkl", "rb") as f:
        scaler_params = pickle.load(f)
    return model, cat_en_to_cat, scaler_params

try:
    model, cat_en_to_cat, scaler_params = load_assets()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Model loading error: {e}")

# ---------- Header ----------
st.title("🍼 Postpartum Depression Risk Calculator")
st.markdown("""
**A machine-learning tool for postpartum depression (PPD) risk screening**

Based on a gradient boosting decision tree (GBDT) model  
Trained on 803 postpartum women from a tertiary hospital  
Model performance: **AUC = 0.771**

This calculator is intended for research and screening support only.  
It does **not** replace clinical diagnosis or professional judgment.
""")
st.markdown("---")

# ---------- Input ----------
st.header("Patient Information")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Demographics & Biochemistry")
    age = st.number_input(
        "Age (years)", min_value=18, max_value=50, value=30, step=1
    )
    ALT = st.number_input(
        "ALT (U/L) — Alanine aminotransferase",
        min_value=1.0, max_value=200.0, value=20.0, step=1.0
    )
    FT4 = st.number_input(
        "FT4 (pmol/L) — Free thyroxine",
        min_value=1.0, max_value=30.0, value=12.0, step=0.1
    )
    TSH = st.number_input(
        "TSH (mIU/L) — Thyroid stimulating hormone",
        min_value=0.01, max_value=10.0, value=2.0, step=0.1
    )
    MIL_care = st.slider(
        "Mother-in-law care score (0–10)",
        min_value=0, max_value=10, value=8,
        help="Higher score indicates more care and support"
    )

with col2:
    st.subheader("Psychosocial Factors")

    sleep_quality_label = st.selectbox(
        "Postpartum sleep quality",
        ["Good sleep", "Easy to wake up", "Difficulty falling asleep",
         "Racing thoughts", "Sleepless all night", "Other"]
    )

    sleep_hours_label = st.selectbox(
        "Postpartum daily sleep hours",
        ["< 5 hours", "5–6 hours", "6–8 hours", "8–12 hours", "> 12 hours"]
    )

    work_worry_label = st.selectbox(
        "Work-related worry", ["No", "Yes", "Sometimes"]
    )

    econ_worry_label = st.selectbox(
        "Economic worry", ["No", "Yes", "Sometimes"]
    )

    delivery_label = st.selectbox(
        "Delivery mode",
        ["Vaginal delivery", "Cesarean section", "Vaginal to cesarean"]
    )

# ---------- Button ----------
st.markdown("---")
col_b1, col_b2, col_b3 = st.columns([1, 1, 1])
with col_b2:
    calculate = st.button(
        "🔍 Calculate Risk",
        use_container_width=True,
        type="primary"
    )

# ---------- Predict ----------
if calculate and model_loaded:

    # Sleep scoring
    sleep_q_score_map = {
        "Good sleep": 3,
        "Easy to wake up": 2,
        "Difficulty falling asleep": 1,
        "Racing thoughts": 1,
        "Sleepless all night": 0,
        "Other": 1,
    }
    sleep_h_score_map = {
        "< 5 hours": 0,
        "5–6 hours": 1,
        "6–8 hours": 2,
        "8–12 hours": 3,
        "> 12 hours": 2,
    }

    sleep_composite = (
        sleep_q_score_map.get(sleep_quality_label, 1) *
        sleep_h_score_map.get(sleep_hours_label, 1)
    )

    stress_total = (
        (1 if work_worry_label == "Yes" else
         0.5 if work_worry_label == "Sometimes" else 0) +
        (1 if econ_worry_label == "Yes" else
         0.5 if econ_worry_label == "Sometimes" else 0)
    )

    # Standardization
    ALT_z = (ALT - scaler_params['ALT_mean']) / scaler_params['ALT_std']
    FT4_z = (FT4 - scaler_params['FT4_mean']) / scaler_params['FT4_std']
    ALT_FT4 = ALT_z * FT4_z

    # Encoding
    def encode_cat(feature_name, english_label):
        m = cat_en_to_cat.get(feature_name, {})
        return m.get(english_label, 'cat_0')

    input_data = pd.DataFrame([{
        'Age': age,
        'MIL_care': MIL_care,
        'ALT': ALT,
        'FT4': FT4,
        'TSH': TSH,
        'ALT_FT4': ALT_FT4,
        'Sleep_composite': sleep_composite,
        'Stress_total': stress_total,
        'Delivery_mode': encode_cat('Delivery_mode', delivery_label),
        'Work_worry': encode_cat('Work_worry', work_worry_label),
        'Econ_worry': encode_cat('Econ_worry', econ_worry_label),
    }])

    # Predict
    prob = model.predict_proba(input_data)[0, 1]

    # Risk stratification
    if prob < 0.079:
        risk_level, risk_color, risk_emoji = "LOW", "#2ca02c", "🟢"
        recommendation = (
            "Routine postpartum care. "
            "Standard EPDS screening every 2 weeks. "
            "Breastfeeding and maternal health education."
        )
    elif prob < 0.182:
        risk_level, risk_color, risk_emoji = "MEDIUM", "#ff7f0e", "🟡"
        recommendation = (
            "Enhanced monitoring. "
            "Weekly EPDS screening. "
            "Supportive counseling by trained nurses. "
            "Consider referral if symptoms persist."
        )
    else:
        risk_level, risk_color, risk_emoji = "HIGH", "#d62728", "🔴"
        recommendation = (
            "Immediate referral to mental health specialist. "
            "Psychiatric evaluation and individualized treatment. "
            "Multi-disciplinary follow-up until 6 months postpartum."
        )

    # ---------- Results ----------
    st.markdown("---")
    st.header("Prediction Result")

    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        st.metric("PPD Risk Probability", f"{prob:.1%}")
    with col_r2:
        st.metric("Risk Level", f"{risk_emoji} {risk_level}")
    with col_r3:
        st.metric("Model AUC", "0.771")

    # Risk bar
    st.markdown(f"""
    <div style="margin-top: 20px;">
        <div style="background-color: #f0f0f0; height: 30px;
                    border-radius: 15px; overflow: hidden;">
            <div style="background-color: {risk_color};
                        width: {prob*100:.1f}%; height: 100%;
                        transition: width 0.5s;"></div>
        </div>
        <p style="text-align: center; margin-top: 10px;">
            <b style="color: {risk_color}; font-size: 20px;">
                {risk_emoji} {risk_level} RISK
            </b>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Recommendation
    st.markdown("### 📋 Clinical Recommendation")
    if risk_level == "LOW":
        st.success(recommendation)
    elif risk_level == "MEDIUM":
        st.warning(recommendation)
    else:
        st.error(recommendation)

    # Reference
    with st.expander("ℹ️ Risk stratification reference"):
        st.markdown("""
        | Risk level | Probability range | Observed PPD rate |
        |------------|-------------------|-------------------|
        | Low        | < 7.9%            | 7.5%              |
        | Medium     | 7.9% – 18.2%      | 11.1%             |
        | High       | > 18.2%           | 37.5%             |

        **Note**: Thresholds derived from quantile-based stratification
        on the test set (n = 241). External validation is warranted.
        """)

    st.info(
        "This tool provides a research-level risk estimate. "
        "All clinical decisions should be made by qualified healthcare "
        "professionals based on comprehensive evaluation."
    )

# ---------- Footer ----------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 12px;">
    <p><b>Disclaimer</b>: This calculator is for research and screening
    support only. It does not replace clinical diagnosis or professional
    judgment.</p>
    <p>Based on: [Fan Xu] et al. (2026).
    Machine learning for postpartum depression risk prediction.
    [Journal Name].</p>
</div>
""", unsafe_allow_html=True)
