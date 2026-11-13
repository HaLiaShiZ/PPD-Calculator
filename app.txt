import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# 1. Page Configuration
st.set_page_config(page_title="PPD Risk Calculator", page_icon="🏥", layout="wide")

# Hide Streamlit default menu and footer
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# 2. Load Models
@st.cache_resource
def load_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pipe = joblib.load(os.path.join(base_dir, 'ppd_9var_pipeline.pkl'))
    iso = joblib.load(os.path.join(base_dir, 'ppd_9var_iso.pkl'))
    features = joblib.load(os.path.join(base_dir, 'ppd_9var_features.pkl'))
    return pipe, iso, features

try:
    pipe, iso, feature_names = load_models()
except Exception as e:
    st.error(f"Model files not found. Please ensure all three .pkl files are in the same directory. Error: {e}")

# 3. Sidebar: Input Area
st.sidebar.header("📋 Maternal Information")
st.sidebar.markdown("Please enter the following 8 indicators based on the 42-day postpartum follow-up (the 9th interaction variable will be calculated automatically):")

age = st.sidebar.number_input("Maternal Age (years)", min_value=18, max_value=50, value=30)
mother_in_law_care = st.sidebar.slider("Mother-in-law Care Score (0-10)", 0, 10, 8, help="Higher score indicates better care")
composite_sleep = st.sidebar.slider("Composite Sleep Score (0-6)", 0, 6, 3, help="Derived from sleep quality and duration")
total_stress = st.sidebar.slider("Total Stress Score (0-2)", 0, 2, 1, help="Includes work and financial concerns")
delivery_mode = st.sidebar.selectbox("Delivery Mode", ["Vaginal Delivery", "Cesarean Section"])
parity = st.sidebar.selectbox("Parity", ["Primipara", "Multipara"])
ALT = st.sidebar.number_input("ALT (U/L)", min_value=0.0, max_value=500.0, value=15.0, step=1.0)
FT4 = st.sidebar.number_input("FT4 (pmol/L)", min_value=1.0, max_value=50.0, value=12.0, step=0.1)

# 4. Construct Features (including the 9th variable ALT×FT4)
input_dict = {
    'age': [age],
    'mother_in_law_care': [mother_in_law_care],
    'composite_sleep': [composite_sleep],
    'total_stress': [total_stress],
    'ALT': [ALT],
    'FT4': [FT4],
    'ALT_FT4': [ALT * FT4],  # 9th variable: automatically calculated interaction
    'delivery_mode_vaginal': [1 if delivery_mode == "Vaginal Delivery" else 0],
    'parity_primipara': [1 if parity == "Primipara" else 0]
}
input_df = pd.DataFrame(input_dict)[feature_names]

# 5. Main Interface Layout
st.title("🏥 Postpartum Depression (PPD) Risk Calculator")
st.markdown("### Based on a Parsimonious 9-Variable Interpretable Machine Learning Model (GBDT + Isotonic)")
st.info("💡 **Note**: This tool is based on routine low-cost indicators and is intended for research and risk stratification purposes only. It is not a substitute for clinical diagnosis.")

# Split main interface into two columns
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("#### 📖 Model Background")
    st.markdown("""
    This study developed a parsimonious PPD risk prediction model based on **803 women** at 42 days postpartum. 
    The model innovatively integrates **psychosocial factors** (sleep, mother-in-law care, stress, etc.) with **liver-thyroid axis biomarkers** (ALT, FT4, and their interaction).
    
    **Model Performance**:
    *   Nested Cross-Validation AUC: **0.750**
    *   Calibrated AUC: **0.769**
    *   Events-per-Variable (EPV): **16.8** (well above the conventional threshold of 10, effectively preventing overfitting)
    """)
    
    st.markdown("#### 📊 Risk Stratification Cutoffs")
    st.markdown("""
    Based on K-means clustering, patients are stratified into three risk tiers:
    *   🟢 **Low Risk**: Probability < **0.117** (Observed PPD incidence ~8.8%)
    *   🟡 **Medium Risk**: 0.117 ≤ Probability < **0.484** (Observed PPD incidence ~22.5%)
    *   🔴 **High Risk**: Probability ≥ **0.484** (Observed PPD incidence ~53.7%)
    """)

with col_right:
    st.markdown("#### 📈 Prediction Results")
    
    # Placeholder for displaying results
    result_placeholder = st.empty()
    
    if st.button("🔄 Calculate PPD Risk Probability", type="primary", use_container_width=True):
        # Prediction
        raw_prob = pipe.predict_proba(input_df)[0, 1]
        calibrated_prob = iso.predict([raw_prob])[0]
        
        # Dynamically render result panel
        with result_placeholder.container():
            st.markdown("---")
            # Large number display
            st.metric(label="Calibrated PPD Risk Probability", value=f"{calibrated_prob:.1%}")
            
            # Progress bar display
            st.progress(float(calibrated_prob))
            
            # Risk tier cards
            if calibrated_prob < 0.117:
                st.success("🟢 **Low Risk**")
                st.markdown("**Recommendation**: Routine postpartum follow-up. Maintain good sleep and family relationships.")
            elif calibrated_prob < 0.484:
                st.warning("🟡 **Medium Risk**")
                st.markdown("**Recommendation**: Enhanced monitoring. Pay attention to mood changes; psychological counseling may be considered.")
            else:
                st.error("🔴 **High Risk**")
                st.markdown("**Recommendation**: Referral to a psychiatric/psychological specialist for further evaluation and intervention.")
                
            st.caption("Note: The model output probability is for risk ranking reference only. Please interpret in conjunction with clinical context.")
    else:
        # Default placeholder content before clicking button
        with result_placeholder.container():
            st.markdown("---")
            st.markdown("<h4 style='text-align: center; color: gray; margin-top: 50px;'>Please fill in the data on the left and click the button above to generate the risk report</h4>", unsafe_allow_html=True)