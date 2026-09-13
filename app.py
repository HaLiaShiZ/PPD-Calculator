# app.py
import streamlit as st
import pandas as pd
import pickle, json

# 英文化界面配置
st.set_page_config(page_title="PPD Risk Calculator", page_icon="🧠", layout="wide")
st.title("Postpartum Depression (PPD) Risk Calculator")
st.caption("Based on LR-L1 + sigmoid model; for research purposes only, not for clinical diagnosis.")

# 输入标签英文映射（UI用英文，传给模型仍用中文）
LABEL_MAP = {
    "年龄": "Age", "独生子女": "Only child", "婚龄": "Years of marriage", "居住地": "Residence",
    "受教育程度": "Education level", "就业状况": "Employment status", "医疗保险": "Medical insurance",
    "家庭月收入": "Monthly household income", "童年创伤": "Childhood trauma", "抑郁症史": "History of depression",
    "家族抑郁": "Family history of depression", "居住情况": "Living situation", "配偶支持": "Spousal support",
    "公婆性别期望": "In-laws' gender expectation", "孕期讲座": "Pregnancy classes", "睡眠状况": "Sleep status",
    "工作担忧": "Work concerns", "经济担忧": "Financial concerns", "信任倾诉": "Trust and confiding",
    "婆婆关爱": "Mother-in-law's care", "胎次": "Parity", "计划怀孕": "Planned pregnancy",
    "分娩方式": "Delivery mode", "分娩孕周": "Gestational age at delivery", "产科并发症": "Obstetric complications",
    "喂养方式": "Feeding method", "月子照顾者": "Postpartum caregiver", "产褥期异常": "Puerperal abnormalities",
    "睡眠质量打分": "Sleep quality score", "睡眠时长打分": "Sleep duration score"
}

# 选项英文映射
OPTION_MAP = {
    "否": "No", "是": "Yes", "1-3年": "1-3 years", "3-5年": "3-5 years", "5-10年": "5-10 years",
    ">10年": ">10 years", "未婚": "Unmarried", "＜1年": "<1 year", "农村": "Rural", "县镇": "Town",
    "城市": "City", "初中及以下": "Junior high or below", "大专及本科": "College/Bachelor", 
    "研究生及以上": "Postgraduate or above", "高中或中专": "High school/Secondary", "其他": "Other",
    "在岗": "Employed", "失业": "Unemployed", "待岗": "Waiting for work", "无": "No", "有": "Yes",
    "有┋无": "Yes/No", "1万元以上": "Over 10,000 CNY", "3000-6000元": "3,000-6,000 CNY",
    "3000元以下": "Under 3,000 CNY", "6000-1万元": "6,000-10,000 CNY", "和女方父母同住": "With maternal parents",
    "和男方父母同住": "With paternal parents", "夫妻两地分居": "Separated from spouse", 
    "夫妻二人世界/夫妻孩子一家三口": "Nuclear family", "一般支持": "Moderate support", 
    "全力支持": "Full support", "极少支持": "Minimal support", "女孩": "Girl", "无所谓": "Indifferent", "男孩": "Boy",
    "入睡困难": "Difficulty falling asleep", "彻夜难眠": "Sleepless all night", "易醒": "Easily awakened",
    "睡眠良好": "Good sleep", "胡思乱想": "Overthinking", "第一胎": "First child", "第三胎": "Third child",
    "第二胎": "Second child", "第四胎": "Fourth child", "剖宫产分娩": "Cesarean section",
    "阴道分娩（顺产）": "Vaginal delivery", "顺产转为剖宫产分娩": "Vaginal to Cesarean",
    "早产（怀孕未满37周即分娩）": "Preterm (<37 weeks)", "足月分娩（孕38-42周分娩）": "Term (38-42 weeks)",
    "过期产（怀孕超过42周分娩）": "Post-term (>42 weeks)", "人工喂养（奶粉）": "Formula feeding",
    "母乳喂养": "Breastfeeding", "混合喂养（母乳+奶粉）": "Mixed feeding", "保姆（月嫂）": "Nanny/Maternity matron",
    "妈妈": "Mother", "婆婆": "Mother-in-law", "爱人": "Spouse"
}

@st.cache_resource
def load_model():
    with open("model_LR-L1.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_schema():
    with open("schema.json", "r", encoding="utf-8") as f:
        return json.load(f)

try:
    model = load_model()
    schema = load_schema()
except Exception as e:
    st.error(f"Failed to load model or schema: {e}")
    st.stop()

TH = schema["thresholds"]

def risk_level(p):
    if p < TH["risk_low"]:
        return "Low Risk", "#2e7d32"
    if p < TH["risk_high"]:
        return "Medium Risk", "#f9a825"
    return "High Risk", "#c62828"

DERIVED = {
    "ALT×FT4", "ALT×TSH", "FT4/TSH", "ALT/AST", "WBC/PLT",
    "睡眠综合", "压力合计",
}

def to_num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0

with st.form("form"):
    st.subheader("Please enter maternal information")
    vals = {}
    cols = st.columns(3)
    i = 0
    for f in schema["user_inputs"]:
        if f["name"] in DERIVED:
            continue
        with cols[i % 3]:
            label = LABEL_MAP.get(f["name"], f["name"])
            if f["type"] == "numeric":
                vals[f["name"]] = st.number_input(
                    label,
                    value=float(f["default"]),
                    format="%.4f",
                    key=f"n_{f['name']}",
                )
            else:
                cn_opts = f["options"] or [""]
                en_opts = [OPTION_MAP.get(o, o) for o in cn_opts]
                selected_en = st.selectbox(
                    label, en_opts, key=f"c_{f['name']}"
                )
                # 翻译回中文后存入字典，确保模型能识别
                vals[f["name"]] = cn_opts[en_opts.index(selected_en)]
        i += 1
    ok = st.form_submit_button("Calculate Risk")

if ok:
    alt = to_num(vals.get("ALT"))
    ft4 = to_num(vals.get("FT4"))
    tsh = to_num(vals.get("TSH"))
    ast = to_num(vals.get("AST"))
    wbc = to_num(vals.get("WBC"))
    plt = to_num(vals.get("PLT"))

    vals["ALT×FT4"] = alt * ft4
    vals["ALT×TSH"] = alt * tsh
    vals["FT4/TSH"] = ft4 / tsh if tsh else 0.0
    vals["ALT/AST"] = alt / ast if ast else 0.0
    vals["WBC/PLT"] = wbc / plt if plt else 0.0
    vals["睡眠综合"] = to_num(vals.get("睡眠质量打分")) + to_num(vals.get("睡眠时长打分"))
    vals["压力合计"] = to_num(vals.get("工作担忧")) + to_num(vals.get("经济担忧"))

    df = pd.DataFrame([vals], columns=schema["features"])

    try:
        prob = float(model.predict_proba(df)[:, 1][0])
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.dataframe(df.T)
        st.stop()

    level, color = risk_level(prob)

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("PPD Risk Probability", f"{prob:.1%}")
    c2.metric("Risk Level", level)
    c3.metric("Decision Threshold", TH["decision"])

    st.markdown(
        f"<h3 style='color:{color}'>Risk Level: {level}</h3>",
        unsafe_allow_html=True,
    )
    st.progress(min(prob, 1.0))

    if prob >= TH["decision"]:
        st.warning(f"Probability exceeds decision threshold {TH['decision']}. Further clinical evaluation is recommended.")
    else:
        st.info(f"Probability is below the decision threshold {TH['decision']}.")

    with st.expander("View Input Data"):
        st.dataframe(df.T)

st.markdown("---")
st.caption(
    "Risk stratification: Low Risk <0.184; Medium Risk 0.184–0.451; High Risk ≥0.451. "
    "Decision threshold 0.159. "
    "Disclaimer: This tool is for research and risk stratification only and cannot replace a doctor's diagnosis."
)
