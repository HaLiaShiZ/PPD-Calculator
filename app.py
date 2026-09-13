# app.py
import streamlit as st
import pandas as pd
import pickle, json

st.set_page_config(page_title="PPD 风险预测计算器", page_icon="🧠", layout="wide")
st.title("产后抑郁（PPD）风险预测计算器")
st.caption("基于 LR-L1 + sigmoid 模型；仅用于研究，不构成临床诊断。")

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
    st.error(f"模型或 schema 加载失败：{e}")
    st.stop()

TH = schema["thresholds"]

def risk_level(p):
    if p < TH["risk_low"]:
        return "低风险", "#2e7d32"
    if p < TH["risk_high"]:
        return "中风险", "#f9a825"
    return "高风险", "#c62828"

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
    st.subheader("请输入产妇信息")
    vals = {}
    cols = st.columns(3)
    i = 0
    for f in schema["user_inputs"]:
        if f["name"] in DERIVED:
            continue
        with cols[i % 3]:
            if f["type"] == "numeric":
                vals[f["name"]] = st.number_input(
                    f["name"],
                    value=float(f["default"]),
                    format="%.4f",
                    key=f"n_{f['name']}",
                )
            else:
                opts = f["options"] or [""]
                vals[f["name"]] = st.selectbox(
                    f["name"], opts, key=f"c_{f['name']}"
                )
        i += 1
    ok = st.form_submit_button("计算风险")

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
        st.error(f"预测失败：{e}")
        st.dataframe(df.T)
        st.stop()

    level, color = risk_level(prob)

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("PPD 风险概率", f"{prob:.1%}")
    c2.metric("风险等级", level)
    c3.metric("决策阈值", TH["decision"])

    st.markdown(
        f"<h3 style='color:{color}'>风险等级：{level}</h3>",
        unsafe_allow_html=True,
    )
    st.progress(min(prob, 1.0))

    if prob >= TH["decision"]:
        st.warning(f"概率超过决策阈值 {TH['decision']}，建议进一步临床评估。")
    else:
        st.info(f"概率低于决策阈值 {TH['decision']}。")

    with st.expander("查看输入数据"):
        st.dataframe(df.T)

st.markdown("---")
st.caption(
    "风险分层：低风险 <0.184；中风险 0.184–0.451；高风险 ≥0.451。"
    "决策阈值 0.159。"
    "免责声明：本工具仅用于科研和风险分层，不能替代医生诊断。"
)
