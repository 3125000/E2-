import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ==== Page setup ====
st.set_page_config(page_title="Gn Starting Protocol Prediction", layout="wide")
st.title("🧬 Personalized Gn Starting Protocol Prediction System")
st.markdown("🔍 Please fill in the baseline information and up to three sets of hormone monitoring data (some values can be missing).")

# ==== Model loading ====
MODEL_DIR = ""
reg_dose_model = joblib.load(MODEL_DIR + "reg_dose_model.pkl")
reg_totaldose_model = joblib.load(MODEL_DIR + "reg_totaldose_model.pkl")
reg_totaldays_model = joblib.load(MODEL_DIR + "reg_totaldays_model.pkl")
clf_drug_model = joblib.load(MODEL_DIR + "clf_drug_model.pkl")
clf_protocol_model = joblib.load(MODEL_DIR + "clf_protocol_model.pkl")
clf_trigger_model = joblib.load(MODEL_DIR + "clf_triggerday_model.pkl")
drug_encoder = joblib.load(MODEL_DIR + "drug_encoder.pkl")
protocol_encoder = joblib.load(MODEL_DIR + "protocol_encoder.pkl")
trigger_label_day_mapping = joblib.load(MODEL_DIR + "trigger_label_day_mapping.pkl")
e2_percentiles = joblib.load(MODEL_DIR + "e2_percentiles.pkl")

# ==== Feature definitions ====
# (A) 模型训练时的列顺序（必须保持不变）
MODEL_CORE_FEATURES = [
    "年龄", "体重指数", "(基础内分泌)FSH", "(基础内分泌)E2", "(基础内分泌)PRL",
    "(基础内分泌)LH", "(基础内分泌)T", "(基础内分泌)AMH", "左窦卵泡数", "右窦卵泡数"
]
MODEL_DYNAMIC_FEATURES = [
    "血E2_1", "血LH_1", "血FSH_1", "血P_1", "Day_1",
    "血E2_2", "血LH_2", "血FSH_2", "血P_2", "Day_2",
    "血E2_3", "血LH_3", "血FSH_3", "血P_3", "Day_3",
    "最大卵泡测定日3", "左侧最大卵泡直径3", "右侧最大卵巢直径3"
]
MODEL_ALL_FEATURES = MODEL_CORE_FEATURES + MODEL_DYNAMIC_FEATURES

# (B) 仅用于界面展示的顺序（可随意调整，不影响模型输入）
UI_CORE_ORDER = [
    "年龄", "体重指数",
    "(基础内分泌)FSH", "(基础内分泌)LH", "(基础内分泌)PRL",
    "(基础内分泌)E2", "(基础内分泌)T", "(基础内分泌)AMH",
    "左窦卵泡数", "右窦卵泡数"
]

# ==== User input ====
user_input = {}
st.sidebar.header("📌 Baseline Information")
for feat in UI_CORE_ORDER:
    user_input[feat] = st.sidebar.number_input(feat, value=0.0, format="%.2f")

st.sidebar.header("📊 Dynamic Monitoring Data (can be partially missing)")
for i in range(1, 3 + 1):
    st.sidebar.markdown(f"###### Monitoring {i}")
    for prefix in ["血E2", "血LH", "血FSH", "血P"]:
        key = f"{prefix}_{i}"
        user_input[key] = st.sidebar.number_input(key, value=0.0, format="%.2f")
    user_input[f"Day_{i}"] = st.sidebar.number_input(f"Day_{i}", value=0.0, format="%.0f")

# 额外三个输入
user_input["最大卵泡测定日3"] = st.sidebar.number_input("Max follicle measurement day 3", value=0.0, format="%.0f")
user_input["左侧最大卵泡直径3"] = st.sidebar.number_input("Left max follicle diameter 3", value=0.0, format="%.2f")
user_input["右侧最大卵巢直径3"] = st.sidebar.number_input("Right max follicle diameter 3", value=0.0, format="%.2f")

# ==== Construct input dataframe ====
input_df = pd.DataFrame([user_input]).astype(float)
# 可根据需要替换为“训练集的中位数/均值”来填充
input_df_filled = input_df.fillna(0.0)

# ！！！关键：传给模型时，严格按“训练时顺序”取列
X_core = input_df_filled[MODEL_CORE_FEATURES]
X_all = input_df_filled[MODEL_ALL_FEATURES]

# ==== Model predictions ====
dose_pred = reg_dose_model.predict(X_core)[0]
total_dose_pred = reg_totaldose_model.predict(X_core)[0]
total_days_pred = reg_totaldays_model.predict(X_all)[0]
drug_pred = drug_encoder.inverse_transform(clf_drug_model.predict(X_core))[0]
protocol_pred = protocol_encoder.inverse_transform(clf_protocol_model.predict(X_core))[0]
trigger_label = clf_trigger_model.predict(X_all)[0]
trigger_day_pred = trigger_label_day_mapping.get(trigger_label, "N/A")

# ==== Results ====
st.subheader("🎯 Prediction Results")
st.markdown(f"""
- 💉 **Recommended Gn starting dose**: {dose_pred:.0f} IU  
- 💊 **Recommended drug type**: {drug_pred}  
- 🧩 **Recommended protocol**: {protocol_pred}  
- 📦 **Predicted total Gn dose**: {total_dose_pred:.0f} IU  
- ⏳ **Predicted total Gn days**: {total_days_pred:.1f} days  
- 🚦 **Recommended Trigger day**: Day {trigger_day_pred} (class {trigger_label})
""")

# ==== Utility functions ====
def get_dist_stats(dist_obj):
    if dist_obj is None:
        return None
    if isinstance(dist_obj, dict):
        return {
            "p25": dist_obj.get("p25", None),
            "p50": dist_obj.get("p50", None),
            "p75": dist_obj.get("p75", None),
            "values": None
        }
    arr = np.asarray(dist_obj, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return None
    return {
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "values": arr
    }

def percentile_rank(values_arr, x):
    if values_arr is None or np.isnan(x):
        return None
    return int(round((np.sum(values_arr < x) / len(values_arr)) * 100))

# ==== E2 percentile plots ====
st.subheader("📈 Serum E2 percentile plot")
fig, ax = plt.subplots()
percentile_text = []

for i, key in enumerate(["血E2_1", "血E2_2", "血E2_3"], start=1):
    dist_obj = e2_percentiles.get(key, None)
    stats = get_dist_stats(dist_obj)
    val = input_df_filled[key].values[0]

    if stats is not None:
        if stats["p25"] is not None and stats["p75"] is not None:
            ax.plot([i, i], [stats["p25"], stats["p75"]], linewidth=6)
        if not np.isnan(val):
            pr = percentile_rank(stats["values"], val)
            ax.scatter(i, val, s=40)
            label_txt = f"{val:.0f}" + (f" (P{pr})" if pr is not None else "")
            ax.text(i + 0.1, val, label_txt, fontsize=9)
            if pr is not None:
                percentile_text.append(f"- **{key}**: {val:.0f} pg/mL, at **P{pr}**")
            else:
                percentile_text.append(f"- **{key}**: {val:.0f} pg/mL (reference P25–P75)")

ax.set_xticks([1, 2, 3])
ax.set_xticklabels(["E2_1", "E2_2", "E2_3"])
ax.set_ylabel("E2 (pg/mL)")
ax.set_title("Serum E2 percentile plot")
st.pyplot(fig)

if percentile_text:
    st.markdown("🔢 **Percentile explanation:**")
    for text in percentile_text:
        st.markdown(text)

# ==== Baseline E2 percentile plot ====
st.subheader("📊 Baseline E2 percentile plot")
base_e2_key = "(基础内分泌)E2"
base_e2_val = input_df_filled[base_e2_key].values[0]
base_stats = get_dist_stats(e2_percentiles.get("基础E2", None))

if base_stats is not None and not np.isnan(base_e2_val):
    fig2, ax2 = plt.subplots()
    if base_stats["p25"] is not None and base_stats["p75"] is not None:
        ax2.plot([1, 1], [base_stats["p25"], base_stats["p75"]], linewidth=6, label="P25–P75")
    if base_stats["p50"] is not None:
        ax2.axhline(base_stats["p50"], linestyle='--', label="P50")
    ax2.scatter(1, base_e2_val, s=80, label=f"Your value: {base_e2_val:.0f}")
    ax2.set_xlim(0.5, 1.5)
    ax2.set_xticks([1])
    ax2.set_xticklabels(["Baseline E2"])
    ax2.set_ylabel("E2 (pg/mL)")
    ax2.set_title("Baseline E2 percentile plot")
    ax2.legend()
    st.pyplot(fig2)

    pr_base = percentile_rank(base_stats["values"], base_e2_val)
    if pr_base is not None:
        st.markdown(f"🔢 Your **Baseline E2** value is **{base_e2_val:.0f} pg/mL**, at about **P{pr_base}**.")
    else:
        st.markdown(f"🔢 Your **Baseline E2** value is **{base_e2_val:.0f} pg/mL** (reference P25–P75).")
else:
    st.warning("⚠️ Baseline E2 missing or no reference data available, cannot display percentile plot.")

