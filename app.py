# app.py
import os
import json
import joblib
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import requests

from rules import triage_rules

# Optional Lottie support
try:
    from streamlit_lottie import st_lottie
    LOTTIE_ENABLED = True
except Exception:
    LOTTIE_ENABLED = False

# --- Paths & config ---
DATA_PATH = "synthetic_vignettes.csv"
LOG_DIR = "logs"
LOG_PATH = os.path.join(LOG_DIR, "triage_logs.csv")
MODEL_PATH = "model.pkl"
VECT_PATH = "vectorizer.json"   # JSON saved by train script
META_PATH = "model_meta.json"

CATEGORIES = ["Emergency", "Urgent", "Routine", "Self-care"]

st.set_page_config(page_title="Tiny Medical Triage Assistant", layout="wide",
                   initial_sidebar_state="expanded")

# --- Styles & Fonts ---
st.markdown(""" 
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap" rel="stylesheet">

<style>
:root{
  --bg:#0f172a;
  --card:#1e293b;
  --muted:#94a3b8;
  --accent:#6366f1;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: var(--bg);
    color: #f1f5f9;
}

.header {
  border-radius: 12px;
  padding: 20px;
  background: linear-gradient(90deg,#0ea5e9, #8b5cf6);
  color: white;
  box-shadow: 0 6px 20px rgba(12, 15, 30, 0.18);
}

.hero-title {
  font-size: 28px;
  font-weight: 800;
  margin-bottom: 4px;
}

.hero-sub {
  opacity: 0.95;
  margin-bottom: 6px;
}

.card {
  background: var(--card);
  border-radius: 12px;
  padding: 16px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.25);
}

.small-muted { color: var(--muted); font-size:13px; }

.banner-emergency {background: linear-gradient(90deg,#ef4444,#f87171); padding:12px; border-radius:10px; color:white; font-weight:700;}
.banner-urgent {background: linear-gradient(90deg,#f59e0b,#fbbf24); padding:12px; border-radius:10px; color:#1e293b; font-weight:700;}
.banner-routine {background: linear-gradient(90deg,#4ade80,#86efac); padding:12px; border-radius:10px; color:#052e16; font-weight:700;}
.banner-self {background: linear-gradient(90deg,#60a5fa,#93c5fd); padding:12px; border-radius:10px; color:#0c4a6e; font-weight:700;}

.stButton > button {
  border-radius: 10px;
  padding: 8px 14px;
  background: linear-gradient(90deg,#4f46e5,#06b6d4);
  color: white;
  border: none;
  font-weight: 600;
  box-shadow: 0 6px 18px rgba(99,102,241,0.15);
  cursor: pointer;
}
.stButton > button:hover {
  opacity: 0.9;
  transform: translateY(-1px);
}

[data-baseweb="select"] > div {
    border-radius: 10px !important;
}

.kv { font-weight:600; color:#e2e8f0; }
</style>
""", unsafe_allow_html=True)

# --- Helper: Lottie loader (cached) ---
@st.cache_data(show_spinner=False)
def load_lottie_url(url: str):
    try:
        r = requests.get(url, timeout=6)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None

LOTTIE_HEART = "https://assets7.lottiefiles.com/packages/lf20_jbrw3hcz.json"

# --- Data load/generate ---
@st.cache_data
def load_demo(path=DATA_PATH, n=600):
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        from data_gen import generate_csv
        generate_csv(path, n=n)
        return pd.read_csv(path)

df_demo = load_demo()

# --- Load ML model if present ---
model = None
vectorizer = None
meta = None
if os.path.exists(MODEL_PATH) and os.path.exists(VECT_PATH) and os.path.exists(META_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        with open(VECT_PATH, "r", encoding="utf-8") as f:
            vectorizer = json.load(f)
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as e:
        st.sidebar.error("Error loading ML artifacts: " + str(e))

# --- Header / Hero ---
hero_col_left, hero_col_right = st.columns([3,2])
with hero_col_left:
    st.markdown('<div class="header">', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">🩺 Tiny Medical Triage Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Hybrid rules + ML triage demo. For education only, not medical advice.</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="small-muted">Enter symptoms on the left. The app shows rule-based and optional '
                'ML predictions, explains the decision, and logs anonymized cases.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
with hero_col_right:
    if LOTTIE_ENABLED:
        lottie_json = load_lottie_url(LOTTIE_HEART)
        if lottie_json:
            try:
                st_lottie(lottie_json, height=160, key="hero_lottie")
            except Exception:
                st.empty()
        else:
            st.empty()
    else:
        st.empty()

st.markdown("---")

# --- Sidebar: inputs + quick actions ---
st.sidebar.markdown("## Patient quick input")
age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=28, step=1)
duration = st.sidebar.selectbox("Duration of main symptoms (days)", options=[1,2,3,4,5,7,10,14], index=0)
st.sidebar.markdown("### Select symptoms (one or more)")

SYMPTOMS = sorted(set(sum([s.split(";") for s in df_demo["symptoms"].dropna().tolist()], [])))
selected_symptoms = st.sidebar.multiselect("Symptoms", options=SYMPTOMS, default=["Fever"])
st.sidebar.markdown("### Any red-flag signs?")
selected_redflags = st.sidebar.multiselect(
    "Red flags",
    options=["Severe bleeding", "Unconscious", "Slurred speech", "Fainting", "Severe chest pain"],
    default=[]
)

st.sidebar.markdown("---")
if st.sidebar.button("About"):
    st.sidebar.info("Educational project. Combines rule-based triage with an optional ML model. "
                    "Not a substitute for professional medical advice.")

# --- Layout: main content ---
left, right = st.columns([2,1])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Patient inputs")
    st.write(f"<span class='kv'>Age:</span> {age}   •   <span class='kv'>Duration:</span> {duration} days",
             unsafe_allow_html=True)
    st.write("**Symptoms:** " + (", ".join(selected_symptoms) if selected_symptoms else "—"))
    st.write("**Red flags:** " + (", ".join(selected_redflags) if selected_redflags else "—"))
    st.markdown("</div>", unsafe_allow_html=True)

    # --- run rule engine ---
    rule_cat, reasons, rule_probs = triage_rules(selected_symptoms, selected_redflags, age, duration)

    # --- ML prediction if available ---
    def build_feature_vector(symptoms, redflags, age_v, duration_v, vectorizer_obj):
        feat_cols = vectorizer_obj.get("feature_columns", [])
        fv = pd.Series(0, index=feat_cols, dtype=int)
        for s in symptoms:
            key = f"sym_{s}"
            if key in fv.index:
                fv[key] = 1
        for rf in redflags:
            key = f"rf_{rf}"
            if key in fv.index:
                fv[key] = 1
        if "age" in fv.index:
            fv["age"] = int(age_v)
        if "duration_days" in fv.index:
            fv["duration_days"] = int(duration_v)
        return fv.values.reshape(1, -1), fv.index.tolist()

    ml_probs = {c: 0.0 for c in CATEGORIES}
    ml_pred_label = None
    if model is not None and vectorizer is not None:
        try:
            Xcase, feat_names = build_feature_vector(selected_symptoms, selected_redflags, age, duration, vectorizer)
            proba = model.predict_proba(Xcase)[0]
            if meta and "idx2label" in meta:
                idx2label = {int(k): v for k, v in meta["idx2label"].items()}
                ml_probs = {idx2label[i]: float(proba[i]) for i in range(len(proba))}
            else:
                ml_probs = {CATEGORIES[i]: float(proba[i]) for i in range(len(proba))}
            ml_pred_label = max(ml_probs.items(), key=lambda x: x[1])[0]
        except Exception as e:
            st.warning("ML prediction error: " + str(e))

    # --- Hybrid decision (rules conservative) ---
    final_cat = rule_cat
    final_reasons = ["Based on rule engine by default."]
    if rule_cat == "Emergency":
        final_cat = "Emergency"
        final_reasons = ["Rule-based emergency detected (override)."]
    else:
        if ml_pred_label:
            p_em = ml_probs.get("Emergency", 0.0)
            p_top = ml_probs.get(ml_pred_label, 0.0)
            if p_em >= 0.60:
                final_cat = "Emergency"
                final_reasons = [f"ML strongly predicts Emergency (p={p_em:.2f})."]
            elif p_top >= 0.60 and ml_pred_label != rule_cat:
                final_cat = ml_pred_label
                final_reasons = [f"ML predicts {ml_pred_label} with high confidence (p={p_top:.2f})."]
            else:
                final_cat = rule_cat
                final_reasons = ["No high-confidence ML override; using rules."]
        else:
            final_cat = rule_cat
            final_reasons = ["ML not available; using rules."]

    # --- banner + action ---
    banner_class = {
        "Emergency": "banner-emergency",
        "Urgent": "banner-urgent",
        "Routine": "banner-routine",
        "Self-care": "banner-self"
    }.get(final_cat, "banner-routine")
    st.markdown(f"<div class='{banner_class}'>{final_cat} — Final recommended action</div>", unsafe_allow_html=True)

    if final_cat == "Emergency":
        st.markdown("**Action:** Call emergency services now or go to the nearest ER. This is potentially life-threatening.")
    elif final_cat == "Urgent":
        st.markdown("**Action:** Seek urgent clinical review (within 24 hours).")
    elif final_cat == "Routine":
        st.markdown("**Action:** Book with your GP / clinician for assessment.")
    else:
        st.markdown("**Action:** Home care and monitor; return if symptoms worsen.")

    # --- explanations ---
    with st.expander("Why this recommendation? (rule + ML)"):
        st.subheader("Rule-based reasoning")
        for r in reasons:
            st.write(" • " + r)

        st.subheader("Model-based probabilities")
        if model is not None and ml_probs is not None:
            prob_df = pd.DataFrame(list(ml_probs.items()), columns=["Category", "Probability"])
            st.table(prob_df.set_index("Category"))
        else:
            st.write("Model not available or failed to predict. Only rules are used.")

    # --- ML probability chart ---
    if model is not None and ml_probs:
        st.subheader("ML predicted probabilities")
        prob_df = pd.DataFrame(list(ml_probs.items()), columns=["Category", "Probability"])
        fig, ax = plt.subplots(figsize=(7, 1.4))
        sns.barplot(x="Probability", y="Category",
                    data=prob_df.sort_values("Probability", ascending=True), ax=ax)
        ax.set_xlim(0, 1)
        ax.set_xlabel("")
        ax.set_ylabel("")
        for p in ax.patches:
            ax.text(p.get_width() + 0.01, p.get_y() + p.get_height() / 2,
                    f"{p.get_width():.2f}", va='center')
        st.pyplot(fig)

    # --- Save / logs (anonymized) ---
    def anonymize_record(age_val:int, symptoms:list, redflags:list, duration_val:int,
                         category:str, reasons_list:list, probs:dict):
        if age_val < 2: age_group = "<2"
        elif age_val < 12: age_group = "2-11"
        elif age_val < 18: age_group = "12-17"
        elif age_val < 30: age_group = "18-29"
        elif age_val < 45: age_group = "30-44"
        elif age_val < 65: age_group = "45-64"
        else: age_group = "65+"
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "age_group": age_group,
            "duration_days": duration_val,
            "symptoms": ";".join(symptoms) if symptoms else "",
            "red_flags": ";".join(redflags) if redflags else "",
            "triage_category": category,
            "reasons": "||".join(reasons_list),
            "probs_json": json.dumps(probs)
        }

    def append_log(record: dict, path=LOG_PATH):
        import csv
        fieldnames = ["timestamp", "age_group", "duration_days",
                      "symptoms", "red_flags", "triage_category",
                      "reasons", "probs_json"]
        header_needed = not os.path.exists(path) or os.path.getsize(path) == 0
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if header_needed:
                writer.writeheader()
            writer.writerow(record)

    st.markdown("### Save / Log this case (anonymized)")
    col_save_left, col_save_right = st.columns([3, 1])
    with col_save_left:
        save_note = st.text_input("Optional short note (non-identifying)", value="", max_chars=140)
    with col_save_right:
        if st.button("Save case to logs"):
            rec = anonymize_record(age, selected_symptoms, selected_redflags,
                                   duration, final_cat, final_reasons,
                                   ml_probs if ml_probs else rule_probs)
            if save_note.strip():
                rec["reasons"] = rec["reasons"] + " || note:" + save_note.strip()
            os.makedirs(LOG_DIR, exist_ok=True)
            append_log(rec)
            st.success(f"Saved to {LOG_PATH} (anonymized).")

    st.markdown("#### Recent logged cases")
    if os.path.exists(LOG_PATH):
        try:
            logs_df = pd.read_csv(LOG_PATH)
            st.dataframe(logs_df.tail(8).reset_index(drop=True))
            csv_bytes = logs_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download logs (CSV)", data=csv_bytes,
                               file_name="triage_logs.csv", mime="text/csv")
        except Exception as e:
            st.error("Error reading logs: " + str(e))
    else:
        st.info("No logs yet — save cases to build a training set.")

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Quick EDA & Model info")
    st.write("Dataset & model snapshot (for report figures).")

    st.markdown("**Top symptoms (demo data)**")
    all_sym = df_demo["symptoms"].dropna().str.split(";").explode()
    top = all_sym.value_counts().nlargest(8)
    fig2, ax2 = plt.subplots(figsize=(4, 3))
    sns.barplot(x=top.values, y=top.index, ax=ax2)
    ax2.set_xlabel("Count")
    st.pyplot(fig2)

    st.markdown("**Age distribution**")
    fig3, ax3 = plt.subplots(figsize=(4, 2.2))
    sns.histplot(df_demo["age"], bins=12, ax=ax3)
    ax3.set_xlabel("Age")
    st.pyplot(fig3)

    if meta:
        st.markdown("**Model summary**")
        st.write(f"Trained on: {meta.get('trained_on', '-')}")
        st.write(f"Train size: {meta.get('n_train', '-')}, Test size: {meta.get('n_test', '-')}")
        if vectorizer:
            st.write(f"Features: {len(vectorizer.get('feature_columns', []))}")
        fi = os.path.join("model_figs", "feature_importances.png")
        cm = os.path.join("model_figs", "confusion_matrix.png")
        if os.path.exists(fi):
            st.image(fi, caption="Feature importances", use_container_width=True)
        if os.path.exists(cm):
            st.image(cm, caption="Confusion matrix", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("<div style='font-size:12px;color:#94a3b8;'>Demo & educational tool. "
            "It is NOT for clinical use. Always contact emergency services in life-threatening situations.</div>",
            unsafe_allow_html=True)
