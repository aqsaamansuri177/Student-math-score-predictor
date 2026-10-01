"""
app.py  –  Streamlit frontend for the Student Math Score Predictor.

Run:
    streamlit run app.py
"""

import json
import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = "model.pkl"
META_PATH = "model_meta.json"
DATA_PATH = "StudentsPerformance.csv"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Math Score Predictor",
    page_icon="🎓",
    layout="centered",
)

# ── Auto-train if artefacts are missing ───────────────────────────────────────
if not os.path.exists(MODEL_PATH) or not os.path.exists(META_PATH):
    with st.spinner("Training models for the first time – please wait…"):
        import subprocess, sys
        subprocess.run([sys.executable, "train_model.py"], check=True)

# ── Load model + metadata ─────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(META_PATH) as f:
        meta = json.load(f)
    return model, meta

model, meta = load_artifacts()
label_maps: dict = meta["label_maps"]
feature_cols: list = meta["feature_cols"]
categorical_cols: list = meta["categorical_cols"]

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🎓 Student Math Score Predictor")
st.markdown(
    "Fill in the student profile below and click **Predict** "
    "to estimate their math exam score (0 – 100)."
)

# ── Model info banner ─────────────────────────────────────────────────────────
with st.expander("📊 Model comparison & selection", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Linear Regression RMSE", f"{meta['lr_rmse']:.3f}")
        st.metric("Linear Regression R²", f"{meta['lr_r2']:.4f}")
    with col2:
        st.metric("Random Forest RMSE", f"{meta['rf_rmse']:.3f}")
        st.metric("Random Forest R²", f"{meta['rf_r2']:.4f}")
    winner_color = "green" if meta["model_name"] == "Random Forest" else "blue"
    st.markdown(
        f"**Active model:** :{winner_color}[{meta['model_name']}]  "
        f"│  RMSE **{meta['rmse']:.3f}**  │  R² **{meta['r2']:.4f}**"
    )

st.divider()

# ── Input form ────────────────────────────────────────────────────────────────
st.subheader("Student Profile")

with st.form("prediction_form"):
    c1, c2 = st.columns(2)

    with c1:
        gender = st.selectbox(
            "Gender",
            options=label_maps["gender"],
        )
        ethnicity = st.selectbox(
            "Race / Ethnicity",
            options=sorted(label_maps["race/ethnicity"]),
        )
        parental_edu = st.selectbox(
            "Parental Level of Education",
            options=[
                "some high school",
                "high school",
                "some college",
                "associate's degree",
                "bachelor's degree",
                "master's degree",
            ],
        )

    with c2:
        lunch = st.selectbox(
            "Lunch Type",
            options=label_maps["lunch"],
        )
        test_prep = st.selectbox(
            "Test Preparation Course",
            options=label_maps["test preparation course"],
        )
        reading_score = st.slider(
            "Reading Score",
            min_value=0,
            max_value=100,
            value=70,
            step=1,
            help="Student's reading exam score (0–100)",
        )
        writing_score = st.slider(
            "Writing Score",
            min_value=0,
            max_value=100,
            value=70,
            step=1,
            help="Student's writing exam score (0–100)",
        )

    predict_btn = st.form_submit_button("🔮 Predict Math Score", type="primary")

# ── Prediction ────────────────────────────────────────────────────────────────
if predict_btn:
    # Encode categoricals using the stored label maps
    def encode(col: str, val: str) -> int:
        classes = label_maps[col]
        if val not in classes:
            # Fallback: pick the closest (shouldn't happen with locked dropdowns)
            return 0
        return classes.index(val)

    row = [
        encode("gender", gender),
        encode("race/ethnicity", ethnicity),
        encode("parental level of education", parental_edu),
        encode("lunch", lunch),
        encode("test preparation course", test_prep),
        reading_score,
        writing_score,
    ]

    pred = model.predict(np.array([row]))[0]
    pred_clipped = float(np.clip(pred, 0, 100))

    st.divider()
    st.subheader("Prediction Result")

    result_col, gauge_col = st.columns([1, 2])
    with result_col:
        st.metric(
            label="Predicted Math Score",
            value=f"{pred_clipped:.1f} / 100",
        )
        if pred_clipped >= 80:
            st.success("Excellent performance predicted! 🏆")
        elif pred_clipped >= 60:
            st.info("Good performance predicted. 📘")
        elif pred_clipped >= 40:
            st.warning("Moderate performance – some support may help. 📝")
        else:
            st.error("Low performance predicted – consider extra resources. ⚠️")

    with gauge_col:
        # Progress-bar style gauge
        st.markdown(f"**Score band: {pred_clipped:.1f}%**")
        st.progress(int(pred_clipped))

# ── Feature Importance Chart ──────────────────────────────────────────────────
st.divider()
st.subheader("📈 Feature Importance")
st.caption(
    "For Random Forest: mean decrease in impurity. "
    "For Linear Regression: normalised absolute coefficient."
)

importance_df = pd.DataFrame(
    {
        "Feature": feature_cols,
        "Importance": meta["importances"],
    }
).sort_values("Importance", ascending=True)

# Use Streamlit's built-in horizontal bar chart via st.bar_chart
# We build it with altair for a horizontal layout
try:
    import altair as alt

    chart = (
        alt.Chart(importance_df)
        .mark_bar(color="#3b82d4")
        .encode(
            x=alt.X("Importance:Q", title="Importance Score"),
            y=alt.Y("Feature:N", sort="-x", title=""),
            tooltip=["Feature", alt.Tooltip("Importance:Q", format=".4f")],
        )
        .properties(height=260)
    )
    st.altair_chart(chart, use_container_width=True)
except ImportError:
    # Fallback to native bar chart (vertical)
    st.bar_chart(importance_df.set_index("Feature")["Importance"])

# ── Dataset peek ─────────────────────────────────────────────────────────────
with st.expander("🗂️ Dataset preview (first 10 rows)", expanded=False):
    try:
        df_preview = pd.read_csv(DATA_PATH)
        st.dataframe(df_preview.head(10), use_container_width=True)
    except FileNotFoundError:
        st.info("Dataset file not found in the current directory.")

st.divider()
st.caption("Student Math Score Predictor · powered by scikit-learn & Streamlit")
