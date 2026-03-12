import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="RA Prediction App",
    page_icon="🧬",
    layout="wide",
)

# ==============================
# Paths
# ==============================
BASE_DIR = Path(".")
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"


# ==============================
# Cached loaders
# ==============================
@st.cache_data
def load_metadata():
    candidates = [
        RESULTS_DIR / "run_metadata.json",
        BASE_DIR / "run_metadata.json",
    ]
    for path in candidates:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8")), path
            except Exception:
                return None, path
    return None, None


@st.cache_resource
def load_artifact():
    candidates = [
        MODELS_DIR / "xgboost_model.joblib",
        BASE_DIR / "xgboost_model.joblib",
    ]
    for path in candidates:
        if path.exists():
            try:
                artifact = joblib.load(path)
                return artifact, path
            except Exception as e:
                return {"error": str(e)}, path
    return None, None


def plot_demo_distribution():
    labels = ["Sample 1", "Sample 2", "Sample 3", "Sample 4"]
    values = [0.10, 0.35, 0.60, 0.85]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(labels, values)
    ax.set_title("Synthetic Demo Risk Pattern")
    ax.set_xlabel("Demo Samples")
    ax.set_ylabel("Relative Pattern")
    st.pyplot(fig)


def generate_synthetic_samples(feature_cols):
    def sample_low():
        return {feature: 0 for feature in feature_cols}

    def sample_mild():
        values = {}
        for i, feature in enumerate(feature_cols):
            values[feature] = 1 if i % 5 == 0 else 0
        return values

    def sample_moderate():
        values = {}
        for i, feature in enumerate(feature_cols):
            if i % 4 == 0:
                values[feature] = 2
            elif i % 2 == 0:
                values[feature] = 1
            else:
                values[feature] = 0
        return values

    def sample_cyclic():
        return {feature: i % 3 for i, feature in enumerate(feature_cols)}

    return {
        "Demo Sample 1 - Low Pattern": sample_low(),
        "Demo Sample 2 - Mild Pattern": sample_mild(),
        "Demo Sample 3 - Moderate Pattern": sample_moderate(),
        "Demo Sample 4 - Cyclic Pattern": sample_cyclic(),
    }


def run_prediction(artifact, input_df):
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]

    aligned_input = input_df.reindex(columns=feature_columns, fill_value=0)

    if hasattr(model, "predict_proba"):
        score = float(model.predict_proba(aligned_input)[0][1])
    else:
        score = float(model.predict(aligned_input)[0])

    prediction = 1 if score >= 0.5 else 0
    return prediction, score


# ==============================
# Sidebar
# ==============================
st.sidebar.title("RA App Navigation")
page = st.sidebar.radio(
    "Select a page",
    [
        "Home",
        "Sample Testing",
        "Run Metadata",
    ],
)

st.sidebar.markdown("---")
st.sidebar.info(
    "This application does not display confidential dataset rows. "
    "It only uses synthetic demo samples for testing."
)

artifact, artifact_path = load_artifact()

# ==============================
# Home
# ==============================
if page == "Home":
    st.title("Rheumatoid Arthritis Prediction App")
    st.write(
        "This Streamlit application uses one trained model only (XGBoost) "
        "and one encoding strategy only (Additive 0/1/2)."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Individuals", "2062")
    col2.metric("Original SNPs", "531,689")
    col3.metric("Refined SNP Panel", "313")
    col4.metric("Final Additive SNPs", "212")

    st.subheader("Application Scope")
    st.markdown(
        """
        - One model only: **XGBoost**
        - One encoding only: **Additive encoding (0/1/2)**
        - No confidential dataset preview is shown
        - Four synthetic demo samples for testing
        - Optional metadata display
        """
    )

    if artifact is None:
        st.error("Model artifact not found. Please place xgboost_model.joblib inside the models folder.")
    elif isinstance(artifact, dict) and "error" in artifact:
        st.error(f"Failed to load model artifact: {artifact['error']}")
    else:
        st.success(f"Loaded model artifact from: {artifact_path}")

        feature_count = len(artifact.get("feature_columns", []))
        st.write(f"Detected feature count: {feature_count}")
        st.write(f"Model name: {artifact.get('model_name', 'Unknown')}")
        st.write(f"Encoding: {artifact.get('encoding', 'Unknown')}")

    st.subheader("Demo Overview")
    plot_demo_distribution()

# ==============================
# Sample Testing
# ==============================
elif page == "Sample Testing":
    st.title("Sample Testing")
    st.write(
        "This page uses synthetic additive-encoded demo samples only. "
        "No confidential dataset records are displayed."
    )

    if artifact is None:
        st.error("Model artifact not found. Please place xgboost_model.joblib inside the models folder.")
    elif isinstance(artifact, dict) and "error" in artifact:
        st.error(f"Failed to load model artifact: {artifact['error']}")
    else:
        feature_cols = artifact.get("feature_columns", [])
        if not feature_cols:
            st.error("No feature columns were found inside the saved artifact.")
        else:
            synthetic_samples = generate_synthetic_samples(feature_cols)

            selected_sample_name = st.selectbox(
                "Choose a synthetic demo sample",
                list(synthetic_samples.keys())
            )

            selected_sample = synthetic_samples[selected_sample_name]
            input_df = pd.DataFrame([selected_sample])

            st.subheader("Selected Sample Preview")
            preview_cols = feature_cols[:12]
            st.dataframe(input_df[preview_cols], use_container_width=True)
            st.caption("Only the first 12 additive features are shown for demonstration.")

            if st.button("Run Prediction"):
                prediction, score = run_prediction(artifact, input_df)

                st.success(f"Predicted Class: {prediction}")
                st.write(f"Predicted Risk Score: {score:.3f}")
                st.success("Prediction generated using the saved XGBoost model.")

# ==============================
# Run Metadata
# ==============================
elif page == "Run Metadata":
    st.title("Run Metadata")

    metadata, path = load_metadata()
    if metadata is None:
        st.warning("No run_metadata.json file was found.")
    else:
        st.success(f"Loaded metadata from: {path}")
        st.json(metadata)