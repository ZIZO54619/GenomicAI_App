import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="GenomicAI - RA Prediction",
    page_icon="🧬",
    layout="wide",
)

# ==============================
# Paths
# ==============================
BASE_DIR = Path(".")
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"


# ==============================
# Loaders
# ==============================
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
                return artifact, path, None
            except Exception as e:
                return None, path, str(e)

    return None, None, "xgboost_model.joblib was not found."


@st.cache_data
def load_demo_samples():
    candidates = [
        DATA_DIR / "demo_samples.csv",
        BASE_DIR / "demo_samples.csv",
    ]

    for path in candidates:
        if path.exists():
            try:
                df = pd.read_csv(path)
                return df, path, None
            except Exception as e:
                return None, path, str(e)

    return None, None, "demo_samples.csv was not found."


# ==============================
# Helper functions
# ==============================
def normalize_expected_label(value):
    if pd.isna(value):
        return None

    text = str(value).strip().lower()

    if text in {"1", "ra", "case", "patient", "positive"}:
        return 1
    if text in {"0", "control", "healthy", "normal", "negative"}:
        return 0

    return None


def class_to_text(label):
    return "RA" if int(label) == 1 else "Control"


def probability_to_percent(x):
    return f"{x * 100:.2f}%"


def run_prediction(artifact, X):
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]

    X_aligned = X.reindex(columns=feature_columns, fill_value=0)

    if hasattr(model, "predict_proba"):
        prob_ra = model.predict_proba(X_aligned)[:, 1]
    else:
        raw_pred = model.predict(X_aligned)
        prob_ra = np.array(raw_pred, dtype=float)

    pred = (prob_ra >= 0.5).astype(int)
    return pred, prob_ra


def build_template(feature_columns):
    return pd.DataFrame(columns=["SampleName"] + feature_columns)


# ==============================
# Page functions
# ==============================
def show_home_page(feature_count):
    st.title("GenomicAI: AI-Based Prediction of Rheumatoid Arthritis")

    st.write(
        "GenomicAI is a healthcare technology project that uses artificial intelligence "
        "and genomic data to support the early prediction of Rheumatoid Arthritis (RA). "
        "The goal of the project is to help identify individuals with higher genetic "
        "susceptibility before clear clinical symptoms appear."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Individuals", "2,062")
    col2.metric("Original Autosomal SNPs", "531,689")
    col3.metric("Refined RA SNP Panel", "313")
    col4.metric("Final Additive Features", str(feature_count))

    st.subheader("The Problem")
    st.write(
        "Rheumatoid Arthritis is a chronic autoimmune disease that may progress silently "
        "before clear clinical signs appear. Delayed diagnosis can lead to irreversible "
        "joint damage, disability, and increased treatment costs. Traditional diagnosis "
        "often depends on symptoms and laboratory findings that may appear only after the "
        "disease has already progressed."
    )

    st.subheader("Our Solution")
    st.write(
        "GenomicAI addresses this challenge by analyzing genomic variations known as SNPs "
        "and transforming them into machine-learning-ready features. The deployed system "
        "uses one XGBoost model trained on additive genomic encoding to estimate the risk "
        "of Rheumatoid Arthritis."
    )

    st.subheader("How the Project Works")
    st.markdown(
        f"""
        - Start with raw genomic genotype data  
        - Validate and clean the dataset  
        - Select Rheumatoid Arthritis-associated SNPs  
        - Impute missing genetic values  
        - Transform the final retained SNPs into additive features  
        - Use one deployed **XGBoost** model to generate RA predictions  

        The full pipeline reduced the data from **531,689 raw SNPs** to **313 refined RA-related SNPs**,  
        then to **{feature_count} final additive features** used by the model.
        """
    )

    st.subheader("Expected Value")
    st.write(
        "By supporting earlier risk prediction, GenomicAI can help improve clinical "
        "decision-making, enable earlier monitoring and intervention, reduce long-term "
        "healthcare costs, and contribute to the advancement of precision medicine."
    )

    st.info(
        "This application is a research and demonstration prototype built to present the "
        "core idea, workflow, and prediction interface of the project."
    )


def show_prediction_page(artifact):
    st.title("Prediction Center")
    st.write(
        "This section demonstrates how GenomicAI can generate Rheumatoid Arthritis risk "
        "predictions from additive-encoded genomic features. Users can either test predefined "
        "demo samples or upload a compatible CSV file to obtain model predictions."
    )

    feature_columns = artifact.get("feature_columns", [])
    if not feature_columns:
        st.error("No feature columns were found inside the saved model artifact.")
        return

    input_mode = st.radio(
        "Select input mode",
        ["Predefined Demo Samples", "Upload CSV File"],
        horizontal=True,
    )

    if input_mode == "Predefined Demo Samples":
        demo_df, demo_path, demo_error = load_demo_samples()

        if demo_df is None:
            st.warning(
                "No demo_samples.csv file was found. "
                "To use this mode, add data/demo_samples.csv to the repository."
            )
            st.info(
                "Expected columns: DisplayName, ExpectedLabel, and all additive feature columns required by the model."
            )
            return

        required_missing = [c for c in feature_columns if c not in demo_df.columns]
        if required_missing:
            st.error(
                f"The demo_samples.csv file is missing {len(required_missing)} required feature columns."
            )
            st.write("First missing columns:", required_missing[:10])
            return

        name_col = "DisplayName" if "DisplayName" in demo_df.columns else None
        label_col = "ExpectedLabel" if "ExpectedLabel" in demo_df.columns else None

        if name_col is None:
            demo_df = demo_df.copy()
            demo_df["DisplayName"] = [f"Demo Sample {i + 1}" for i in range(len(demo_df))]
            name_col = "DisplayName"

        selected_name = st.selectbox(
            "Choose one predefined sample",
            demo_df[name_col].tolist()
        )

        selected_row = demo_df[demo_df[name_col] == selected_name].iloc[[0]].copy()
        X = selected_row[feature_columns].copy()

        st.subheader("Selected Sample")
        st.write(f"**Sample Name:** {selected_name}")

        if label_col is not None:
            expected = normalize_expected_label(selected_row[label_col].iloc[0])
            if expected is not None:
                st.write(f"**Expected Group:** {class_to_text(expected)}")

        if st.button("Run Prediction", key="predict_demo"):
            pred, prob_ra = run_prediction(artifact, X)

            predicted_class = int(pred[0])
            predicted_prob = float(prob_ra[0])

            st.success(f"Predicted Class: {class_to_text(predicted_class)}")
            st.write(f"Predicted RA Probability: {probability_to_percent(predicted_prob)}")

            if label_col is not None:
                expected = normalize_expected_label(selected_row[label_col].iloc[0])
                if expected is not None:
                    if expected == predicted_class:
                        st.success("Prediction matches the expected group.")
                    else:
                        st.warning("Prediction does not match the expected group.")

    else:
        st.subheader("Upload Requirements")
        st.write(
            "Upload a CSV file that contains the additive feature columns expected by the model. "
            "The file may contain one sample or multiple samples."
        )

        template_df = build_template(feature_columns)
        st.download_button(
            label="Download CSV Template",
            data=template_df.to_csv(index=False).encode("utf-8"),
            file_name="ra_input_template.csv",
            mime="text/csv",
        )

        uploaded_file = st.file_uploader(
            "Upload additive input CSV",
            type=["csv"]
        )

        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Failed to read the uploaded file: {e}")
                return

            missing_features = [c for c in feature_columns if c not in uploaded_df.columns]
            if missing_features:
                st.error(
                    f"The uploaded file is missing {len(missing_features)} required feature columns."
                )
                st.write("First missing columns:", missing_features[:10])
                return

            X = uploaded_df[feature_columns].copy()
            pred, prob_ra = run_prediction(artifact, X)

            sample_names = (
                uploaded_df["SampleName"].astype(str).tolist()
                if "SampleName" in uploaded_df.columns
                else [f"Uploaded Sample {i + 1}" for i in range(len(uploaded_df))]
            )

            results_df = pd.DataFrame({
                "SampleName": sample_names,
                "PredictedClass": [class_to_text(x) for x in pred],
                "Predicted_RA_Probability": [round(float(x), 4) for x in prob_ra],
            })

            true_label_col = None
            for candidate in ["ExpectedLabel", "Phenotype", "TrueLabel"]:
                if candidate in uploaded_df.columns:
                    true_label_col = candidate
                    break

            if true_label_col is not None:
                expected_vals = uploaded_df[true_label_col].apply(normalize_expected_label)
                results_df["ExpectedClass"] = expected_vals.apply(
                    lambda x: class_to_text(x) if x in [0, 1] else "Unknown"
                )
                results_df["Match"] = [
                    "Yes" if ev in [0, 1] and int(p) == int(ev) else "No"
                    for p, ev in zip(pred, expected_vals)
                ]

            st.subheader("Prediction Results")
            st.dataframe(results_df, use_container_width=True)

            st.info(
                "The uploaded raw feature values are not displayed in the interface. "
                "Only the prediction results are shown."
            )


def show_about_project_page(feature_count):
    st.title("About the Project")

    st.subheader("Project Vision")
    st.write(
        "GenomicAI aims to support the early prediction of Rheumatoid Arthritis through "
        "the integration of artificial intelligence, bioinformatics, and genomic data science. "
        "The project is designed as a step toward preventive healthcare and precision medicine."
    )

    st.subheader("Why This Matters")
    st.write(
        "Rheumatoid Arthritis can cause severe joint damage and long-term disability if it is "
        "not detected and managed early. A data-driven risk prediction approach can support "
        "faster clinical decisions and more personalized patient care."
    )

    st.subheader("Innovation")
    st.write(
        "The innovation of the project lies in combining genomic SNP analysis with machine "
        "learning to create a practical prediction tool for Rheumatoid Arthritis risk. "
        "Instead of relying only on symptoms that may appear late, the system uses genomic "
        "patterns to provide earlier predictive insight."
    )

    st.subheader("Deployment Scope")
    st.markdown(
        f"""
        - One deployed model: **XGBoost**
        - One deployed encoding: **Additive (0/1/2)**
        - Final feature count: **{feature_count}**
        - Demo mode for predefined samples
        - Upload mode for compatible additive CSV files
        """
    )

    st.subheader("Potential Impact")
    st.write(
        "The project has the potential to support clinicians, researchers, and healthcare "
        "institutions by improving early screening, reducing disease burden, and advancing "
        "AI-driven healthcare innovation in Egypt."
    )

    st.subheader("Important Note")
    st.warning(
        "This application is currently a research and demonstration prototype. "
        "It is not intended to be used as a standalone medical diagnostic system."
    )


# ==============================
# Main App
# ==============================
artifact, artifact_path, artifact_error = load_artifact()

st.sidebar.title("GenomicAI Navigation")
page = st.sidebar.radio(
    "Select a page",
    ["Home", "Prediction Center", "About the Project"]
)

st.sidebar.markdown("---")

if artifact is None:
    st.sidebar.error("Model artifact not loaded.")
else:
    st.sidebar.success("XGBoost artifact loaded successfully.")

if artifact_path is not None:
    st.sidebar.caption(f"Model path: {artifact_path}")

if page == "Home":
    if artifact is None:
        st.error(f"Failed to load model artifact: {artifact_error}")
    else:
        feature_count = len(artifact.get("feature_columns", []))
        show_home_page(feature_count)

elif page == "Prediction Center":
    if artifact is None:
        st.error(f"Failed to load model artifact: {artifact_error}")
    else:
        show_prediction_page(artifact)

elif page == "About the Project":
    feature_count = len(artifact.get("feature_columns", [])) if artifact is not None else 0
    show_about_project_page(feature_count)
