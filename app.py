import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

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


def normalize_expected_label(value):
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in {"1", "ra", "case", "patient"}:
        return 1
    if text in {"0", "control", "healthy", "normal"}:
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
    template = pd.DataFrame(columns=["SampleName"] + feature_columns)
    return template


def show_home_page(feature_count):
    st.title("Rheumatoid Arthritis Genomic Prediction App")

    st.write(
        "This application presents the final deployment view of the rheumatoid arthritis "
        "genomic pipeline. The goal of the project is to predict RA status from genotype data "
        "after a multi-step preprocessing and refinement workflow."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Individuals", "2,062")
    col2.metric("Original Autosomal SNPs", "531,689")
    col3.metric("Refined RA SNP Panel", "313")
    col4.metric("Final Additive Features", str(feature_count))

    st.subheader("Problem Statement")
    st.write(
        "Raw genomic data are extremely high-dimensional and not directly suitable for robust "
        "machine learning. The project addresses this by transforming large raw NARAC genotype files "
        "into a smaller, biologically informed, technically clean feature set for RA prediction."
    )

    st.subheader("How the Project Solves the Problem")
    st.markdown(
        """
        1. **Raw genotype ingestion** from chromosome-wise PED/MAP files.  
        2. **Discrepancy detection and validation** to check structure, IDs, phenotype coding, sex coding, and genotype integrity.  
        3. **RA SNP intersection and refinement** to keep SNPs that are biologically relevant and technically reliable.  
        4. **Imputation** to resolve missing genotypes while preserving original observed calls.  
        5. **Additive encoding (0/1/2)** to convert retained SNPs into ML-ready numerical features.  
        6. **Modeling** using machine learning, where this deployed app uses one final XGBoost model only.
        """
    )

    st.subheader("Deployment Scope")
    st.markdown(
        """
        - One deployed model only: **XGBoost**
        - One deployed encoding only: **Additive encoding (0/1/2)**
        - No confidential dataset rows are displayed directly
        - Two testing modes:
          - predefined demo samples
          - user-uploaded additive feature CSV
        """
    )

    st.subheader("Why Additive Encoding?")
    st.write(
        "The transformation stage retained 212 SNPs after filtering and encoded them as 0/1/2 "
        "to represent major-homozygous, heterozygous, and minor-homozygous states in an "
        "ML-ready additive format."
    )

    fig, ax = plt.subplots(figsize=(7, 3))
    labels = ["Raw SNPs", "RA Panel", "Final Additive"]
    values = [531689, 313, feature_count]
    ax.bar(labels, values)
    ax.set_title("Feature Reduction Through the Pipeline")
    ax.set_ylabel("Count")
    st.pyplot(fig)


def show_prediction_page(artifact):
    st.title("Prediction Center")
    st.write(
        "Choose one of the two prediction modes below. "
        "The first mode uses predefined demo samples. "
        "The second mode accepts a user-uploaded additive CSV file."
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

    # -------------------------------------------------
    # Mode 1: Predefined demo samples
    # -------------------------------------------------
    if input_mode == "Predefined Demo Samples":
        demo_df, demo_path, demo_error = load_demo_samples()

        if demo_df is None:
            st.warning(
                "No demo_samples.csv file was found. "
                "To use this mode, add data/demo_samples.csv to the repository."
            )
            st.info(
                "Expected columns: DisplayName, ExpectedLabel, and all additive feature columns "
                "required by the model."
            )
            return

        required_missing = [c for c in feature_columns if c not in demo_df.columns]
        if required_missing:
            st.error(
                f"The demo_samples.csv file is missing {len(required_missing)} required feature columns."
            )
            return

        name_col = "DisplayName" if "DisplayName" in demo_df.columns else None
        label_col = "ExpectedLabel" if "ExpectedLabel" in demo_df.columns else None

        if name_col is None:
            demo_df = demo_df.copy()
            demo_df["DisplayName"] = [f"Demo Sample {i+1}" for i in range(len(demo_df))]
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

    # -------------------------------------------------
    # Mode 2: Upload CSV
    # -------------------------------------------------
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
                else [f"Uploaded Sample {i+1}" for i in range(len(uploaded_df))]
            )

            results_df = pd.DataFrame({
                "SampleName": sample_names,
                "PredictedClass": [class_to_text(x) for x in pred],
                "Predicted_RA_Probability": [round(float(x), 4) for x in prob_ra],
            })

            # Optional true label comparison
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


def show_pipeline_page():
    st.title("Pipeline and Methodology")

    st.subheader("1. Data Description")
    st.write(
        "The pipeline begins with chromosome-wise NARAC PED/MAP genotype files. "
        "The dataset contains 2,062 individuals and approximately 531,689 autosomal SNPs."
    )

    st.subheader("2. Discrepancy Detection")
    st.write(
        "Structural consistency, duplicate IDs, phenotype coding, sex coding, genotype token integrity, "
        "and missingness patterns are checked before any downstream modeling."
    )

    st.subheader("3. RA SNP Intersection and Refinement")
    st.write(
        "The project intersects NARAC SNPs with externally reported RA-associated SNPs to build "
        "a biologically informed RA-focused panel. After refinement, the final panel contains 313 SNPs."
    )

    st.subheader("4. Genotype Imputation")
    st.write(
        "Missing genotypes are imputed using Beagle. A missing-only overlay is then applied so that "
        "original non-missing genotype calls remain unchanged while only missing positions are replaced."
    )

    st.subheader("5. Transformation and Additive Encoding")
    st.write(
        "The imputed genotype matrix is transformed into machine-learning-ready numerical features. "
        "After filtering, 212 SNPs are retained and encoded in additive form as 0/1/2."
    )

    st.subheader("6. Modeling")
    st.write(
        "The original notebook benchmarks multiple models using nested cross-validation and explainability analysis. "
        "For deployment, this application uses one final XGBoost model trained on the additive encoding only."
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    steps = [
        "Raw PED/MAP",
        "Validation",
        "RA SNP Intersection",
        "Imputation",
        "Additive Encoding",
        "XGBoost Deployment"
    ]
    y = [1, 1, 1, 1, 1, 1]
    ax.plot(range(len(steps)), y, marker="o")
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels(steps, rotation=25, ha="right")
    ax.set_yticks([])
    ax.set_title("Project Flow")
    st.pyplot(fig)


# ==============================
# Main App
# ==============================
artifact, artifact_path, artifact_error = load_artifact()

st.sidebar.title("GenomicAI Navigation")
page = st.sidebar.radio(
    "Select a page",
    ["Home", "Prediction Center", "Pipeline and Methodology"]
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

elif page == "Pipeline and Methodology":
    show_pipeline_page()
