from __future__ import annotations

import html
import io
import json
import os
import warnings
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb


# -----------------------------------------------------------------------------
# App configuration and paths
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ASSET_DIR = BASE_DIR / "assets"
FIGURE_DIR = ASSET_DIR / "figures"
MODEL_PATH = BASE_DIR / "models" / "xgboost_model.joblib"
NATIVE_MODEL_PATH = BASE_DIR / "models" / "xgboost_model.json"
MODEL_METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"

st.set_page_config(
    page_title="GenomicAI | RA Genomic ML Research Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def load_css() -> None:
    css_path = ASSET_DIR / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


load_css()

PLOT_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}

RED = "#b91c1c"
DARK_RED = "#7f1d1d"
CHARCOAL = "#334155"
LIGHT_RED = "#fecaca"
GRID = "#e5e7eb"


# -----------------------------------------------------------------------------
# Data and model loading
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model_artifact() -> dict:
    # Prefer XGBoost's native JSON format for cross-version portability.
    if NATIVE_MODEL_PATH.exists() and MODEL_METADATA_PATH.exists():
        metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
        booster = xgb.Booster()
        booster.load_model(str(NATIVE_MODEL_PATH))
        return {"booster": booster, **metadata}

    # Backward-compatible fallback for the original serialized artifact.
    if not MODEL_PATH.exists():
        raise FileNotFoundError("No native model or joblib model artifact was found.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        artifact = joblib.load(MODEL_PATH)
    required = {"model", "feature_columns"}
    if not isinstance(artifact, dict) or not required.issubset(artifact):
        raise ValueError("The model artifact does not contain the expected model metadata.")
    return artifact


@st.cache_data(show_spinner=False)
def read_csv(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_demo_samples() -> pd.DataFrame:
    path = DATA_DIR / "demo_samples.csv"
    return pd.read_csv(path, index_col=0)


# -----------------------------------------------------------------------------
# Formatting helpers
# -----------------------------------------------------------------------------
def page_header(kicker: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="page-head">
          <div class="page-kicker">{html.escape(kicker)}</div>
          <div class="page-title">{html.escape(title)}</div>
          <div class="page-sub">{html.escape(subtitle)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title: str) -> None:
    st.markdown(f'<div class="section-title">{html.escape(title)}</div>', unsafe_allow_html=True)


def metrics_html(items: Iterable[tuple[str, str, str, bool]]) -> str:
    cards = []
    for label, value, note, primary in items:
        cls = "metric-card primary" if primary else "metric-card"
        cards.append(
            f"""
            <div class="{cls}">
              <div class="metric-label">{html.escape(label)}</div>
              <div class="metric-value">{html.escape(value)}</div>
              <div class="metric-note">{html.escape(note)}</div>
            </div>
            """
        )
    return '<div class="metric-grid">' + "".join(cards) + "</div>"


def image_with_caption(path: Path, caption: str) -> None:
    if path.exists():
        st.markdown('<div class="figure-frame">', unsafe_allow_html=True)
        st.image(str(path), use_container_width=True)
        st.markdown(f'<div class="figure-caption">{html.escape(caption)}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


def plot_layout(fig: go.Figure, height: int = 520, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        font=dict(family="Inter, Arial, sans-serif", color="#111827", size=13),
        margin=dict(l=35, r=25, t=55, b=45),
        hoverlabel=dict(bgcolor="white", font_size=12),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


# -----------------------------------------------------------------------------
# Prediction and validation helpers
# -----------------------------------------------------------------------------
LABEL_ALIASES = {"label", "expectedlabel", "expected_label", "phenotype", "target", "class", "y"}
ID_ALIASES = {"sample", "sample_id", "sampleid", "id", "displayname", "subject", "subject_id"}


def prepare_uploaded_frame(raw: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, pd.Series | None, list[str]]:
    if raw.empty:
        raise ValueError("The uploaded file does not contain any rows.")

    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Remove common exported index column.
    unnamed = [c for c in df.columns if c.lower().startswith("unnamed:")]
    if unnamed:
        df = df.drop(columns=unnamed)

    # Detect an optional sample identifier.
    id_column = next((c for c in df.columns if c.lower().replace(" ", "_") in ID_ALIASES), None)
    if id_column is not None:
        df.index = df[id_column].astype(str)
        df = df.drop(columns=[id_column])
    else:
        df.index = [f"Sample {i + 1}" for i in range(len(df))]

    # Detect an optional expected label.
    label_column = next((c for c in df.columns if c.lower().replace(" ", "_") in LABEL_ALIASES), None)
    labels = None
    if label_column is not None:
        labels = df[label_column].copy()
        df = df.drop(columns=[label_column])

    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        preview = ", ".join(missing[:10])
        suffix = " …" if len(missing) > 10 else ""
        raise ValueError(f"Missing {len(missing)} required SNP columns: {preview}{suffix}")

    extras = [c for c in df.columns if c not in feature_columns]
    x = df.reindex(columns=feature_columns)
    x = x.apply(pd.to_numeric, errors="coerce")

    if x.isna().any().any():
        bad_cols = x.columns[x.isna().any()].tolist()
        preview = ", ".join(bad_cols[:10])
        raise ValueError(f"Non-numeric or missing genotype values were found in: {preview}")

    invalid_mask = ~x.isin([0, 1, 2])
    if invalid_mask.any().any():
        row_idx, col_idx = np.argwhere(invalid_mask.to_numpy())[0]
        snp = x.columns[col_idx]
        sample = x.index[row_idx]
        value = x.iloc[row_idx, col_idx]
        raise ValueError(
            f"Invalid additive genotype value {value!r} for {snp} in {sample}. Allowed values are 0, 1, and 2."
        )

    return x.astype(float), labels, extras


def _booster_from_artifact(artifact: dict) -> xgb.Booster:
    if "booster" in artifact:
        return artifact["booster"]
    return artifact["model"].get_booster()


def predict_probabilities(artifact: dict, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    feature_columns = list(artifact["feature_columns"])
    dmatrix = xgb.DMatrix(x, feature_names=feature_columns)
    probabilities = _booster_from_artifact(artifact).predict(dmatrix)
    threshold = float(artifact.get("demonstration_threshold", 0.50))
    predictions = (probabilities >= threshold).astype(int)
    return predictions, probabilities


def local_xgb_contributions(artifact: dict, x: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    feature_columns = list(artifact["feature_columns"])
    dmatrix = xgb.DMatrix(x, feature_names=feature_columns)
    values = _booster_from_artifact(artifact).predict(dmatrix, pred_contribs=True)
    contributions = pd.DataFrame(values[:, :-1], index=x.index, columns=feature_columns)
    base_values = values[:, -1]
    return contributions, base_values


def downloadable_template(feature_columns: list[str]) -> bytes:
    template = pd.DataFrame([[0] * len(feature_columns)], columns=feature_columns)
    template.insert(0, "sample_id", "Example_Sample")
    return template.to_csv(index=False).encode("utf-8")


def normalize_expected_labels(labels: pd.Series) -> pd.Series:
    def convert(value: object) -> float:
        text = str(value).strip().lower()
        if text in {"1", "ra", "case", "positive"}:
            return 1.0
        if text in {"0", "control", "negative"}:
            return 0.0
        try:
            numeric = float(value)
            return numeric if numeric in {0.0, 1.0} else np.nan
        except (TypeError, ValueError):
            return np.nan

    return labels.map(convert)


# -----------------------------------------------------------------------------
# Header and navigation
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
      <div class="brand">
        <div class="brand-mark">🧬</div>
        <div>
          <div class="brand-name">GenomicAI</div>
          <div class="brand-sub">Explainable SNP-based machine learning for rheumatoid arthritis research</div>
        </div>
      </div>
      <div class="header-badge">Research prototype · NARAC internal validation</div>
    </div>
    """,
    unsafe_allow_html=True,
)

NAV = ["Overview", "Prediction", "Model Validation", "Explainable AI", "Biological Interpretation", "About"]
page = st.radio("Navigation", NAV, horizontal=True, label_visibility="collapsed")


# -----------------------------------------------------------------------------
# OVERVIEW
# -----------------------------------------------------------------------------
if page == "Overview":
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">Integrated genomic machine-learning workflow</div>
          <h1>From <span>531,689 SNPs</span> to an explainable RA research model</h1>
          <p>
            GenomicAI combines structural genotype validation, GWAS Catalog-guided feature construction,
            targeted Beagle imputation, comparative machine learning, repeated nested cross-validation,
            SHAP stability analysis, interaction screening, and conservative biological enrichment.
          </p>
          <div class="hero-actions">
            <span class="hero-chip">2,062 individuals</span>
            <span class="hero-chip">8 models × 2 encodings</span>
            <span class="hero-chip">XGBoost Additive selected</span>
            <span class="hero-chip">22 stable SNPs at K = 25</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        metrics_html(
            [
                ("RA cases", "868", "42.1% of NARAC", False),
                ("Controls", "1,194", "57.9% of NARAC", False),
                ("Final features", "212", "Additive SNP dosage", False),
                ("Final model", "XGBoost", "Additive encoding", True),
            ]
        ),
        unsafe_allow_html=True,
    )

    section("End-to-end analytical pipeline")
    pipeline = read_csv("pipeline_stages.csv")
    pipe_cards = []
    for _, row in pipeline.iterrows():
        pipe_cards.append(
            f"""
            <div class="pipe">
              <div class="pipe-num">STAGE {int(row['Stage']):02d}</div>
              <div class="pipe-title">{html.escape(str(row['Title']))}</div>
              <div class="pipe-output">{html.escape(str(row['Primary_output']))}</div>
            </div>
            """
        )
    st.markdown('<div class="pipeline">' + "".join(pipe_cards) + "</div>", unsafe_allow_html=True)

    section("Final result snapshot")
    st.markdown(
        metrics_html(
            [
                ("ROC-AUC", "0.9220", "± 0.0142 across outer folds", True),
                ("PR-AUC", "0.8914", "± 0.0203 across outer folds", True),
                ("F1-score", "0.8161", "Stored outer-fold predictions", False),
                ("MCC", "0.6765", "Uses all confusion-matrix cells", False),
            ]
        ),
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    cards = [
        ("🛡️", "Validation first", "Five outer folds repeated twice produced ten held-out evaluations per model–encoding combination. Optuna tuning occurred only inside the inner loop."),
        ("🔍", "Explainability with stability", "Global SHAP rankings were checked across folds. K = 25 achieved the highest mean Jaccard value and yielded 22 SNPs selected in at least 80% of outer folds."),
        ("🧬", "Conservative biology", "The final 22 stable SNPs mapped directly to 29 GWAS Catalog genes. No STRING or network-neighbor expansion was used in the final interpretation."),
    ]
    for column, (icon, title, text) in zip([col1, col2, col3], cards):
        with column:
            st.markdown(
                f'<div class="card"><div class="icon-box">{icon}</div><div class="card-title">{title}</div><div class="card-text">{text}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="notice red"><strong>Interpretation boundary:</strong> the model was evaluated internally within NARAC. Its scores, SHAP values, interactions, genes, and pathways are research outputs—not clinical diagnosis, individual risk communication, or proof of causality.</div>',
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# PREDICTION
# -----------------------------------------------------------------------------
elif page == "Prediction":
    page_header(
        "Interactive research demo",
        "Run the frozen XGBoost Additive model",
        "Use one of the anonymized demonstration profiles or upload an additive-encoded CSV containing the complete 212-SNP feature panel.",
    )

    try:
        artifact = load_model_artifact()
        features = list(artifact["feature_columns"])
        model_ready = True
    except Exception as exc:  # pragma: no cover - visible deployment error path
        model_ready = False
        artifact = {}
        features = []
        st.error(f"The model could not be loaded: {exc}")

    if model_ready:
        st.markdown(
            metrics_html(
                [
                    ("Model", "XGBoost", "Frozen final classifier", True),
                    ("Encoding", "0 / 1 / 2", "Additive allele dosage", False),
                    ("Required SNPs", "212", "Exact feature names required", False),
                    ("Demo threshold", "0.50", "Not clinically calibrated", False),
                ]
            ),
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="notice warn"><strong>Research-use wording:</strong> the displayed probability is a model score for an RA-like genomic pattern within the training framework. It is not an individual clinical risk estimate and must not be used for diagnosis or treatment.</div>',
            unsafe_allow_html=True,
        )

        demo_tab, upload_tab = st.tabs(["Demo samples", "Upload CSV"])

        with demo_tab:
            demo = load_demo_samples()
            sample_name = st.selectbox("Select an anonymized profile", demo.index.tolist())
            expected_column = next((c for c in demo.columns if c.lower() in {"expectedlabel", "label"}), None)
            x_demo = demo.drop(columns=[expected_column]) if expected_column else demo.copy()
            x_demo = x_demo.reindex(columns=features)

            if st.button("Analyze selected profile", type="primary", use_container_width=True):
                predictions, probabilities = predict_probabilities(artifact, x_demo.loc[[sample_name]])
                contributions, base_values = local_xgb_contributions(artifact, x_demo.loc[[sample_name]])
                probability = float(probabilities[0])
                prediction = int(predictions[0])
                result_class = "higher" if prediction == 1 else "lower"
                result_text = "RA-like genomic pattern" if prediction == 1 else "Control-like genomic pattern"
                sample_safe = html.escape(str(sample_name))

                st.markdown(
                    f"""
                    <div class="result-card {result_class}">
                      <div>
                        <div class="small-label">MODEL CLASS AT 0.50 DEMONSTRATION THRESHOLD</div>
                        <div class="result-main">{result_text}</div>
                        <div class="result-sub">Anonymized demonstration profile: {sample_safe}</div>
                      </div>
                      <div>
                        <div class="small-label">MODEL OUTPUT FOR RA CLASS</div>
                        <div class="score">{probability:.1%}</div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="gauge"><div style="width:{max(1, probability * 100):.1f}%"></div></div>',
                    unsafe_allow_html=True,
                )

                if expected_column:
                    expected = int(demo.loc[sample_name, expected_column])
                    match = expected == prediction
                    note_cls = "good" if match else "warn"
                    st.markdown(
                        f'<div class="notice {note_cls}"><strong>Demo verification:</strong> expected class = {expected}; predicted class = {prediction}; match = {"yes" if match else "no"}.</div>',
                        unsafe_allow_html=True,
                    )

                section("Local SHAP contribution profile")
                row = contributions.loc[sample_name]
                top = row.reindex(row.abs().sort_values(ascending=False).head(12).index).sort_values()
                stable = read_csv("stable_snps_k25.csv")
                stable_set = set(stable["SNP"]) if not stable.empty else set()
                labels = [f"{snp}{'  ★' if snp in stable_set else ''}" for snp in top.index]
                colors = [RED if v > 0 else "#64748b" for v in top.values]
                fig = go.Figure(
                    go.Bar(
                        x=top.values,
                        y=labels,
                        orientation="h",
                        marker_color=colors,
                        customdata=x_demo.loc[sample_name, top.index].values,
                        hovertemplate="SNP: %{y}<br>SHAP contribution: %{x:.4f}<br>Genotype dosage: %{customdata}<extra></extra>",
                    )
                )
                fig.add_vline(x=0, line_color="#94a3b8", line_width=1)
                fig.update_layout(title="Top local contributions to the model output")
                fig.update_xaxes(title="SHAP contribution (positive → RA model output)")
                fig.update_yaxes(title=None)
                st.plotly_chart(plot_layout(fig, 510, legend=False), use_container_width=True, config=PLOT_CONFIG)
                st.caption("★ marks SNPs that are also included in the final K = 25 cross-fold stable set. Contributions explain this fitted model output; they do not establish allelic causality.")

        with upload_tab:
            left, right = st.columns([2, 1])
            with left:
                uploaded = st.file_uploader("Upload additive SNP matrix", type=["csv"])
            with right:
                st.download_button(
                    "Download 212-SNP template",
                    data=downloadable_template(features),
                    file_name="GenomicAI_212_SNP_input_template.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            st.markdown(
                '<div class="notice info"><strong>Accepted structure:</strong> one row per sample; one column for every required rsID; values limited to 0, 1, or 2. An optional <code>sample_id</code> column and optional label column are supported. Extra columns are ignored and reported.</div>',
                unsafe_allow_html=True,
            )

            if uploaded is not None:
                try:
                    raw = pd.read_csv(uploaded)
                    x_upload, labels, extras = prepare_uploaded_frame(raw, features)
                    st.success(f"Validated {len(x_upload)} sample(s) with all {len(features)} required SNP features.")
                    if extras:
                        st.info(f"Ignored {len(extras)} extra column(s): {', '.join(extras[:8])}{' …' if len(extras) > 8 else ''}")

                    selected_upload = st.selectbox("Sample to explain after batch prediction", x_upload.index.tolist())
                    if st.button("Run prediction", type="primary", use_container_width=True, key="upload_predict"):
                        predictions, probabilities = predict_probabilities(artifact, x_upload)
                        result = pd.DataFrame(
                            {
                                "Sample": x_upload.index,
                                "Predicted_class": np.where(predictions == 1, "RA-like", "Control-like"),
                                "RA_model_output": probabilities,
                                "Threshold": 0.50,
                            }
                        )
                        if labels is not None:
                            normalized = normalize_expected_labels(labels)
                            result["Expected_class"] = normalized.values
                            result["Match"] = np.where(normalized.notna(), normalized.values == predictions, np.nan)

                        st.dataframe(
                            result.style.format({"RA_model_output": "{:.4f}", "Threshold": "{:.2f}"}),
                            use_container_width=True,
                            hide_index=True,
                        )
                        st.download_button(
                            "Download prediction results",
                            result.to_csv(index=False).encode("utf-8"),
                            file_name="GenomicAI_prediction_results.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )

                        contrib, _ = local_xgb_contributions(artifact, x_upload.loc[[selected_upload]])
                        row = contrib.loc[selected_upload]
                        top = row.reindex(row.abs().sort_values(ascending=False).head(12).index).sort_values()
                        colors = [RED if v > 0 else "#64748b" for v in top.values]
                        fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h", marker_color=colors))
                        fig.add_vline(x=0, line_color="#94a3b8", line_width=1)
                        fig.update_layout(title=f"Local SHAP profile — {selected_upload}")
                        fig.update_xaxes(title="SHAP contribution")
                        fig.update_yaxes(title=None)
                        st.plotly_chart(plot_layout(fig, 500, legend=False), use_container_width=True, config=PLOT_CONFIG)

                except Exception as exc:
                    st.error(str(exc))


# -----------------------------------------------------------------------------
# MODEL VALIDATION
# -----------------------------------------------------------------------------
elif page == "Model Validation":
    page_header(
        "Processing and evaluation",
        "Why XGBoost Additive was selected",
        "The definitive Python benchmark compared eight supervised-learning families under additive and one-hot genotype representations using completion-safe repeated nested cross-validation.",
    )

    st.markdown(
        metrics_html(
            [
                ("Completed combinations", "16 / 16", "8 models × 2 encodings", False),
                ("Outer evaluations", "10 each", "5 folds × 2 repeats", False),
                ("Optimization target", "PR-AUC", "Mean inner-fold score", False),
                ("Selected model", "XGBoost Additive", "Selection score 0.9186", True),
            ]
        ),
        unsafe_allow_html=True,
    )

    section("Repeated nested cross-validation with Optuna")
    image_with_caption(
        FIGURE_DIR / "nested_cv_optuna.png",
        "Only the outer-training partition enters the Optuna inner loop. The held-out outer fold is evaluated once after the best inner-loop hyperparameters are selected and the model is refitted.",
    )

    section("Complete model–encoding benchmark")
    results = read_csv("model_results.csv")
    encoding_colors = {"Additive": RED, "OneHot": CHARCOAL}
    symbols = {"Additive": "circle", "OneHot": "square"}
    results = results.copy()
    results["Marker_size"] = results["Selection_score"] - results["Selection_score"].min() + 0.08
    fig = px.scatter(
        results,
        x="ROC_AUC",
        y="PR_AUC",
        color="Encoding",
        symbol="Encoding",
        color_discrete_map=encoding_colors,
        symbol_map=symbols,
        size="Marker_size",
        size_max=18,
        hover_name="Model",
        hover_data={"Rank": True, "F1": ":.4f", "MCC": ":.4f", "Selection_score": ":.4f", "Marker_size": False, "ROC_AUC": ":.4f", "PR_AUC": ":.4f"},
        labels={"ROC_AUC": "Mean ROC-AUC", "PR_AUC": "Mean PR-AUC"},
        title="ROC-AUC versus PR-AUC model map",
    )
    selected = results.loc[results["Rank"] == 1].iloc[0]
    fig.add_annotation(
        x=selected["ROC_AUC"], y=selected["PR_AUC"], text="Selected: XGBoost Additive",
        showarrow=True, arrowhead=2, ax=-115, ay=-40, arrowcolor=RED, font=dict(color=RED, size=13),
    )
    fig.update_traces(marker=dict(line=dict(width=1, color="white")), selector=dict(mode="markers"))
    fig.update_xaxes(range=[0.795, 0.929])
    fig.update_yaxes(range=[0.675, 0.900])
    st.plotly_chart(plot_layout(fig, 590), use_container_width=True, config=PLOT_CONFIG)

    ranking_view = results.rename(
        columns={
            "ROC_AUC": "ROC-AUC",
            "PR_AUC": "PR-AUC",
            "Selection_score": "Selection score",
        }
    )
    st.dataframe(
        ranking_view.style.format({"ROC-AUC": "{:.4f}", "PR-AUC": "{:.4f}", "F1": "{:.4f}", "MCC": "{:.4f}", "Selection score": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )

    section("How the two primary AUC metrics complement each other")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            '<div class="card"><div class="icon-box">↗</div><div class="card-title">ROC-AUC — overall class discrimination</div><div class="card-text">Summarizes how well the model ranks RA cases above controls across all thresholds using sensitivity versus false-positive rate. It provides the broad separation view.</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="card"><div class="icon-box">◎</div><div class="card-title">PR-AUC — positive-class performance</div><div class="card-text">Summarizes the precision–recall trade-off for RA cases. Because RA was the smaller class and positive predictions were central to the task, PR-AUC was used as the Optuna objective.</div></div>',
            unsafe_allow_html=True,
        )

    section("Final model card")
    model_card = pd.DataFrame(
        {
            "Element": [
                "Prediction task", "Input", "Estimator", "Evaluation", "Primary metrics",
                "Operating-point metrics", "Generalization gaps", "Validated use", "Excluded use",
            ],
            "Final specification": [
                "Binary RA case–control classification within NARAC",
                "212 additive SNP dosage features coded 0/1/2",
                "XGBoost classifier optimized with Optuna",
                "Five outer folds repeated twice; tuning inside inner CV",
                "ROC-AUC 0.9220 ± 0.0142; PR-AUC 0.8914 ± 0.0203",
                "Sensitivity 0.8451; specificity 0.8358; precision 0.7903; F1 0.8161; MCC 0.6765",
                "ROC 0.0535; PR 0.0743",
                "Internal research comparison and hypothesis prioritization",
                "Clinical diagnosis, treatment selection, causal variant identification",
            ],
        }
    )
    st.dataframe(model_card, use_container_width=True, hide_index=True)
    st.markdown(
        '<div class="notice warn"><strong>Important methodological boundary:</strong> outer-test samples were isolated from model fitting and Optuna tuning, but the final 212-feature matrix was constructed cohort-wide before cross-validation. The reported metrics therefore evaluate modeling conditional on the fixed preprocessed matrix, not a fully nested raw-data-to-prediction pipeline.</div>',
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# EXPLAINABLE AI
# -----------------------------------------------------------------------------
elif page == "Explainable AI":
    page_header(
        "Post-processing and explainability",
        "From feature attribution to cross-fold stability",
        "SHAP quantified model contributions, while the stability layer restricted interpretation to SNPs that remained influential across the ten outer evaluations.",
    )

    st.markdown(
        metrics_html(
            [
                ("Top global SNP", "rs660895", "Mean |SHAP| = 0.7430", True),
                ("Selected K", "25", "Highest mean Jaccard", False),
                ("Stable SNPs", "22", "Frequency ≥ 0.80", False),
                ("Chromosome 6 SNPs", "8", "Stable HLA/MHC-region cluster", False),
            ]
        ),
        unsafe_allow_html=True,
    )

    section("What SHAP represents in this project")
    image_with_caption(
        FIGURE_DIR / "shap_explainer.png",
        "Positive SHAP values push the fitted model output toward the RA class and negative values push it toward the control class. The values explain model behavior and are not odds ratios or causal effects.",
    )

    section("Global SHAP importance")
    shap_top = read_csv("global_shap_top20.csv").sort_values("Mean_abs_SHAP")
    fig = go.Figure(
        go.Bar(
            x=shap_top["Mean_abs_SHAP"],
            y=shap_top["SNP"],
            orientation="h",
            marker_color=[RED if snp == "rs660895" else "#ef4444" for snp in shap_top["SNP"]],
            customdata=np.stack([shap_top["Frequency"]], axis=-1),
            hovertemplate="SNP: %{y}<br>Mean |SHAP|: %{x:.4f}<br>K25 frequency: %{customdata[0]:.1f}<extra></extra>",
        )
    )
    fig.update_layout(title="Top 20 SNPs by global mean absolute SHAP")
    fig.update_xaxes(title="Mean absolute SHAP value")
    fig.update_yaxes(title=None)
    st.plotly_chart(plot_layout(fig, 630, legend=False), use_container_width=True, config=PLOT_CONFIG)

    with st.expander("Open the SHAP beeswarm distribution"):
        image_with_caption(
            FIGURE_DIR / "shap_beeswarm_top20.png",
            "Each point represents a sample-level contribution. Color indicates genotype feature value, while horizontal position indicates direction and magnitude of the contribution to the model output.",
        )

    section("Stability behavior across candidate K values")
    stability = read_csv("stability_summary.csv")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=stability["K"], y=stability["Mean_Jaccard"], mode="lines+markers",
            name="Mean Jaccard", line=dict(color=RED, width=3), marker=dict(size=9),
            error_y=dict(type="data", array=stability["SD_Jaccard"], visible=True, color="#fca5a5"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=stability["K"], y=stability["Stable_SNP_Count"], mode="lines+markers",
            name="Stable SNP count", line=dict(color=CHARCOAL, width=2, dash="dot"), marker=dict(size=8), yaxis="y2",
        )
    )
    fig.add_vline(x=25, line_color=RED, line_dash="dash")
    fig.add_annotation(x=25, y=0.7396, text="Selected K = 25", showarrow=True, ax=65, ay=-45, arrowcolor=RED, font=dict(color=RED))
    fig.update_layout(
        title="Cross-fold reproducibility and retained stable-feature count",
        yaxis=dict(title="Mean pairwise Jaccard", range=[0.4, 0.81], gridcolor=GRID),
        yaxis2=dict(title="Stable SNP count", overlaying="y", side="right", range=[0, 42], showgrid=False),
    )
    fig.update_xaxes(title="Top-K features per outer fold")
    st.plotly_chart(plot_layout(fig, 540), use_container_width=True, config=PLOT_CONFIG)

    section("Final K = 25 stable SNP set")
    stable = read_csv("stable_snps_k25.csv")
    fig = go.Figure(
        go.Bar(
            x=stable["Frequency"],
            y=stable["SNP"],
            orientation="h",
            marker_color=np.where(stable["Frequency"] == 1.0, RED, "#64748b"),
            customdata=np.stack([stable["Cross_fold_mean_abs_SHAP"], stable["Chromosome"], stable["Folds_selected"]], axis=-1),
            hovertemplate="SNP: %{y}<br>Selection frequency: %{x:.1f}<br>Folds: %{customdata[2]:.0f}/10<br>Cross-fold mean |SHAP|: %{customdata[0]:.4f}<br>Chromosome: %{customdata[1]:.0f}<extra></extra>",
        )
    )
    fig.add_vline(x=0.80, line_color="#f59e0b", line_dash="dash")
    fig.update_layout(title="Stable SNP selection frequency at K = 25")
    fig.update_xaxes(title="Fraction of outer folds", range=[0.75, 1.02])
    fig.update_yaxes(title=None, autorange="reversed")
    st.plotly_chart(plot_layout(fig, 700, legend=False), use_container_width=True, config=PLOT_CONFIG)

    section("Genomic distribution of stable SNPs")
    chromosome_order = sorted(stable["Chromosome"].unique())
    fig = px.scatter(
        stable,
        x="Base_pair_position",
        y="Cross_fold_mean_abs_SHAP",
        color=stable["Chromosome"].astype(str),
        symbol=stable["Chromosome"].astype(str),
        hover_name="SNP",
        hover_data={"Frequency": ":.1f", "Chromosome": True, "Base_pair_position": True},
        facet_col="Chromosome",
        facet_col_wrap=6,
        labels={"Cross_fold_mean_abs_SHAP": "Cross-fold mean |SHAP|", "Base_pair_position": "Genomic coordinate"},
        title="K = 25 stable-SNP genomic importance map",
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_layout(showlegend=False)
    st.plotly_chart(plot_layout(fig, 690, legend=False), use_container_width=True, config=PLOT_CONFIG)

    section("Focused SNP–SNP interaction screen")
    interactions = read_csv("interactions_top15.csv").sort_values("mean_absolute_interaction")
    interactions["Pair"] = interactions["SNP_A"] + " × " + interactions["SNP_B"]
    fig = go.Figure(
        go.Bar(
            x=interactions["mean_absolute_interaction"], y=interactions["Pair"], orientation="h",
            marker_color=[RED if rank == 1 else "#475569" for rank in interactions["rank"]],
            hovertemplate="%{y}<br>Mean absolute SHAP interaction: %{x:.5f}<extra></extra>",
        )
    )
    fig.update_layout(title="Top 15 model-based interactions among the 22 stable SNPs")
    fig.update_xaxes(title="Mean absolute SHAP interaction")
    fig.update_yaxes(title=None)
    st.plotly_chart(plot_layout(fig, 620, legend=False), use_container_width=True, config=PLOT_CONFIG)
    st.markdown(
        '<div class="notice red"><strong>Interaction boundary:</strong> rs660895 × rs13192471 was the strongest pair, but SHAP interaction magnitude describes a pattern learned by the fitted trees. It is not a formal statistical epistasis test and does not prove a biological interaction.</div>',
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# BIOLOGICAL INTERPRETATION
# -----------------------------------------------------------------------------
elif page == "Biological Interpretation":
    page_header(
        "Direct mapping and enrichment",
        "Biological context of the final stable SNP signature",
        "The K = 25 stable set was mapped directly through the rheumatoid-arthritis GWAS Catalog file, then evaluated with GO Biological Process, KEGG, Reactome, Jensen DISEASES, and DisGeNET libraries without network expansion.",
    )

    st.markdown(
        metrics_html(
            [
                ("Stable SNPs", "22", "All matched to GWAS records", False),
                ("SNP–gene pairs", "37", "After splitting mapped annotations", False),
                ("Direct mapped genes", "29", "Unique genes", True),
                ("HLA / non-HLA", "5 / 24", "Direct mapping groups", False),
            ]
        ),
        unsafe_allow_html=True,
    )

    section("Analysis summary")
    image_with_caption(
        FIGURE_DIR / "enrichment_summary.png",
        "The final result was HLA/MHC-centered. HLA genes drove coherent immune-pathway enrichment, while the non-HLA set contributed broader disease-annotation context.",
    )

    section("Significant term counts by gene group")
    counts = read_csv("enrichment_counts.csv")
    long = counts.melt(id_vars="Gene_group", var_name="Database_category", value_name="Significant_terms")
    fig = px.bar(
        long,
        x="Database_category",
        y="Significant_terms",
        color="Gene_group",
        barmode="group",
        color_discrete_map={
            "HLA Genes": RED,
            "Non-HLA Genes": "#94a3b8",
            "All Direct GWAS-Mapped Genes": "#334155",
        },
        labels={"Database_category": "Database category", "Significant_terms": "Adjusted-significant terms"},
        title="Enrichment breadth after direct GWAS-mapped-gene analysis",
    )
    st.plotly_chart(plot_layout(fig, 530), use_container_width=True, config=PLOT_CONFIG)
    st.caption("Counts use adjusted p-value < 0.05. Disease combines the Jensen DISEASES and DisGeNET output categories used in the project summary.")

    section("Direct mapped gene groups")
    genes = read_csv("gene_groups.csv")
    c1, c2 = st.columns([1, 2])
    with c1:
        hla = genes.loc[genes["Group"] == "HLA", "Gene"].tolist()
        hla_html = "".join(f'<span class="gene-chip hla">{html.escape(g)}</span>' for g in hla)
        st.markdown(f'<div class="card"><div class="card-title">HLA genes · 5</div><div class="card-text">{hla_html}</div></div>', unsafe_allow_html=True)
    with c2:
        non_hla = genes.loc[genes["Group"] == "Non-HLA", "Gene"].tolist()
        non_hla_html = "".join(f'<span class="gene-chip">{html.escape(g)}</span>' for g in non_hla)
        st.markdown(f'<div class="card"><div class="card-title">Non-HLA genes · 24</div><div class="card-text">{non_hla_html}</div></div>', unsafe_allow_html=True)

    section("Representative significant terms")
    terms = read_csv("enrichment_terms.csv")
    display_terms = terms.rename(
        columns={
            "Gene_group": "Gene group",
            "Representative_term": "Representative term",
            "Adjusted_p_value": "Adjusted p-value",
            "Contributing_genes": "Contributing genes",
        }
    )
    st.dataframe(
        display_terms.style.format({"Adjusted p-value": "{:.2e}"}),
        use_container_width=True,
        hide_index=True,
    )

    section("Final interpretation")
    st.markdown(
        """
        <div class="interpretation">
          <h3>HLA/MHC-centered immune signal</h3>
          <ul>
            <li>The five HLA genes dominated the coherent pathway-level enrichment.</li>
            <li>Leading biological themes converged on MHC class II antigen processing and presentation, interferon-gamma signaling, T-cell biology, PD-1 signaling, and the KEGG rheumatoid arthritis pathway.</li>
            <li>The 24 non-HLA genes did not produce adjusted-significant GO_BP, KEGG, or Reactome terms as a separate group, but they contributed disease-association context in the direct annotation layer.</li>
            <li>Direct mapping preserved traceability from stable SNP to gene and enriched term, but it may miss distal regulatory targets, eQTL links, and chromatin-mediated effects.</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="notice red"><strong>Biological boundary:</strong> mapped genes are candidate annotation links, and enrichment describes over-represented database annotations. It does not establish that the SNPs regulate those genes or that the pathways are causal in an individual.</div>',
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# ABOUT
# -----------------------------------------------------------------------------
elif page == "About":
    page_header(
        "Project identity and boundaries",
        "Integrated AI Pipeline for Early Prediction of RA and Biological Pathway Interpretation",
        "A Biomedical Engineering graduation project from the Faculty of Engineering, Minia University, developed as a reproducible research workflow and interactive demonstration—not a clinical device.",
    )

    section("Project team")
    c1, c2, c3 = st.columns(3)
    students = [
        ("GA", "Giovanni Ayman Fahim Takla", "Project student"),
        ("AA", "Abdulaziz Mohamed Abdulaziz", "Project student"),
        ("OA", "Omnya Ahmed Fathy Hassan", "Project student"),
    ]
    for col, (initials, name, role) in zip([c1, c2, c3], students):
        with col:
            st.markdown(
                f'<div class="card team-card"><div class="avatar">{initials}</div><div class="team-name">{html.escape(name)}</div><div class="team-role">{role}</div></div>',
                unsafe_allow_html=True,
            )

    section("Project advisors")
    advisors = [
        "Dr. Mohamed Nagy Saad Mohamed",
        "Dr. Wael Abouelwafa Ahmed Ali",
        "Dr. Nesreen Abdelwahab Mohamed Massoud",
    ]
    a1, a2, a3 = st.columns(3)
    for col, advisor in zip([a1, a2, a3], advisors):
        with col:
            st.markdown(
                f'<div class="card"><div class="icon-box">🎓</div><div class="card-title">{html.escape(advisor)}</div><div class="card-text">Project advisor</div></div>',
                unsafe_allow_html=True,
            )

    section("Scope, strengths, and limitations")
    left, right = st.columns(2)
    with left:
        st.markdown(
            """
            <div class="card">
              <div class="card-title">Implemented strengths</div>
              <div class="card-text">
                ✓ Auditable raw-data-to-results workflow<br><br>
                ✓ GWAS Catalog-guided RA feature construction<br><br>
                ✓ Missing-only Beagle imputation overlay<br><br>
                ✓ Eight-model, two-encoding benchmark<br><br>
                ✓ Repeated nested cross-validation with Optuna<br><br>
                ✓ Global and local model explainability<br><br>
                ✓ Cross-fold stable-SNP interpretation layer
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            """
            <div class="card">
              <div class="card-title">Current limitations</div>
              <div class="card-text">
                • Internal validation within one NARAC cohort<br><br>
                • No independent external or multi-ancestry validation<br><br>
                • Cohort-derived preprocessing was not fully nested<br><br>
                • Project-specific genotype-frequency heuristic needs sensitivity analysis<br><br>
                • SNP-only phenotype without clinical or serological variables<br><br>
                • No clinical calibration or validated decision threshold<br><br>
                • SHAP and enrichment do not prove causality
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    section("Technology stack")
    stack = pd.DataFrame(
        {
            "Layer": ["Genotype handling", "Imputation", "Transformation", "Optimization", "Modeling", "Explainability", "Interface"],
            "Tools": ["PLINK v1.9 · PED/MAP · VCF", "Beagle v5.5", "R · Python · pandas · NumPy", "Optuna", "scikit-learn · XGBoost · MLP", "SHAP / XGBoost contribution values", "Streamlit · Plotly"],
            "Role": ["Format conversion and structural checks", "Targeted missing-call completion", "QC summaries and genotype encoding", "Inner-loop hyperparameter search", "Comparative supervised-learning benchmark", "Global, local, stability, and interaction interpretation", "Research dashboard and prediction demo"],
        }
    )
    st.dataframe(stack, use_container_width=True, hide_index=True)

    section("Responsible-use statement")
    st.markdown(
        """
        <div class="interpretation">
          <h3>Research prototype only</h3>
          <p style="line-height:1.7;margin:0;">
            GenomicAI does not diagnose rheumatoid arthritis, determine treatment, or communicate validated personal genetic risk.
            Clinical deployment would require a frozen pipeline, independent external validation, population-specific calibration,
            privacy and governance controls, professional interpretation, regulatory review, and prospective clinical evaluation.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    '<div class="footer">GenomicAI · Biomedical Engineering Programme · Faculty of Engineering, Minia University · Research dashboard aligned to the final graduation book</div>',
    unsafe_allow_html=True,
)
