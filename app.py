import os
import re
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as pgo
import streamlit as st

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    shap = None
    SHAP_AVAILABLE = False


# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
CLASSIFICATION_THRESHOLD = 0.50
LABEL_COLUMNS = ["label", "ExpectedLabel"]
TOP_N_SAMPLE = 15
TOP_N_BATCH = 20

DISCLAIMER_TEXT = (
    "Important note: This application is a research and educational prototype. "
    "Its output is a model-based estimate using genomic data only, not a medical "
    "diagnosis. Final diagnosis must be made by a physician using clinical "
    "evaluation, laboratory tests, and imaging."
)

BASE_DIR = os.path.dirname(__file__)

st.set_page_config(
    page_title="GenomicAI — Rheumatoid Arthritis Genomic Risk",
    page_icon="🧬",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ══════════════════════════════════════════════════════════════
# BASIC HELPERS
# ══════════════════════════════════════════════════════════════
def path_in(*parts: str) -> str:
    return os.path.join(BASE_DIR, *parts)


def load_css() -> None:
    css_path = path_in("assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css()


@st.cache_resource
def load_model():
    model_path = path_in("models", "xgboost_model.joblib")
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)


def get_required_feature_columns(artifact) -> List[str]:
    if isinstance(artifact, dict) and "feature_columns" in artifact:
        return list(artifact["feature_columns"])
    if hasattr(artifact, "feature_names_in_"):
        return list(artifact.feature_names_in_)
    raise ValueError("Model artifact must contain feature_columns or feature_names_in_.")


def get_model_object(artifact):
    if isinstance(artifact, dict) and "model" in artifact:
        return artifact["model"]
    return artifact


def validate_features(df: pd.DataFrame, required_cols: List[str]) -> pd.DataFrame:
    missing = [c for c in required_cols if c not in df.columns]
    extra = [c for c in df.columns if c not in required_cols and c not in LABEL_COLUMNS]

    if missing:
        st.error(f"The uploaded file is missing {len(missing)} required genetic markers.")
        st.write(missing[:25])
        st.stop()

    if extra:
        st.warning(f"{len(extra)} extra column(s) will be ignored because they are not used by the model.")

    X = df.reindex(columns=required_cols).copy()
    X = X.apply(pd.to_numeric, errors="coerce")

    if X.isna().any().any():
        bad_cols = X.columns[X.isna().any()].tolist()[:25]
        st.error("Missing or non-numeric genotype values were found. Marker values must be 0, 1, or 2 only.")
        st.write(bad_cols)
        st.stop()

    valid_mask = X.isin([0, 1, 2])
    if not valid_mask.all().all():
        bad_cols = X.columns[~valid_mask.all(axis=0)].tolist()[:25]
        st.error("Invalid genotype values found. Allowed additive encoding values are only 0, 1, or 2.")
        st.write(bad_cols)
        st.stop()

    return X.astype(float)


def predict(artifact, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    model = get_model_object(artifact)
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= CLASSIFICATION_THRESHOLD).astype(int)
    return preds, probs


def render_disclaimer() -> None:
    st.markdown(
        f"""
        <div class="disclaimer-card presentation-card">
          <div class="dc-kicker">Important note</div>
          <div class="dc-body">{DISCLAIMER_TEXT}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════
# BIOLOGICAL MAPPING + SHAP
# ══════════════════════════════════════════════════════════════
MAPPING_CANDIDATES = [
    path_in("data", "snp_gene_protein_pathway_mapping.csv"),
    path_in("data", "biological_mapping.csv"),
    path_in("data", "snp_biology_mapping.csv"),
    path_in("assets", "snp_gene_protein_pathway_mapping.csv"),
    path_in("assets", "biological_mapping.csv"),
]


def _clean_text_value(value) -> str:
    if pd.isna(value):
        return "Not available"
    value = str(value).strip()
    return value if value else "Not available"


def _split_multi(value: str) -> List[str]:
    value = _clean_text_value(value)
    if value in ["Unmapped", "Not available"]:
        return [value]
    parts = re.split(r"[;,|]", value)
    cleaned = [p.strip() for p in parts if p.strip()]
    return cleaned or ["Not available"]


def _join_unique(values: pd.Series, fallback: str = "Not available") -> str:
    out = []
    for v in values:
        for item in _split_multi(v):
            if item not in ["", "nan", "None"] and item not in out:
                out.append(item)
    out = [x for x in out if x not in ["Not available", "Unmapped"]]
    return "; ".join(out) if out else fallback


@st.cache_data
def load_biological_mapping() -> pd.DataFrame:
    selected_path = next((p for p in MAPPING_CANDIDATES if os.path.exists(p)), None)
    if selected_path is None:
        return pd.DataFrame(columns=["SNP", "Gene", "Protein", "Pathway"])

    raw = pd.read_csv(selected_path)
    rename_map = {}
    for col in raw.columns:
        normalized = col.strip().lower().replace(" ", "_")
        if normalized in ["snp", "rsid", "variant", "marker"]:
            rename_map[col] = "SNP"
        elif normalized in ["gene", "candidate_gene", "candidate_genes", "genes"]:
            rename_map[col] = "Gene"
        elif normalized in ["protein", "proteins", "string", "string_node", "string_network"]:
            rename_map[col] = "Protein"
        elif normalized in ["pathway", "pathways", "term", "biological_pathway"]:
            rename_map[col] = "Pathway"

    df = raw.rename(columns=rename_map).copy()
    if "SNP" not in df.columns:
        return pd.DataFrame(columns=["SNP", "Gene", "Protein", "Pathway"])

    for col in ["Gene", "Protein", "Pathway"]:
        if col not in df.columns:
            df[col] = "Not available"

    df = df[["SNP", "Gene", "Protein", "Pathway"]].copy()
    df["SNP"] = df["SNP"].astype(str).str.strip()
    for col in ["Gene", "Protein", "Pathway"]:
        df[col] = df[col].apply(_clean_text_value)

    df = (
        df.groupby("SNP", as_index=False)
        .agg(
            Gene=("Gene", lambda s: _join_unique(s, fallback="Unmapped")),
            Protein=("Protein", lambda s: _join_unique(s, fallback="Not available")),
            Pathway=("Pathway", lambda s: _join_unique(s, fallback="Not available")),
        )
    )
    return df


@st.cache_resource
def load_shap_explainer(_artifact):
    if not SHAP_AVAILABLE:
        return None
    model = get_model_object(_artifact)
    return shap.TreeExplainer(model)


def compute_shap_values(artifact, X: pd.DataFrame) -> Optional[np.ndarray]:
    if not SHAP_AVAILABLE:
        return None

    explainer = load_shap_explainer(artifact)
    if explainer is None:
        return None

    values = explainer.shap_values(X)

    if isinstance(values, list):
        values = values[-1]

    values = np.asarray(values)

    if values.ndim == 3:
        # Some explainers return [samples, features, classes]. Use the positive class when present.
        values = values[:, :, -1]

    return values


def add_mapping(top_df: pd.DataFrame) -> pd.DataFrame:
    mapping = load_biological_mapping()
    if mapping.empty:
        out = top_df.copy()
        out["Gene"] = "Unmapped"
        out["Protein"] = "Not available"
        out["Pathway"] = "Not available"
        return out

    out = top_df.merge(mapping, on="SNP", how="left")
    out["Gene"] = out["Gene"].fillna("Unmapped")
    out["Protein"] = out["Protein"].fillna("Not available")
    out["Pathway"] = out["Pathway"].fillna("Not available")
    return out


def top_snps_for_sample(X_row: pd.DataFrame, shap_row: np.ndarray, top_n: int = TOP_N_SAMPLE) -> pd.DataFrame:
    row = X_row.iloc[0]
    df = pd.DataFrame(
        {
            "SNP": X_row.columns,
            "Additive Encoding": [int(row[c]) for c in X_row.columns],
            "SHAP Value": shap_row,
            "Absolute SHAP": np.abs(shap_row),
        }
    )
    df["Model Effect"] = np.where(
        df["SHAP Value"] >= 0,
        "Pushes risk score upward",
        "Pushes risk score downward",
    )
    df = df.sort_values("Absolute SHAP", ascending=False).head(top_n).reset_index(drop=True)
    return add_mapping(df)


def batch_shap_summary(X: pd.DataFrame, shap_values: np.ndarray, top_n_each: int = TOP_N_SAMPLE) -> pd.DataFrame:
    records = []
    for i, sample_id in enumerate(X.index):
        temp = pd.DataFrame(
            {
                "Sample": sample_id,
                "SNP": X.columns,
                "Additive Encoding": X.iloc[i].values.astype(int),
                "SHAP Value": shap_values[i],
                "Absolute SHAP": np.abs(shap_values[i]),
            }
        )
        temp = temp.sort_values("Absolute SHAP", ascending=False).head(top_n_each)
        records.append(temp)

    stacked = pd.concat(records, ignore_index=True)
    summary = (
        stacked.groupby("SNP", as_index=False)
        .agg(
            Top_Frequency=("Sample", "nunique"),
            Mean_Absolute_SHAP=("Absolute SHAP", "mean"),
            Mean_SHAP_Value=("SHAP Value", "mean"),
            Mean_Additive_Encoding=("Additive Encoding", "mean"),
        )
        .sort_values(["Top_Frequency", "Mean_Absolute_SHAP"], ascending=False)
        .head(TOP_N_BATCH)
        .reset_index(drop=True)
    )
    return add_mapping(summary)


def count_unique_mapped(df: pd.DataFrame, col: str, unmapped_tokens: Optional[List[str]] = None) -> int:
    if col not in df.columns:
        return 0
    unmapped_tokens = unmapped_tokens or ["Unmapped", "Not available"]
    items = []
    for value in df[col].dropna():
        for item in _split_multi(value):
            if item not in unmapped_tokens and item not in items:
                items.append(item)
    return len(items)


# ══════════════════════════════════════════════════════════════
# DYNAMIC FIGURES
# ══════════════════════════════════════════════════════════════
def make_shap_bar(df: pd.DataFrame, title: str) -> pgo.Figure:
    if "Absolute SHAP" in df.columns:
        x_col = "Absolute SHAP"
    else:
        x_col = "Mean_Absolute_SHAP"

    y = df["SNP"].iloc[::-1]
    x = df[x_col].iloc[::-1]

    if "SHAP Value" in df.columns:
        sign_values = df["SHAP Value"].iloc[::-1]
    else:
        sign_values = df["Mean_SHAP_Value"].iloc[::-1]

    colors = ["rgba(245, 158, 11, 0.90)" if v >= 0 else "rgba(20, 184, 166, 0.90)" for v in sign_values]

    fig = pgo.Figure(
        data=[
            pgo.Bar(
                x=x,
                y=y,
                orientation="h",
                marker=dict(color=colors),
                hovertemplate="SNP=%{y}<br>|SHAP|=%{x:.4f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=title,
        height=420,
        margin=dict(l=10, r=10, t=55, b=30),
        xaxis_title="Absolute SHAP contribution",
        yaxis_title="",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb", size=12),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.18)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.08)")
    return fig


def build_dynamic_sankey(df: pd.DataFrame, title: str = "Dynamic Biological Interpretation") -> pgo.Figure:
    working = df.head(10).copy()
    value_col = "Absolute SHAP" if "Absolute SHAP" in working.columns else "Mean_Absolute_SHAP"

    labels: List[str] = []
    label_to_idx: Dict[str, int] = {}
    sources: List[int] = []
    targets: List[int] = []
    values: List[float] = []

    def idx(label: str) -> int:
        label = str(label).strip() if str(label).strip() else "Not available"
        if label not in label_to_idx:
            label_to_idx[label] = len(labels)
            labels.append(label)
        return label_to_idx[label]

    terminal = "RA Risk Interpretation"
    for _, row in working.iterrows():
        snp = str(row["SNP"])
        weight = float(row.get(value_col, 1.0))
        weight = max(weight, 0.001)

        genes = _split_multi(row.get("Gene", "Unmapped"))
        proteins = _split_multi(row.get("Protein", "Not available"))
        pathways = _split_multi(row.get("Pathway", "Not available"))

        gene_weight = weight / max(len(genes), 1)
        for gene in genes:
            sources.append(idx(snp))
            targets.append(idx(gene))
            values.append(gene_weight)

            protein_weight = gene_weight / max(len(proteins), 1)
            for protein in proteins:
                sources.append(idx(gene))
                targets.append(idx(protein))
                values.append(protein_weight)

                pathway_weight = protein_weight / max(len(pathways), 1)
                for pathway in pathways:
                    sources.append(idx(protein))
                    targets.append(idx(pathway))
                    values.append(pathway_weight)

                    sources.append(idx(pathway))
                    targets.append(idx(terminal))
                    values.append(pathway_weight)

    node_colors = []
    for label in labels:
        if label.startswith("rs"):
            node_colors.append("rgba(37, 99, 235, 0.90)")
        elif label in ["Unmapped", "Not available"]:
            node_colors.append("rgba(100, 116, 139, 0.75)")
        elif label == terminal:
            node_colors.append("rgba(180, 83, 9, 0.95)")
        else:
            node_colors.append("rgba(20, 184, 166, 0.86)")

    fig = pgo.Figure(
        data=[
            pgo.Sankey(
                arrangement="snap",
                node=dict(
                    pad=18,
                    thickness=16,
                    line=dict(color="rgba(255,255,255,0.20)", width=0.5),
                    label=labels,
                    color=node_colors,
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color="rgba(148, 163, 184, 0.22)",
                ),
            )
        ]
    )
    fig.update_layout(
        title=title,
        height=430,
        margin=dict(l=5, r=5, t=50, b=5),
        font=dict(size=11, color="#e5e7eb"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ══════════════════════════════════════════════════════════════
# BIOLOGICAL INTERPRETATION UI
# ══════════════════════════════════════════════════════════════
def render_bio_styles() -> None:
    st.markdown(
        """
        <style>
        .bio-dyn-wrap {
            margin-top: 1.0rem;
            padding: 1.15rem 1.25rem;
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(124,58,237,0.09), rgba(13,148,136,0.08));
            border: 1px solid rgba(124,58,237,0.25);
            box-shadow: 0 16px 36px rgba(0,0,0,0.18);
        }
        .bio-dyn-kicker {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.58rem;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            color: #5eead4;
            font-weight: 800;
            margin-bottom: 0.35rem;
        }
        .bio-dyn-title {
            font-size: 1.35rem;
            line-height: 1.2;
            font-weight: 900;
            color: #f8fafc;
            margin-bottom: 0.45rem;
        }
        .bio-dyn-text {
            color: #aab4cf;
            line-height: 1.65;
            font-size: 0.86rem;
        }
        .bio-dyn-card {
            margin: 0.9rem 0;
            padding: 0.95rem 1.0rem;
            border-radius: 14px;
            background: rgba(15,23,42,0.72);
            border: 1px solid rgba(148,163,184,0.16);
        }
        .bio-dyn-card-title {
            font-weight: 850;
            color: #f8fafc;
            margin-bottom: 0.35rem;
            font-size: 0.98rem;
        }
        .bio-dyn-small {
            color: #94a3b8;
            line-height: 1.55;
            font-size: 0.82rem;
        }
        .bio-pill {
            display: inline-block;
            padding: 0.24rem 0.55rem;
            border-radius: 999px;
            background: rgba(45,212,191,0.12);
            border: 1px solid rgba(45,212,191,0.20);
            color: #5eead4;
            font-weight: 750;
            margin: 0.14rem 0.18rem 0.14rem 0;
            font-size: 0.76rem;
        }
        .bio-warn {
            padding: 0.8rem 0.9rem;
            border-radius: 12px;
            background: rgba(245,158,11,0.10);
            border: 1px solid rgba(245,158,11,0.25);
            color: #fcd34d;
            font-size: 0.82rem;
            line-height: 1.55;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_dynamic_sample_interpretation(
    artifact,
    X: pd.DataFrame,
    prob: float,
    sample_name: str,
    top_n: int = TOP_N_SAMPLE,
) -> None:
    render_bio_styles()

    st.markdown(
        f"""
        <div class="bio-dyn-wrap">
          <div class="bio-dyn-kicker">Dynamic Explainable Genomic AI</div>
          <div class="bio-dyn-title">Biological Interpretation for {sample_name}</div>
          <div class="bio-dyn-text">
            GenomicAI does not only predict RA genomic risk; it explains which additive SNP markers
            contributed to this specific prediction and connects them to immune-related biological pathways
            when a real mapping file is available. Current risk score: <strong>{prob:.1%}</strong>.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    shap_values = compute_shap_values(artifact, X)
    if shap_values is None:
        st.error("SHAP is not available. Add shap to requirements.txt to enable dynamic interpretation.")
        return

    top_df = top_snps_for_sample(X.iloc[[0]], shap_values[0], top_n=top_n)

    mapped_genes = count_unique_mapped(top_df, "Gene", ["Unmapped", "Not available"])
    mapped_pathways = count_unique_mapped(top_df, "Pathway", ["Unmapped", "Not available"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Top contributing SNPs", len(top_df))
    c2.metric("Mapped candidate genes", mapped_genes)
    c3.metric("Mapped immune pathways", mapped_pathways)

    show_cols = ["SNP", "Additive Encoding", "SHAP Value", "Absolute SHAP", "Model Effect", "Gene", "Protein", "Pathway"]
    display_df = top_df[show_cols].copy()
    display_df["SHAP Value"] = display_df["SHAP Value"].round(5)
    display_df["Absolute SHAP"] = display_df["Absolute SHAP"].round(5)

    st.markdown("#### Dynamic Top SNPs for this sample")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.plotly_chart(make_shap_bar(top_df, "Sample-level SHAP contribution plot"), use_container_width=True)

    if mapped_genes == 0 and mapped_pathways == 0:
        st.markdown(
            """
            <div class="bio-warn">
            No biological mapping was found for these top SNPs. The interpretation above is still dynamic
            because it is generated from this sample's SHAP values, but gene/protein/pathway links are shown
            as Unmapped / Not available until a real mapping file is added.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Dynamic SNP → Gene → Protein → Pathway flow")
    st.plotly_chart(build_dynamic_sankey(top_df, "Sample-specific dynamic Sankey"), use_container_width=True)

    pathways = []
    for p in top_df["Pathway"].dropna():
        for item in _split_multi(p):
            if item not in ["Not available", "Unmapped"] and item not in pathways:
                pathways.append(item)
    pathways = pathways[:6]

    if pathways:
        pill_html = "".join([f'<span class="bio-pill">{p}</span>' for p in pathways])
        meaning = f"The mapped top SNPs point toward the following biological pathway signals:<br><br>{pill_html}"
    else:
        meaning = (
            "The current explanation identifies the most influential SNPs for this sample. "
            "Biological pathway interpretation will become more specific when mapped SNP-gene-pathway evidence is available."
        )

    st.markdown(
        f"""
        <div class="bio-dyn-card">
          <div class="bio-dyn-card-title">Biological meaning</div>
          <div class="bio-dyn-small">{meaning}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="bio-dyn-card">
          <div class="bio-dyn-card-title">Research disclaimer</div>
          <div class="bio-dyn-small">
            This is a research and educational prototype, not a medical diagnosis.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dynamic_batch_interpretation(artifact, X: pd.DataFrame, top_n_each: int = TOP_N_SAMPLE) -> None:
    render_bio_styles()

    st.markdown(
        """
        <div class="bio-dyn-wrap">
          <div class="bio-dyn-kicker">Dynamic Batch Explainability</div>
          <div class="bio-dyn-title">Batch-level Biological Interpretation</div>
          <div class="bio-dyn-text">
            This summary is generated from SHAP values across the uploaded samples. It highlights SNPs that
            repeatedly appear among the strongest contributors across the batch, then maps them to biological
            evidence when a real mapping file is available.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    shap_values = compute_shap_values(artifact, X)
    if shap_values is None:
        st.error("SHAP is not available. Add shap to requirements.txt to enable dynamic interpretation.")
        return

    summary = batch_shap_summary(X, shap_values, top_n_each=top_n_each)
    mapped_genes = count_unique_mapped(summary, "Gene", ["Unmapped", "Not available"])
    mapped_pathways = count_unique_mapped(summary, "Pathway", ["Unmapped", "Not available"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Repeated top SNPs", len(summary))
    c2.metric("Mapped candidate genes", mapped_genes)
    c3.metric("Mapped immune pathways", mapped_pathways)

    display_df = summary.copy()
    for col in ["Mean_Absolute_SHAP", "Mean_SHAP_Value", "Mean_Additive_Encoding"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(5)

    st.markdown("#### Batch-level Top SHAP SNPs")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.plotly_chart(make_shap_bar(summary, "Batch-level mean absolute SHAP plot"), use_container_width=True)

    if mapped_genes == 0 and mapped_pathways == 0:
        st.markdown(
            """
            <div class="bio-warn">
            No biological mapping was found for the repeated top SNPs. The batch summary remains dynamic
            because it is calculated from uploaded-sample SHAP values, but mapping will show Unmapped / Not available
            until a real mapping file is added.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Dynamic batch SNP → Gene → Protein → Pathway flow")
    st.plotly_chart(build_dynamic_sankey(summary, "Batch-level dynamic Sankey"), use_container_width=True)

    st.markdown(
        """
        <div class="bio-dyn-card">
          <div class="bio-dyn-card-title">Research disclaimer</div>
          <div class="bio-dyn-small">
            This is a research and educational prototype, not a medical diagnosis.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════
# STATE + NAVIGATION
# ══════════════════════════════════════════════════════════════
if "page" not in st.session_state:
    st.session_state.page = "Home"


def go(p: str) -> None:
    st.session_state.page = p
    st.rerun()


# ══════════════════════════════════════════════════════════════
# NAVBAR
# ══════════════════════════════════════════════════════════════
st.markdown(
    """
    <div class="navbar">
      <div class="nb-brand">
        <span class="nb-dna">🧬</span>
        <span class="nb-name">GenomicAI</span>
      </div>
      <div class="nb-tagline">Research prototype · Genomic data · Risk estimation</div>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)
with c1:
    if st.button("🏠  Home", use_container_width=True, type="primary" if st.session_state.page == "Home" else "secondary"):
        go("Home")
with c2:
    if st.button("🔬  Try it", use_container_width=True, type="primary" if st.session_state.page == "Predict" else "secondary"):
        go("Predict")
with c3:
    if st.button("📄  About", use_container_width=True, type="primary" if st.session_state.page == "About" else "secondary"):
        go("About")

st.markdown('<div class="nb-line"></div>', unsafe_allow_html=True)

presentation_mode = st.checkbox(
    "Simple presentation mode",
    value=True,
    help="Recommended for presenting to engineering faculty from different fields.",
)

page = st.session_state.page


# ══════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════
if page == "Home":
    st.markdown(
        """
        <div class="hero presentation-card">
          <div class="hero-eyebrow">Interactive research prototype for academic presentation</div>
          <h1 class="hero-title">GenomicAI</h1>
          <p class="hero-sub">
            Estimating rheumatoid arthritis genomic risk using additive SNP data and explainable AI
          </p>
          <p class="hero-sub hero-sub-small">
            A research prototype that estimates genetic susceptibility patterns and explains the most influential markers.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="disclaimer-card presentation-card">
          <div class="dc-kicker">The Problem</div>
          <div class="dc-body">
            Rheumatoid arthritis can be detected after symptoms and inflammation have already progressed.
            Earlier risk awareness is challenging because clinical diagnosis depends on physician evaluation,
            laboratory tests, and imaging after biological changes may already be active.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="simple-intro presentation-card">
          <div class="simple-kicker">Our Solution</div>
          <p>
            GenomicAI uses additive genotype encoding <strong>0 / 1 / 2</strong> and an XGBoost model to estimate
            a genomic risk score. After prediction, it dynamically explains which SNP markers contributed most
            to the current result and connects them to biological evidence when mapping data is available.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sec-label presentation-label">Project workflow</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="workflow-simple presentation-card">
          <div class="simple-step"><div class="step-num">1</div><div>Real genetic data</div></div>
          <div class="simple-step"><div class="step-num">2</div><div>Data quality review</div></div>
          <div class="simple-step"><div class="step-num">3</div><div>Additive SNP encoding 0 / 1 / 2</div></div>
          <div class="simple-step"><div class="step-num">4</div><div>XGBoost risk estimation</div></div>
          <div class="simple-step"><div class="step-num">5</div><div>Dynamic SHAP explanation</div></div>
          <div class="simple-step"><div class="step-num">6</div><div>SNP-to-pathway interpretation</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sec-label presentation-label">Simple numbers for the presentation</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="simple-numbers presentation-card">
          <div class="number-card"><span>People in the dataset</span><strong>2,062</strong></div>
          <div class="number-card"><span>Rheumatoid arthritis cases</span><strong>868</strong></div>
          <div class="number-card"><span>Non-RA controls in the dataset</span><strong>1,194</strong></div>
          <div class="number-card"><span>Initial genetic markers</span><strong>531,689</strong></div>
          <div class="number-card"><span>Focused final marker set</span><strong>212 markers</strong></div>
          <div class="number-card"><span>Discrimination index reached about</span><strong>92%</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="bio-note presentation-card">
          <div class="simple-kicker">Why this is more than a prediction number</div>
          <p>
            The app is designed to explain each prediction dynamically. For every sample, SHAP identifies the
            SNP markers that most influenced the score. If biological mapping is available, those markers are
            linked to genes, proteins, and immune-related pathways.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Technical details for specialists", expanded=not presentation_mode):
        st.markdown(
            """
            - Dataset: NARAC
            - 2,062 samples: 868 cases, 1,194 controls
            - Raw SNPs: 531,689 autosomal SNPs
            - Final additive features: 212
            - Model: XGBoost
            - Encoding: additive genotype encoding 0 / 1 / 2
            - Explainability: dynamic sample-level SHAP
            - Biological interpretation: optional SNP-to-gene/protein/pathway mapping
            """
        )

    st.stop()


# ══════════════════════════════════════════════════════════════
# PREDICT
# ══════════════════════════════════════════════════════════════
elif page == "Predict":
    st.markdown(
        """
        <div class="pred-header">
          <div class="pred-icon-wrap">🔬</div>
          <div>
            <h1 class="pred-title">Risk Estimation Demo</h1>
            <p class="pred-sub">Try the app with a built-in demo sample, or upload a CSV using the required additive SNP marker columns.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_disclaimer()

    model_artifact = load_model()

    if model_artifact:
        st.markdown(
            f"""
            <div class="model-status ms-ok">
              <div class="ms-left">
                <div class="ms-dot ms-dot-ok"></div>
                <div>
                  <div class="ms-title">Prediction program loaded</div>
                  <div class="ms-sub">Uses additive SNP markers · Demo threshold {CLASSIFICATION_THRESHOLD:.2f}</div>
                </div>
              </div>
              <div class="ms-badge">READY</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="model-status ms-warn">
              <div class="ms-left">
                <div class="ms-dot ms-dot-warn"></div>
                <div>
                  <div class="ms-title">Model file not found</div>
                  <div class="ms-sub">Place xgboost_model.joblib inside the models/ folder</div>
                </div>
              </div>
              <div class="ms-badge ms-badge-warn">OFFLINE</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="mode-label presentation-label">Choose a demo mode</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📋  Built-in demo sample", "📤  Upload CSV file"])

    # ── Built-in demo sample ──────────────────────────────────
    with tab1:
        st.markdown(
            """
            <div class="tab-desc presentation-card">
              Choose a built-in sample to demonstrate the idea without uploading a file.
              This output is for educational demonstration only, not diagnosis.
            </div>
            """,
            unsafe_allow_html=True,
        )

        demo_path = path_in("data", "demo_samples.csv")
        if os.path.exists(demo_path):
            demo_df = pd.read_csv(demo_path, index_col=0)
            demo_labels = {f"Demo sample {i + 1}": idx for i, idx in enumerate(demo_df.index)}

            col_s, col_b = st.columns([3, 1])
            with col_s:
                selected_label = st.selectbox("Choose a demo sample", list(demo_labels.keys()), label_visibility="collapsed")
                selected = demo_labels[selected_label]
            with col_b:
                run = st.button("▶  Estimate risk", type="primary", use_container_width=True, key="dr")

            if run and model_artifact:
                row = demo_df.loc[[selected]]
                label_col = next((c for c in LABEL_COLUMNS if c in row.columns), None)
                exp = row[label_col].values[0] if label_col else None

                required_cols = get_required_feature_columns(model_artifact)
                X = validate_features(row, required_cols)
                preds, probs = predict(model_artifact, X)

                prob = float(probs[0])
                is_higher = preds[0] == 1
                cls = (
                    "Genetic pattern suggesting higher rheumatoid arthritis susceptibility"
                    if is_higher
                    else "Genetic pattern suggesting lower rheumatoid arthritis susceptibility"
                )

                card_cls = "result-ra" if is_higher else "result-ctrl"
                icon = "▲" if is_higher else "●"
                st.markdown(
                    f"""
                    <div class="result-card {card_cls}">
                      <div class="rc-left">
                        <div class="rc-status">{icon} {'Higher susceptibility pattern' if is_higher else 'Lower susceptibility pattern'}</div>
                        <div class="rc-class">{cls}</div>
                        <div class="rc-sample">Sample: {selected_label}</div>
                      </div>
                      <div class="rc-right">
                        <div class="rc-plabel">Program-based genomic risk score</div>
                        <div class="rc-prob {'rp-ra' if is_higher else 'rp-ctrl'}">{prob:.1%}</div>
                        <div class="rc-sublabel">Higher values mean the genetic pattern is closer to higher-risk examples learned by the program.</div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"""
                    <div class="threshold-note presentation-card">
                      This result is a risk estimate, not a diagnosis.<br>
                      Threshold used in this demo: {CLASSIFICATION_THRESHOLD:.2f}.<br>
                      The probability score is more informative than the final class because it represents the program-based risk level.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                bar_pct = int(prob * 100)
                bar_color = "#b45309" if is_higher else "#0d9488"
                st.markdown(
                    f"""
                    <div class="gauge-wrap">
                      <div class="gauge-label">
                        <span>Genomic risk score</span>
                        <span style="color:{bar_color};font-weight:700">{prob:.4f}</span>
                      </div>
                      <div class="gauge-track">
                        <div class="gauge-fill" style="width:{bar_pct}%;background:{bar_color};">
                          <div class="gauge-shine"></div>
                        </div>
                      </div>
                      <div class="gauge-marks">
                        <span>Lower</span><span>Medium</span><span>Higher</span>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Dynamic interpretation directly after risk gauge
                render_dynamic_sample_interpretation(
                    artifact=model_artifact,
                    X=X,
                    prob=prob,
                    sample_name=selected_label,
                    top_n=TOP_N_SAMPLE,
                )

                if exp is not None:
                    ok = is_higher == (str(exp).upper() in ["1", "RA", "CASE"])
                    st.markdown(
                        f"""
                        <div class="verify-row">
                          <span class="vr-badge {'vr-ok' if ok else 'vr-err'}">
                            {'Matches reference label' if ok else 'Differs from reference label'}
                          </span>
                          <span class="vr-exp">Reference label in data: <strong>{exp}</strong></span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        else:
            st.markdown(
                """
                <div class="empty-state">
                  <div class="es-icon">📂</div>
                  <div class="es-title">No demo samples found</div>
                  <div class="es-body">Add <code>demo_samples.csv</code> to the <code>data/</code> folder.<br>
                  Columns = genetic markers (0/1/2), with optional <code>label</code> column.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Upload CSV ────────────────────────────────────────────
    with tab2:
        st.markdown(
            """
            <div class="tab-desc presentation-card">
              Upload a CSV containing the required genetic marker columns. Each row is one sample.
              Marker values must be 0, 1, or 2 only. An optional <code>label</code> column can be included
              for reference-label comparison.
            </div>
            """,
            unsafe_allow_html=True,
        )

        f = st.file_uploader("Upload CSV file", type=["csv"], label_visibility="collapsed")
        if f:
            try:
                df = pd.read_csv(f, index_col=0)
                st.markdown(
                    f"""
                    <div class="file-info">
                      <span class="fi-icon">📊</span>
                      <span>Loaded <strong>{len(df)}</strong> sample(s) · Found <strong>{df.shape[1]}</strong> column(s)</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if st.button("▶  Estimate risk for all samples", type="primary", key="ur"):
                    if model_artifact:
                        label_col = next((c for c in LABEL_COLUMNS if c in df.columns), None)
                        lc = df[label_col] if label_col else None

                        required_cols = get_required_feature_columns(model_artifact)
                        X = validate_features(df, required_cols)
                        preds, probs = predict(model_artifact, X)

                        res = pd.DataFrame(
                            {
                                "Sample": X.index,
                                "Genetic Pattern": [
                                    "Genetic pattern suggesting higher rheumatoid arthritis susceptibility"
                                    if p == 1
                                    else "Genetic pattern suggesting lower rheumatoid arthritis susceptibility"
                                    for p in preds
                                ],
                                "Program-based Genomic Risk Score": [round(float(p), 4) for p in probs],
                                "Approximate Level": [
                                    "Higher susceptibility" if p >= 0.7 else "Intermediate susceptibility" if p >= 0.4 else "Lower susceptibility"
                                    for p in probs
                                ],
                            }
                        )
                        if lc is not None:
                            res["Reference Label"] = lc.values

                        def style_r(r):
                            if r["Genetic Pattern"] == "Genetic pattern suggesting higher rheumatoid arthritis susceptibility":
                                return ["background:rgba(180,83,9,.10)"] * len(r)
                            return ["background:rgba(20,184,166,.08)"] * len(r)

                        st.markdown(
                            f"""
                            <div class="threshold-note presentation-card">
                              This result is a risk estimate, not a diagnosis.<br>
                              Threshold used in this demo: {CLASSIFICATION_THRESHOLD:.2f}.<br>
                              The probability score is more informative than the final class because it represents the program-based risk level.
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        st.dataframe(
                            res.style.apply(style_r, axis=1),
                            use_container_width=True,
                            height=min(480, 60 + len(res) * 38),
                        )

                        if len(X) == 1:
                            render_dynamic_sample_interpretation(
                                artifact=model_artifact,
                                X=X,
                                prob=float(probs[0]),
                                sample_name=str(X.index[0]),
                                top_n=TOP_N_SAMPLE,
                            )
                        else:
                            render_dynamic_batch_interpretation(
                                artifact=model_artifact,
                                X=X,
                                top_n_each=TOP_N_SAMPLE,
                            )
            except Exception as e:
                st.error(f"Error reading file: {e}")


# ══════════════════════════════════════════════════════════════
# ABOUT
# ══════════════════════════════════════════════════════════════
elif page == "About":
    st.markdown(
        """
        <div class="vision-banner presentation-card">
          <div class="vb-label">About the project</div>
          <blockquote class="vb-quote">
            GenomicAI is a graduation research prototype showing how genomic data and
            a smart prediction program can estimate rheumatoid arthritis susceptibility
            patterns. The goal is academic demonstration and education, not diagnosis.
          </blockquote>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sec-label presentation-label">What makes the project clear?</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="impact-grid presentation-card">
          <div class="ig-card"><div class="ig-title">Clear engineering idea</div><div class="ig-body">Turn a very large genetic file into an understandable genomic risk score.</div></div>
          <div class="ig-card"><div class="ig-title">Focused data filtering</div><div class="ig-body">The pipeline focuses on SNP markers most relevant to the research question.</div></div>
          <div class="ig-card"><div class="ig-title">Dynamic explainability</div><div class="ig-body">The app explains each prediction using sample-level SHAP contributions.</div></div>
          <div class="ig-card"><div class="ig-title">Clear limitations</div><div class="ig-body">The model is a research prototype and requires external validation before practical use.</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Technical details for specialists", expanded=not presentation_mode):
        st.markdown(
            """
            - Dataset: NARAC
            - 2,062 samples: 868 cases, 1,194 controls
            - Raw SNPs: 531,689 autosomal SNPs
            - Final additive features: 212
            - Model: XGBoost
            - Encoding: additive genotype encoding 0 / 1 / 2
            - Explainability: dynamic SHAP
            - Biological interpretation: SNP-to-gene/protein/pathway mapping when available
            """
        )

    render_disclaimer()
    st.stop()
