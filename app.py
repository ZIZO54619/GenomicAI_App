import streamlit as st
import pandas as pd
import numpy as np
import joblib, os
import plotly.graph_objects as pgo
import shap

CLASSIFICATION_THRESHOLD = 0.50
DISCLAIMER_TEXT = (
    "Important note: This application is a research and educational prototype. "
    "Its output is a model-based estimate using genomic data only, not a medical "
    "diagnosis. Final diagnosis must be made by a physician using clinical "
    "evaluation, laboratory tests, and imaging."
)
LABEL_COLUMNS = ["label", "ExpectedLabel"]

st.set_page_config(
    page_title="GenomicAI — Rheumatoid Arthritis Genomic Risk",
    page_icon="🧬",
    layout="centered",          # CENTERED — fixes the extremes issue
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────
def load_css():
    p = os.path.join(os.path.dirname(__file__), "assets", "style.css")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css()

# ── Model ─────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    p = os.path.join(os.path.dirname(__file__), "models", "xgboost_model.joblib")
    return joblib.load(p) if os.path.exists(p) else None

def get_required_feature_columns(artifact):
    return list(artifact["feature_columns"])

def validate_features(df, required_cols):
    missing = [c for c in required_cols if c not in df.columns]
    extra = [c for c in df.columns if c not in required_cols and c not in LABEL_COLUMNS]

    if missing:
        st.error(f"The uploaded file is missing {len(missing)} required genetic markers.")
        st.write(missing[:20])
        st.stop()

    if extra:
        st.warning(f"{len(extra)} extra column(s) will be ignored because they are not used by the model.")

    X = df.reindex(columns=required_cols)
    values = pd.unique(X.values.ravel())
    invalid = [v for v in values if v not in [0, 1, 2]]

    if invalid:
        st.error(f"Invalid values found in the uploaded file: {invalid}. Allowed values are only 0, 1, or 2.")
        st.stop()

    return X

def predict(artifact, df):
    m, cols = artifact["model"], get_required_feature_columns(artifact)
    probs = m.predict_proba(df.reindex(columns=cols))[:, 1]
    return (probs >= CLASSIFICATION_THRESHOLD).astype(int), probs

def render_disclaimer():
    st.markdown(f"""
    <div class="disclaimer-card presentation-card">
      <div class="dc-kicker">Important note</div>
      <div class="dc-body">{DISCLAIMER_TEXT}</div>
    </div>
    """, unsafe_allow_html=True)


# ── Dynamic Biological Interpretation ─────────────────────────
def _safe_text(value, fallback="Not available"):
    if value is None:
        return fallback
    try:
        if pd.isna(value):
            return fallback
    except Exception:
        pass
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "na", "n/a", "-"}:
        return fallback
    return text


def _split_multi_value(value, fallback="Not available"):
    text = _safe_text(value, fallback=fallback)
    if text == fallback:
        return [fallback]
    for sep in [";", "|", ","]:
        if sep in text:
            return [x.strip() for x in text.split(sep) if x.strip()] or [fallback]
    return [text]


@st.cache_data
def load_biological_mapping():
    """
    Optional mapping file.
    Expected useful columns: SNP, Gene, Protein, Pathway.
    If no file exists, dynamic SHAP still works and biology fields become Unmapped / Not available.
    """
    base = os.path.dirname(__file__)
    candidates = [
        os.path.join(base, "data", "snp_gene_protein_pathway_mapping.csv"),
        os.path.join(base, "data", "biological_mapping.csv"),
        os.path.join(base, "data", "snp_biology_mapping.csv"),
        os.path.join(base, "assets", "snp_gene_protein_pathway_mapping.csv"),
        os.path.join(base, "assets", "biological_mapping.csv"),
    ]

    for path in candidates:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df.columns = [str(c).strip() for c in df.columns]

            lower_to_original = {c.lower(): c for c in df.columns}
            rename = {}
            aliases = {
                "SNP": ["snp", "rsid", "rs_id", "marker", "variant"],
                "Gene": ["gene", "candidate_gene", "candidate_genes", "mapped_gene", "mapped_genes", "genes"],
                "Protein": ["protein", "proteins", "string_protein", "string_node", "network_node"],
                "Pathway": ["pathway", "pathways", "term", "enrichment_term", "biological_pathway"],
            }

            for standard, names in aliases.items():
                for name in names:
                    if name in lower_to_original:
                        rename[lower_to_original[name]] = standard
                        break

            df = df.rename(columns=rename)
            if "SNP" not in df.columns:
                return pd.DataFrame(columns=["SNP", "Gene", "Protein", "Pathway"])

            for col in ["Gene", "Protein", "Pathway"]:
                if col not in df.columns:
                    df[col] = "Not available"

            df["SNP"] = df["SNP"].astype(str).str.strip()
            return df[["SNP", "Gene", "Protein", "Pathway"]].copy()

    return pd.DataFrame(columns=["SNP", "Gene", "Protein", "Pathway"])


@st.cache_resource
def get_shap_explainer(model):
    return shap.TreeExplainer(model)


def _extract_shap_matrix(shap_values):
    """Return a 2D samples × features SHAP matrix for binary or single-output models."""
    if isinstance(shap_values, list):
        # Binary classifiers commonly return [class_0, class_1]
        arr = np.asarray(shap_values[-1])
    else:
        arr = np.asarray(shap_values)

    if arr.ndim == 3:
        # Possible shape: samples × features × classes
        if arr.shape[-1] > 1:
            arr = arr[:, :, -1]
        else:
            arr = arr[:, :, 0]
    return arr


def compute_shap_values(artifact, X):
    model = artifact["model"]
    explainer = get_shap_explainer(model)
    raw = explainer.shap_values(X)
    shap_matrix = _extract_shap_matrix(raw)
    return pd.DataFrame(shap_matrix, index=X.index, columns=X.columns)


def attach_biology_mapping(top_df, mapping_df):
    top_df = top_df.copy()

    if mapping_df is None or mapping_df.empty:
        top_df["Gene"] = "Unmapped"
        top_df["Protein"] = "Not available"
        top_df["Pathway"] = "Not available"
        return top_df

    merged = top_df.merge(mapping_df, on="SNP", how="left")
    merged["Gene"] = merged["Gene"].apply(lambda x: _safe_text(x, "Unmapped"))
    merged["Protein"] = merged["Protein"].apply(lambda x: _safe_text(x, "Not available"))
    merged["Pathway"] = merged["Pathway"].apply(lambda x: _safe_text(x, "Not available"))
    return merged


def get_top_shap_for_sample(shap_df, X, sample_index, top_n=15):
    row_shap = shap_df.loc[sample_index]
    row_x = X.loc[sample_index]

    top = (
        pd.DataFrame({
            "SNP": row_shap.index,
            "Additive Encoding": row_x.values,
            "SHAP Value": row_shap.values,
            "Absolute SHAP": np.abs(row_shap.values),
        })
        .sort_values("Absolute SHAP", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    top["Contribution Direction"] = np.where(
        top["SHAP Value"] > 0,
        "Pushes risk score higher",
        np.where(top["SHAP Value"] < 0, "Pushes risk score lower", "Neutral")
    )
    top["Additive Encoding"] = top["Additive Encoding"].astype(int).astype(str)
    top["SHAP Value"] = top["SHAP Value"].astype(float).round(5)
    top["Absolute SHAP"] = top["Absolute SHAP"].astype(float).round(5)
    return top


def summarize_batch_shap(shap_df, X, mapping_df, top_n=15):
    rows = []
    for sample_id in shap_df.index:
        top = get_top_shap_for_sample(shap_df, X, sample_id, top_n=top_n)
        top["Sample"] = sample_id
        rows.append(top)

    if not rows:
        return pd.DataFrame(), pd.DataFrame()

    all_top = pd.concat(rows, ignore_index=True)
    summary = (
        all_top.groupby("SNP", as_index=False)
        .agg(
            Top_SHAP_Frequency=("Sample", "nunique"),
            Mean_Absolute_SHAP=("Absolute SHAP", "mean"),
            Mean_SHAP_Value=("SHAP Value", "mean"),
        )
        .sort_values(["Top_SHAP_Frequency", "Mean_Absolute_SHAP"], ascending=[False, False])
        .head(top_n)
        .reset_index(drop=True)
    )
    summary["Mean_Absolute_SHAP"] = summary["Mean_Absolute_SHAP"].astype(float).round(5)
    summary["Mean_SHAP_Value"] = summary["Mean_SHAP_Value"].astype(float).round(5)
    summary = attach_biology_mapping(summary, mapping_df)
    return summary, all_top


def count_available(series, unavailable=("Unmapped", "Not available")):
    if series is None or len(series) == 0:
        return 0
    values = []
    for item in series.dropna():
        values.extend(_split_multi_value(item))
    return len({v for v in values if v not in unavailable})


def make_dynamic_shap_bar(top_df, title="Top SNP contributions for this prediction"):
    df = top_df.sort_values("Absolute SHAP", ascending=True).copy()
    colors = ["#b45309" if v > 0 else "#0d9488" for v in df["SHAP Value"]]

    fig = pgo.Figure()
    fig.add_trace(
        pgo.Bar(
            x=df["SHAP Value"],
            y=df["SNP"],
            orientation="h",
            marker=dict(color=colors),
            hovertemplate="SNP: %{y}<br>SHAP value: %{x:.5f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        height=max(360, 24 * len(df) + 130),
        margin=dict(l=10, r=10, t=55, b=30),
        xaxis_title="SHAP value impact on model output",
        yaxis_title="",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f0f2ff", size=12),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)", zeroline=True, zerolinecolor="rgba(255,255,255,0.45)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
    return fig


def build_dynamic_sankey(mapped_df, value_col="Absolute SHAP"):
    if mapped_df is None or mapped_df.empty:
        return None

    labels = []
    label_to_id = {}
    sources, targets, values = [], [], []

    def node(label):
        label = _safe_text(label)
        if label not in label_to_id:
            label_to_id[label] = len(labels)
            labels.append(label)
        return label_to_id[label]

    for _, row in mapped_df.iterrows():
        snp = _safe_text(row.get("SNP"), "Unknown SNP")
        weight = float(abs(row.get(value_col, 1.0))) if value_col in mapped_df.columns else 1.0
        weight = max(weight, 0.0001)

        genes = _split_multi_value(row.get("Gene"), "Unmapped")
        proteins = _split_multi_value(row.get("Protein"), "Not available")
        pathways = _split_multi_value(row.get("Pathway"), "Not available")

        for gene in genes:
            sources.append(node(snp))
            targets.append(node(gene))
            values.append(weight)

            for protein in proteins:
                sources.append(node(gene))
                targets.append(node(protein))
                values.append(weight)

                for pathway in pathways:
                    sources.append(node(protein))
                    targets.append(node(pathway))
                    values.append(weight)

                    sources.append(node(pathway))
                    targets.append(node("RA Risk Interpretation"))
                    values.append(weight)

    if not labels or not sources:
        return None

    fig = pgo.Figure(
        data=[
            pgo.Sankey(
                arrangement="snap",
                node=dict(
                    pad=18,
                    thickness=16,
                    line=dict(color="rgba(255,255,255,0.18)", width=0.4),
                    label=labels,
                    color="rgba(14,165,233,0.82)",
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color="rgba(14,165,233,0.20)",
                ),
            )
        ]
    )
    fig.update_layout(
        height=420,
        margin=dict(l=5, r=5, t=10, b=5),
        font=dict(size=11, color="#f0f2ff"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_optional_research_images():
    base = os.path.join(os.path.dirname(__file__), "assets")
    shap_candidates = [
        os.path.join(base, "XGB_Additive_SHAP_Top20.png"),
        os.path.join(base, "XGB_Additive_SHAP_beeswarm_Top20.png"),
    ]
    string_path = os.path.join(base, "string_network.png")
    shap_path = next((p for p in shap_candidates if os.path.exists(p)), None)

    with st.expander("Optional research figures", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            if shap_path:
                st.image(shap_path, caption="Research SHAP summary figure", use_container_width=True)
            else:
                st.info("Optional SHAP image is not available in assets/.")
        with c2:
            if os.path.exists(string_path):
                st.image(string_path, caption="Optional STRING/network figure", use_container_width=True)
            else:
                st.info("Optional STRING/network image is not available in assets/.")


def render_dynamic_biological_interpretation(artifact, X, probs=None, sample_name=None, top_n=15):
    """
    Dynamic interpretation.
    - If X has one row: sample-level SHAP explanation.
    - If X has multiple rows: batch-level summary of recurring top SHAP SNPs.
    """
    mapping_df = load_biological_mapping()

    st.markdown("""
    <div class="bio-note presentation-card">
      <div class="simple-kicker">Dynamic Biological Interpretation</div>
      <p>
        GenomicAI does not only predict RA genomic risk; it explains which additive SNP markers
        contributed to this specific prediction and connects them to immune-related biological pathways
        when a real mapping file is available.
      </p>
    </div>
    """, unsafe_allow_html=True)

    try:
        shap_df = compute_shap_values(artifact, X)
    except Exception as e:
        st.warning(f"Dynamic SHAP explanation could not be generated for this run: {e}")
        render_optional_research_images()
        return

    if len(X) == 1:
        sample_id = X.index[0]
        top = get_top_shap_for_sample(shap_df, X, sample_id, top_n=top_n)
        mapped = attach_biology_mapping(top, mapping_df)

        mapped_genes = count_available(mapped["Gene"])
        mapped_pathways = count_available(mapped["Pathway"])

        m1, m2, m3 = st.columns(3)
        m1.metric("Top contributing SNPs", len(mapped))
        m2.metric("Mapped candidate genes", mapped_genes)
        m3.metric("Mapped immune pathways", mapped_pathways)

        label = sample_name if sample_name else str(sample_id)
        if probs is not None:
            st.markdown(f"**Current sample interpretation:** `{label}` · Risk score: **{float(np.ravel(probs)[0]):.1%}**")
        else:
            st.markdown(f"**Current sample interpretation:** `{label}`")

        display_cols = [
            "SNP", "Additive Encoding", "SHAP Value", "Absolute SHAP",
            "Contribution Direction", "Gene", "Protein", "Pathway"
        ]
        st.dataframe(mapped[display_cols], use_container_width=True, hide_index=True)
        st.plotly_chart(make_dynamic_shap_bar(mapped), use_container_width=True)

        sankey = build_dynamic_sankey(mapped, value_col="Absolute SHAP")
        if sankey is not None:
            st.markdown("#### Dynamic SNP → Gene → Protein → Pathway interpretation")
            st.plotly_chart(sankey, use_container_width=True)
        else:
            st.info("Sankey diagram is not available because no interpretable mapping could be built.")

    else:
        summary, all_top = summarize_batch_shap(shap_df, X, mapping_df, top_n=top_n)
        mapped_genes = count_available(summary["Gene"]) if not summary.empty else 0
        mapped_pathways = count_available(summary["Pathway"]) if not summary.empty else 0

        m1, m2, m3 = st.columns(3)
        m1.metric("Samples interpreted", len(X))
        m2.metric("Recurring top SNPs", len(summary))
        m3.metric("Mapped immune pathways", mapped_pathways)

        st.markdown("**Batch-level interpretation:** recurring SNPs among the top SHAP contributors across uploaded samples.")
        display_cols = [
            "SNP", "Top_SHAP_Frequency", "Mean_Absolute_SHAP", "Mean_SHAP_Value",
            "Gene", "Protein", "Pathway"
        ]
        st.dataframe(summary[display_cols], use_container_width=True, hide_index=True)

        if not summary.empty:
            batch_bar = summary.rename(columns={"Mean_Absolute_SHAP": "Absolute SHAP", "Mean_SHAP_Value": "SHAP Value"})
            st.plotly_chart(make_dynamic_shap_bar(batch_bar, title="Recurring SNPs by mean absolute SHAP across uploaded samples"), use_container_width=True)
            sankey = build_dynamic_sankey(batch_bar, value_col="Absolute SHAP")
            if sankey is not None:
                st.markdown("#### Dynamic batch SNP → Gene → Protein → Pathway interpretation")
                st.plotly_chart(sankey, use_container_width=True)

    if mapping_df.empty:
        st.info("No biological mapping CSV was found. SHAP-based Top SNPs are dynamic; gene/protein/pathway fields are shown as Unmapped / Not available until a real mapping file is added.")

    render_optional_research_images()

    st.markdown(f"""
    <div class="disclaimer-card presentation-card">
      <div class="dc-kicker">Research disclaimer</div>
      <div class="dc-body">{DISCLAIMER_TEXT}</div>
    </div>
    """, unsafe_allow_html=True)

# ── State ─────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "Home"

def go(p):
    st.session_state.page = p
    st.rerun()

# ══════════════════════════════════════════════════════════════
#  NAVBAR
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="navbar">
  <div class="nb-brand">
    <span class="nb-dna">🧬</span>
    <span class="nb-name">GenomicAI</span>
  </div>
  <div class="nb-tagline">Research prototype · Genomic data · Risk estimation</div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    if st.button("🏠  Home", use_container_width=True,
                 type="primary" if st.session_state.page == "Home" else "secondary"):
        go("Home")
with c2:
    if st.button("🔬  Try it", use_container_width=True,
                 type="primary" if st.session_state.page == "Predict" else "secondary"):
        go("Predict")
with c3:
    if st.button("📄  About", use_container_width=True,
                 type="primary" if st.session_state.page == "About" else "secondary"):
        go("About")

st.markdown('<div class="nb-line"></div>', unsafe_allow_html=True)

presentation_mode = st.checkbox(
    "Simple presentation mode",
    value=True,
    help="Recommended for presenting to engineering faculty from different fields."
)

page = st.session_state.page

# ══════════════════════════════════════════════════════════════
#  HOME
# ══════════════════════════════════════════════════════════════
if page == "Home":
    st.markdown("""
    <div class="hero presentation-card">
      <div class="hero-eyebrow">Interactive research prototype for academic presentation</div>
      <h1 class="hero-title">GenomicAI</h1>
      <p class="hero-sub">
        Estimating rheumatoid arthritis genomic risk using genetic data
      </p>
      <p class="hero-sub hero-sub-small">
        A research prototype that estimates genetic susceptibility patterns.
        It is not a medical diagnosis.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-card presentation-card">
      <div class="dc-kicker">The Problem</div>
      <div class="dc-body">
        Rheumatoid arthritis is often detected after symptoms appear, when joint inflammation
        and tissue damage may have already started. Early risk awareness remains difficult
        before clear disease progression, especially when genomic information is not used
        to support research-grade risk stratification.
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="simple-intro presentation-card">
      <div class="simple-kicker">Our Solution</div>
      <p>
        GenomicAI uses additive genomic SNP encoding <strong>0 / 1 / 2</strong> and machine learning
        to estimate a program-based rheumatoid arthritis risk score. Instead of stopping at
        prediction, the app dynamically explains which genetic markers contributed to the result.
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-label presentation-label">Project workflow</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="workflow-simple presentation-card">
      <div class="simple-step"><div class="step-num">1</div><div>Real genetic data</div></div>
      <div class="simple-step"><div class="step-num">2</div><div>Clean and review the data</div></div>
      <div class="simple-step"><div class="step-num">3</div><div>Select important genetic markers</div></div>
      <div class="simple-step"><div class="step-num">4</div><div>Train a smart prediction program</div></div>
      <div class="simple-step"><div class="step-num">5</div><div>Estimate risk level</div></div>
      <div class="simple-step"><div class="step-num">6</div><div>Explain the result biologically</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-label presentation-label">Simple numbers for the presentation</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="simple-numbers presentation-card">
      <div class="number-card"><span>People in the dataset</span><strong>2,062</strong></div>
      <div class="number-card"><span>Rheumatoid arthritis cases</span><strong>868</strong></div>
      <div class="number-card"><span>Non-RA controls in the dataset</span><strong>1,194</strong></div>
      <div class="number-card"><span>Initial genetic markers</span><strong>531,689</strong></div>
      <div class="number-card"><span>Focused final marker set</span><strong>212 markers</strong></div>
      <div class="number-card"><span>Discrimination index reached about</span><strong>92%</strong></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="bio-note presentation-card">
      <div class="simple-kicker">Why not stop at one prediction number?</div>
      <p>
        In medical research, a number alone is not enough. We also looked at which
        genetic markers the program relied on, then checked whether those markers
        connect to genes and biological pathways related to immunity and rheumatoid
        arthritis. This supports the biological interpretation without claiming
        proof, certainty, or clinical diagnosis.
      </p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Technical details for specialists", expanded=not presentation_mode):
        st.markdown("""
        - Dataset: NARAC
        - 2,062 samples: 868 cases, 1,194 controls
        - Raw SNPs: 531,689 autosomal SNPs
        - Final additive features: 212
        - Models tested: KNN, Linear SVM, Logistic Regression, Random Forest, Naive Bayes, XGBoost
        - Best model: XGBoost
        - ROC-AUC ≈ 0.919
        - PR-AUC ≈ 0.885
        - XAI: SHAP and permutation importance
        - Biological interpretation: SNP-to-gene mapping, STRING, enrichment analysis
        """)

    st.stop()

    # ── Hero ──────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
      <div class="hero-eyebrow">
        <span class="pulse-dot"></span>
        Research Prototype &nbsp;·&nbsp; Precision Genomics &nbsp;·&nbsp; Academic Demonstration
      </div>

      <h1 class="hero-title">
        Estimate Rheumatoid Arthritis<br>
        Genomic Risk
      </h1>

      <p class="hero-sub">
        An interactive research prototype for SNP-based RA risk prediction
        using machine learning.
      </p>

      <div class="hero-pills">
        <span class="pill">531,689 SNPs</span>
        <span class="arr">→</span>
        <span class="pill">313 RA Markers</span>
        <span class="arr">→</span>
        <span class="pill">212 Features</span>
        <span class="arr">→</span>
        <span class="pill pill-teal">XGBoost · RA Risk Score</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    render_disclaimer()

    # ── Why it matters ────────────────────────────────────────
    st.markdown("""
    <div class="why-grid">
      <div class="why-card wc-animate" style="animation-delay:0s">
        <div class="why-icon">⏱️</div>
        <div class="why-title">Risk Stratification</div>
        <div class="why-body">A model-based genomic risk score can support research discussion of RA-like genomic patterns before clinical evaluation.</div>
      </div>
      <div class="why-card wc-animate" style="animation-delay:0.1s">
        <div class="why-icon">🧬</div>
        <div class="why-title">Biology-Driven AI</div>
        <div class="why-body">Not black-box statistics. Every feature is a validated RA-associated SNP marker — selected from peer-reviewed genomic literature, not a generic filter.</div>
      </div>
      <div class="why-card wc-animate" style="animation-delay:0.2s">
        <div class="why-icon">🌍</div>
        <div class="why-title">Built for Scale</div>
        <div class="why-body">Designed as a clinical decision-support concept for populations with limited specialist access. In Egypt, RA onset averages 38.4 years, motivating earlier research-grade risk stratification.</div>
      </div>
      <div class="why-card wc-animate" style="animation-delay:0.3s">
        <div class="why-icon">🔍</div>
        <div class="why-title">Explainable Output</div>
        <div class="why-body">Explainability analysis was performed in the full research pipeline. Future versions can integrate patient-level SHAP explanations directly into the application.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Numbers ───────────────────────────────────────────────
    st.markdown('<div class="sec-label">DATASET AT A GLANCE</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="stats-row">
      <div class="stat-card sc-purple">
        <div class="sc-val">2,062</div>
        <div class="sc-name">Individuals</div>
        <div class="sc-sub">NARAC cohort</div>
      </div>
      <div class="stat-card">
        <div class="sc-val" style="color:#94a3b8">531K</div>
        <div class="sc-name">Raw SNPs</div>
        <div class="sc-sub">22 autosomes</div>
      </div>
      <div class="stat-card sc-teal">
        <div class="sc-val">313</div>
        <div class="sc-name">RA Markers</div>
        <div class="sc-sub">Knowledge-driven</div>
      </div>
      <div class="stat-card sc-purple">
        <div class="sc-val">212</div>
        <div class="sc-name">ML Features</div>
        <div class="sc-sub">Additive 0/1/2</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Pipeline Flowchart ────────────────────────────────────
    st.markdown('<div class="sec-label">6-STAGE PIPELINE</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="pipeline-wrap">
      <div class="pipe-step ps-animate" style="animation-delay:0s">
        <div class="ps-num">01</div>
        <div class="ps-icon">📂</div>
        <div class="ps-title">Data<br>Description</div>
        <div class="ps-body">PED/MAP inspection per chromosome · SNP counts · missing data mapping</div>
      </div>
      <div class="pipe-conn"><div class="pc-line"></div><div class="pc-arrow">▶</div></div>
      <div class="pipe-step ps-animate" style="animation-delay:0.08s">
        <div class="ps-num">02</div>
        <div class="ps-icon">🛡️</div>
        <div class="ps-title">QC<br>Validation</div>
        <div class="ps-body">Structural consistency · phenotype coding · genotype token validity</div>
      </div>
      <div class="pipe-conn"><div class="pc-line"></div><div class="pc-arrow">▶</div></div>
      <div class="pipe-step ps-animate ps-highlight" style="animation-delay:0.16s">
        <div class="ps-num">03</div>
        <div class="ps-icon">🎯</div>
        <div class="ps-title">SNP<br>Refinement</div>
        <div class="ps-body">531,689 → 313 RA-validated markers via external literature intersection</div>
      </div>
      <div class="pipe-conn"><div class="pc-line"></div><div class="pc-arrow">▶</div></div>
      <div class="pipe-step ps-animate" style="animation-delay:0.24s">
        <div class="ps-num">04</div>
        <div class="ps-icon">🔄</div>
        <div class="ps-title">Beagle<br>Imputation</div>
        <div class="ps-body">Missing-only overlay · original calls preserved · PLINK → VCF → Beagle</div>
      </div>
      <div class="pipe-conn"><div class="pc-line"></div><div class="pc-arrow">▶</div></div>
      <div class="pipe-step ps-animate" style="animation-delay:0.32s">
        <div class="ps-num">05</div>
        <div class="ps-icon">⚙️</div>
        <div class="ps-title">Additive<br>Encoding</div>
        <div class="ps-body">0/1/2 schema · filter monomorphic & non-biallelic · 313 → 212 features</div>
      </div>
      <div class="pipe-conn"><div class="pc-line"></div><div class="pc-arrow">▶</div></div>
      <div class="pipe-step ps-animate ps-end" style="animation-delay:0.40s">
        <div class="ps-num">06</div>
        <div class="ps-icon">🤖</div>
        <div class="ps-title">XGBoost<br>Prediction</div>
        <div class="ps-body">Nested CV · hyperparameter tuning · model-based genomic risk score</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Data flow (horizontal animated arrow strip) ───────────
    st.markdown('<div class="sec-label">DATA FLOW</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="flow-strip">
      <div class="fs-node fs-start">
        <div class="fs-icon">📁</div>
        <div class="fs-label">Raw PED/MAP</div>
        <div class="fs-sub">531,689 SNPs</div>
      </div>
      <div class="fs-arrow">
        <div class="fs-line"></div>
        <div class="fs-dot"></div>
      </div>
      <div class="fs-node">
        <div class="fs-icon">🛡️</div>
        <div class="fs-label">QC Filter</div>
        <div class="fs-sub">Validated</div>
      </div>
      <div class="fs-arrow">
        <div class="fs-line"></div>
        <div class="fs-dot"></div>
      </div>
      <div class="fs-node fs-teal">
        <div class="fs-icon">🎯</div>
        <div class="fs-label">RA Panel</div>
        <div class="fs-sub">313 markers</div>
      </div>
      <div class="fs-arrow">
        <div class="fs-line"></div>
        <div class="fs-dot"></div>
      </div>
      <div class="fs-node">
        <div class="fs-icon">🔄</div>
        <div class="fs-label">Imputation</div>
        <div class="fs-sub">Beagle</div>
      </div>
      <div class="fs-arrow">
        <div class="fs-line"></div>
        <div class="fs-dot"></div>
      </div>
      <div class="fs-node">
        <div class="fs-icon">⚙️</div>
        <div class="fs-label">Encoding</div>
        <div class="fs-sub">212 features</div>
      </div>
      <div class="fs-arrow">
        <div class="fs-line"></div>
        <div class="fs-dot"></div>
      </div>
      <div class="fs-node fs-end">
        <div class="fs-icon">🤖</div>
        <div class="fs-label">XGBoost</div>
        <div class="fs-sub">RA Risk Score</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tech stack ────────────────────────────────────────────
    st.markdown('<div class="sec-label">TECHNOLOGY STACK</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="tech-row">
      <div class="tech-card">
        <div class="tc-head">🧬 Bioinformatics</div>
        <div class="tc-item"><span class="tc-k">PLINK</span><span class="tc-v">Binary genotype conversion</span></div>
        <div class="tc-item"><span class="tc-k">Beagle</span><span class="tc-v">Probabilistic imputation</span></div>
        <div class="tc-item"><span class="tc-k">VCF / PED-MAP</span><span class="tc-v">Standard genomic formats</span></div>
      </div>
      <div class="tech-card">
        <div class="tc-head">🤖 Machine Learning</div>
        <div class="tc-item"><span class="tc-k">XGBoost</span><span class="tc-v">Deployed gradient booster</span></div>
        <div class="tc-item"><span class="tc-k">Scikit-learn</span><span class="tc-v">Preprocessing · nested CV</span></div>
        <div class="tc-item"><span class="tc-k">SHAP</span><span class="tc-v">Research-pipeline explainability</span></div>
      </div>
      <div class="tech-card">
        <div class="tc-head">🚀 Deployment</div>
        <div class="tc-item"><span class="tc-k">Streamlit</span><span class="tc-v">Interactive web interface</span></div>
        <div class="tc-item"><span class="tc-k">Joblib</span><span class="tc-v">Model serialization</span></div>
        <div class="tc-item"><span class="tc-k">Pandas / NumPy</span><span class="tc-v">Data alignment & handling</span></div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PREDICT
# ══════════════════════════════════════════════════════════════
elif page == "Predict":

    st.markdown("""
    <div class="pred-header">
      <div class="pred-icon-wrap">🔬</div>
      <div>
        <h1 class="pred-title">Risk Estimation Demo</h1>
        <p class="pred-sub">Try the app with a built-in demo sample, or upload a CSV using the required genetic marker columns.</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    render_disclaimer()

    model_artifact = load_model()

    # Model status card
    if model_artifact:
        st.markdown(f"""
        <div class="model-status ms-ok">
          <div class="ms-left">
            <div class="ms-dot ms-dot-ok"></div>
            <div>
              <div class="ms-title">Prediction program loaded</div>
              <div class="ms-sub">Uses 212 genetic markers · Demo threshold {CLASSIFICATION_THRESHOLD:.2f}</div>
            </div>
          </div>
          <div class="ms-badge">READY</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
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
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Mode selector
    st.markdown('<div class="mode-label presentation-label">Choose a demo mode</div>', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["📋  Built-in demo sample", "📤  Upload CSV file"])

    # ── TAB 1 ─────────────────────────────────────────────────
    with tab1:
        st.markdown("""
        <div class="tab-desc presentation-card">
          Choose a built-in sample to demonstrate the idea without uploading a file.
          This output is for educational demonstration only, not diagnosis.
        </div>
        """, unsafe_allow_html=True)

        demo_path = os.path.join(os.path.dirname(__file__), "data", "demo_samples.csv")
        if os.path.exists(demo_path):
            demo_df = pd.read_csv(demo_path, index_col=0)
            demo_labels = {f"Demo sample {i + 1}": idx for i, idx in enumerate(demo_df.index)}

            col_s, col_b = st.columns([3, 1])
            with col_s:
                selected_label = st.selectbox("Choose a demo sample", list(demo_labels.keys()),
                                              label_visibility="collapsed")
                selected = demo_labels[selected_label]
            with col_b:
                run = st.button("▶  Estimate risk", type="primary",
                                use_container_width=True, key="dr")

            if run and model_artifact:
                row   = demo_df.loc[[selected]]
                label_col = next((c for c in LABEL_COLUMNS if c in row.columns), None)
                exp   = row[label_col].values[0] if label_col else None
                required_cols = get_required_feature_columns(model_artifact)
                feat  = validate_features(row, required_cols)
                preds, probs = predict(model_artifact, feat)
                cls   = "Genetic pattern suggesting higher rheumatoid arthritis susceptibility" if preds[0] == 1 else "Genetic pattern suggesting lower rheumatoid arthritis susceptibility"
                prob  = float(probs[0])
                is_higher = preds[0] == 1

                # Big result card
                card_cls = "result-ra" if is_higher else "result-ctrl"
                icon     = "▲" if is_higher else "●"
                st.markdown(f"""
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
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="threshold-note presentation-card">
                  This result is a risk estimate, not a diagnosis.<br>
                  Threshold used in this demo: {CLASSIFICATION_THRESHOLD:.2f}.<br>
                  The probability score is more informative than the final class because it represents the program-based risk level.
                </div>
                """, unsafe_allow_html=True)

                # Risk gauge bar
                bar_pct = int(prob * 100)
                bar_color = "#b45309" if is_higher else "#0d9488"
                st.markdown(f"""
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
                """, unsafe_allow_html=True)

                render_dynamic_biological_interpretation(
                    model_artifact,
                    feat,
                    probs=[prob],
                    sample_name=selected_label,
                    top_n=15,
                )

                if exp is not None:
                    ok = is_higher == (str(exp).upper() in ["1", "RA", "CASE"])
                    st.markdown(f"""
                    <div class="verify-row">
                      <span class="vr-badge {'vr-ok' if ok else 'vr-err'}">
                        {'Matches reference label' if ok else 'Differs from reference label'}
                      </span>
                      <span class="vr-exp">Reference label in data: <strong>{exp}</strong></span>
                    </div>
                    """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="empty-state">
              <div class="es-icon">📂</div>
              <div class="es-title">No demo samples found</div>
              <div class="es-body">Add <code>demo_samples.csv</code> to the <code>data/</code> folder.<br>
              Columns = genetic markers (0/1/2), with optional <code>label</code> column.</div>
            </div>
            """, unsafe_allow_html=True)

    # ── TAB 2 ─────────────────────────────────────────────────
    with tab2:
        st.markdown("""
        <div class="tab-desc presentation-card">
          Upload a CSV containing the required genetic marker columns. Each row is one sample.
          Marker values must be 0, 1, or 2 only. An optional <code>label</code> column can be included
          for reference-label comparison.
        </div>
        """, unsafe_allow_html=True)

        f = st.file_uploader("Upload CSV file", type=["csv"], label_visibility="collapsed")
        if f:
            try:
                df = pd.read_csv(f, index_col=0)
                st.markdown(f"""
                <div class="file-info">
                  <span class="fi-icon">📊</span>
                  <span>Loaded <strong>{len(df)}</strong> sample(s) · Found <strong>{df.shape[1]}</strong> column(s)</span>
                </div>
                """, unsafe_allow_html=True)

                if st.button("▶  Estimate risk for all samples",
                             type="primary", key="ur"):
                    if model_artifact:
                        label_col = next((c for c in LABEL_COLUMNS if c in df.columns), None)
                        lc   = df[label_col] if label_col else None
                        required_cols = get_required_feature_columns(model_artifact)
                        feat = validate_features(df, required_cols)
                        preds, probs = predict(model_artifact, feat)

                        res = pd.DataFrame({
                            "Sample": feat.index,
                            "Genetic Pattern": [
                                "Genetic pattern suggesting higher rheumatoid arthritis susceptibility" if p == 1
                                else "Genetic pattern suggesting lower rheumatoid arthritis susceptibility" for p in preds
                            ],
                            "Program-based Genomic Risk Score": [round(float(p), 4) for p in probs],
                            "Approximate Level": [
                                "Higher susceptibility" if p >= .7 else
                                "Intermediate susceptibility" if p >= .4 else
                                "Lower susceptibility" for p in probs
                            ],
                        })
                        if lc is not None:
                            res["Reference Label"] = lc.values

                        def style_r(r):
                            if r["Genetic Pattern"] == "Genetic pattern suggesting higher rheumatoid arthritis susceptibility":
                                return ["background:rgba(180,83,9,.10)"]*len(r)
                            return ["background:rgba(20,184,166,.08)"]*len(r)

                        st.markdown(f"""
                        <div class="threshold-note presentation-card">
                          This result is a risk estimate, not a diagnosis.<br>
                          Threshold used in this demo: {CLASSIFICATION_THRESHOLD:.2f}.<br>
                          The probability score is more informative than the final class because it represents the program-based risk level.
                        </div>
                        """, unsafe_allow_html=True)

                        st.dataframe(
                            res.style.apply(style_r, axis=1),
                            use_container_width=True,
                            height=min(480, 60 + len(res)*38),
                        )

                        if len(feat) == 1:
                            render_dynamic_biological_interpretation(
                                model_artifact,
                                feat,
                                probs=probs,
                                sample_name=str(feat.index[0]),
                                top_n=15,
                            )
                        else:
                            render_dynamic_biological_interpretation(
                                model_artifact,
                                feat,
                                probs=probs,
                                sample_name="uploaded batch",
                                top_n=15,
                            )
            except Exception as e:
                st.error(f"Error reading file: {e}")


# ══════════════════════════════════════════════════════════════
#  ABOUT
# ══════════════════════════════════════════════════════════════
elif page == "About":
    st.markdown("""
    <div class="vision-banner presentation-card">
      <div class="vb-label">About the project</div>
      <blockquote class="vb-quote">
        GenomicAI is a graduation research prototype showing how genomic data and
        a smart prediction program can estimate rheumatoid arthritis susceptibility
        patterns. The goal is academic demonstration and education, not diagnosis.
      </blockquote>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-label presentation-label">What makes the project clear?</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="impact-grid presentation-card">
      <div class="ig-card"><div class="ig-title">Clear engineering idea</div><div class="ig-body">Turn a very large genetic file into an understandable risk score.</div></div>
      <div class="ig-card"><div class="ig-title">Focused data filtering</div><div class="ig-body">Instead of using every marker, the pipeline focuses on markers most relevant to the research question.</div></div>
      <div class="ig-card"><div class="ig-title">Biological interpretation</div><div class="ig-body">The project does not stop at a number; it tries to give biological meaning to important markers.</div></div>
      <div class="ig-card"><div class="ig-title">Clear limitations</div><div class="ig-body">The model is a research prototype and needs validation on local data before practical use.</div></div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Technical details for specialists", expanded=not presentation_mode):
        st.markdown("""
        - Dataset: NARAC
        - 2,062 samples: 868 cases, 1,194 controls
        - Raw SNPs: 531,689 autosomal SNPs
        - Final additive features: 212
        - Models tested: KNN, Linear SVM, Logistic Regression, Random Forest, Naive Bayes, XGBoost
        - Best model: XGBoost
        - ROC-AUC ≈ 0.919, PR-AUC ≈ 0.885
        - XAI: SHAP and permutation importance
        - Biological interpretation: SNP-to-gene mapping, STRING, enrichment analysis
        """)

    render_disclaimer()
    st.stop()

    # Vision banner
    st.markdown("""
    <div class="vision-banner">
      <div class="vb-glow"></div>
      <div class="vb-label">THE MISSION</div>
      <blockquote class="vb-quote">
        "As a research prototype, GenomicAI demonstrates how SNP-based
        machine learning could support RA genomic risk stratification
        as a future clinical decision-support concept."
      </blockquote>
    </div>
    """, unsafe_allow_html=True)

    # Impact grid
    st.markdown('<div class="sec-label">WHY THIS PROJECT MATTERS</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="impact-grid">
      <div class="ig-card ig-animate" style="animation-delay:0s">
        <div class="ig-icon">🩺</div>
        <div class="ig-title">Research Impact</div>
        <div class="ig-body">RA affects 1% of the global population with lifelong consequences.
        In Egypt, onset averages <strong>38.4 years</strong> — a working-age demographic.
        SNP-based genomic risk prediction can support earlier academic discussion of risk patterns.</div>
      </div>
      <div class="ig-card ig-animate" style="animation-delay:0.1s">
        <div class="ig-icon">🌍</div>
        <div class="ig-title">Egyptian Healthcare Context</div>
        <div class="ig-body">Egypt's Universal Health Insurance expansion and Digital Egypt 2030
        create a potential institutional path for evaluated decision-support concepts.
        With specialists concentrated in Cairo and Alexandria,
        scalable genomic risk stratification research may help study geographic access gaps.</div>
      </div>
      <div class="ig-card ig-animate" style="animation-delay:0.2s">
        <div class="ig-icon">🔬</div>
        <div class="ig-title">Scientific Contribution</div>
        <div class="ig-body">Knowledge-driven SNP selection — 531,689 raw markers distilled to
        313 RA-validated ones — is reproducible, biologically grounded, and
        extensible to other autoimmune diseases and MENA-region cohorts.</div>
      </div>
      <div class="ig-card ig-animate" style="animation-delay:0.3s">
        <div class="ig-icon">💡</div>
        <div class="ig-title">Innovation Edge</div>
        <div class="ig-body">Explainability analysis was performed in the full research pipeline.
        Future versions can integrate patient-level SHAP explanations directly into the application.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Dataset stats
    st.markdown('<div class="sec-label">NARAC COHORT</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="stats-row">
      <div class="stat-card sc-purple"><div class="sc-val">2,062</div><div class="sc-name">Individuals</div><div class="sc-sub">Full cohort</div></div>
      <div class="stat-card"><div class="sc-val" style="color:#f87171">868</div><div class="sc-name">RA Cases</div><div class="sc-sub">42.1%</div></div>
      <div class="stat-card sc-teal"><div class="sc-val">1,194</div><div class="sc-name">Controls</div><div class="sc-sub">57.9%</div></div>
      <div class="stat-card sc-purple"><div class="sc-val">22</div><div class="sc-name">Chromosomes</div><div class="sc-sub">Autosomal only</div></div>
    </div>
    """, unsafe_allow_html=True)

    # Strengths + Limits
    st.markdown('<div class="sec-label">STRENGTHS & LIMITATIONS</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sl-wrap">
      <div class="sl-card">
        <div class="slc-head ok-head">✅ Strengths</div>
        <div class="sl-item ok-item">End-to-end pipeline: raw chromosomal files → live web prototype</div>
        <div class="sl-item ok-item">Biology-driven SNP selection — not statistical shortcuts</div>
        <div class="sl-item ok-item">Missing-only imputation preserves original genotype calls</div>
        <div class="sl-item ok-item">Explainability analysis completed in the full research pipeline</div>
        <div class="sl-item ok-item">Nested cross-validation — unbiased generalization estimates</div>
      </div>
      <div class="sl-card">
        <div class="slc-head warn-head">⚠️ Current Limitations</div>
        <div class="sl-item warn-item">Research prototype — not clinically certified</div>
        <div class="sl-item warn-item">Genomic features only — no imaging or lab values</div>
        <div class="sl-item warn-item">Trained on North American cohort — needs Egyptian validation</div>
        <div class="sl-item warn-item">Production use requires security & governance hardening</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Roadmap
    st.markdown('<div class="sec-label">ROADMAP TO REAL-WORLD IMPACT</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="roadmap">
      <div class="rm-step"><div class="rm-num">1</div><div class="rm-content"><div class="rm-title">Egyptian Cohort Validation</div><div class="rm-body">Partner with ECR-affiliated centers (Cairo University, Ain Shams, Sohag) to collect Egyptian RA samples for model retraining.</div></div></div>
      <div class="rm-conn"></div>
      <div class="rm-step"><div class="rm-num">2</div><div class="rm-content"><div class="rm-title">Institutional Pilot</div><div class="rm-body">Evaluate as a research decision-support concept in 2–3 tertiary rheumatology centers under IRB oversight.</div></div></div>
      <div class="rm-conn"></div>
      <div class="rm-step"><div class="rm-num">3</div><div class="rm-content"><div class="rm-title">Digital Egypt 2030 Integration</div><div class="rm-body">Align with UHI infrastructure and EHR interoperability standards for national health system embedding.</div></div></div>
      <div class="rm-conn"></div>
      <div class="rm-step"><div class="rm-num">4</div><div class="rm-content"><div class="rm-title">Regional Scale-Up</div><div class="rm-body">Study extension to Upper Egypt and rural governorates where specialist density is lowest and risk-stratification research value may be highest.</div></div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
      <strong>Research Disclaimer:</strong> This application is a research prototype for
      educational and academic demonstration purposes. It provides a model-based genomic
      risk estimate and does not replace physician evaluation, laboratory testing,
      imaging, or clinical diagnosis.
    </div>
    """, unsafe_allow_html=True)
