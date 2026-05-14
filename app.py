import streamlit as st
import pandas as pd
import joblib, os
import plotly.graph_objects as pgo

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


# ══════════════════════════════════════════════════════════════
# BIOLOGICAL INTERPRETATION SECTION
# ══════════════════════════════════════════════════════════════

def _asset_path(filename: str) -> str:
    return os.path.join(os.path.dirname(__file__), "assets", filename)


def _first_existing_asset(filenames):
    for filename in filenames:
        path = _asset_path(filename)
        if os.path.exists(path):
            return path
    return None


def build_biological_sankey():
    labels = [
        "Prioritized SNPs",
        "Candidate Genes",
        "Proteins / STRING Network",
        "Immune Pathways",
        "RA Risk Interpretation",
    ]

    fig = pgo.Figure(
        data=[
            pgo.Sankey(
                arrangement="snap",
                node=dict(
                    pad=20,
                    thickness=18,
                    line=dict(color="rgba(15, 23, 42, 0.25)", width=0.5),
                    label=labels,
                    color=[
                        "rgba(124, 58, 237, 0.92)",
                        "rgba(14, 165, 233, 0.90)",
                        "rgba(20, 184, 166, 0.90)",
                        "rgba(245, 158, 11, 0.90)",
                        "rgba(180, 83, 9, 0.95)",
                    ],
                ),
                link=dict(
                    source=[0, 1, 2, 3],
                    target=[1, 2, 3, 4],
                    value=[32, 54, 90, 90],
                    color=[
                        "rgba(124, 58, 237, 0.22)",
                        "rgba(14, 165, 233, 0.22)",
                        "rgba(20, 184, 166, 0.22)",
                        "rgba(245, 158, 11, 0.25)",
                    ],
                ),
            )
        ]
    )

    fig.update_layout(
        height=330,
        margin=dict(l=5, r=5, t=10, b=5),
        font=dict(size=13, color="#f0f2ff"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_biological_interpretation(prob=None, sample_name=None):
    st.markdown(
        """
        <style>
        .bio-wrap {
            margin-top: 1.1rem;
            padding: 1.1rem 1.15rem;
            border-radius: 14px;
            background: linear-gradient(135deg, rgba(124,58,237,0.10), rgba(13,148,136,0.07));
            border: 1px solid rgba(124,58,237,0.20);
            border-left: 4px solid #2dd4bf;
            box-shadow: 0 12px 28px rgba(0,0,0,0.16);
        }
        .bio-kicker {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.58rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: #2dd4bf;
            font-weight: 800;
            margin-bottom: 0.35rem;
        }
        .bio-title {
            font-family: 'Fraunces', serif;
            font-size: 1.55rem;
            line-height: 1.15;
            font-weight: 700;
            color: #f0f2ff;
            margin-bottom: 0.45rem;
        }
        .bio-text {
            font-size: 0.86rem;
            color: #8892b0;
            line-height: 1.6;
            margin-bottom: 0.1rem;
        }
        .bio-card {
            padding: 0.85rem 0.95rem;
            border-radius: 12px;
            background: rgba(13, 15, 30, 0.80);
            border: 1px solid rgba(255,255,255,0.06);
            margin: 0.8rem 0;
        }
        .bio-card-title {
            font-weight: 800;
            color: #f0f2ff;
            font-size: 0.95rem;
            margin-bottom: 0.25rem;
        }
        .bio-small {
            font-size: 0.80rem;
            color: #8892b0;
            line-height: 1.55;
        }
        .bio-pill {
            display: inline-block;
            padding: 0.22rem 0.55rem;
            border-radius: 999px;
            background: rgba(13,148,136,0.12);
            border: 1px solid rgba(45,212,191,0.20);
            color: #2dd4bf;
            font-weight: 750;
            margin: 0.12rem 0.16rem 0.12rem 0;
            font-size: 0.74rem;
        }
        .bio-disclaimer {
            padding: 0.85rem 0.95rem;
            border-radius: 12px;
            background: rgba(180,83,9,0.10);
            border: 1px solid rgba(245,158,11,0.25);
            color: #fbbf24;
            font-weight: 800;
            line-height: 1.45;
            margin-top: 0.9rem;
            font-size: 0.80rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="bio-wrap presentation-card">
            <div class="bio-kicker">Explainable Genomic AI</div>
            <div class="bio-title">Biological Interpretation</div>
            <div class="bio-text">
                The risk score is not only a number. GenomicAI links the prediction to
                additive genotype markers coded as <b>0 / 1 / 2</b>, then summarizes the
                biological context through SNPs, genes, protein-network evidence, and immune pathways.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if prob is not None:
        st.markdown(
            f"""
            <div class="bio-card presentation-card">
                <div class="bio-card-title">Current sample context</div>
                <div class="bio-small">
                    For <b>{sample_name if sample_name else "this sample"}</b>, the model generated a
                    program-based genomic risk score of <b>{float(prob):.1%}</b>. The explanation below
                    connects important SNP patterns to immune-related biology.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="bio-card presentation-card">
                <div class="bio-card-title">Project-level interpretation</div>
                <div class="bio-small">
                    For uploaded batches, this section summarizes the biological interpretation layer of the project.
                    It does not assign one probability to the whole file; each row keeps its own risk score in the results table.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    c1, c2, c3 = st.columns(3)
    c1.metric("Prioritized SNPs", "32")
    c2.metric("Candidate Genes", "54")
    c3.metric("Network Nodes", "90")

    st.markdown("#### Top SNPs used in the interpretation layer")

    top_snps = pd.DataFrame(
        [
            ["rs660895", "0 / 1 / 2", "Dominant HLA-region signal; antigen presentation context"],
            ["rs6910071", "0 / 1 / 2", "Prioritized immune/genomic marker"],
            ["rs13192471", "0 / 1 / 2", "HLA-related adaptive immune context"],
            ["rs17533090", "0 / 1 / 2", "Prioritized RA-associated SNP signal"],
            ["rs1182531", "0 / 1 / 2", "Candidate locus contributing to explanation"],
        ],
        columns=["SNP", "Additive Encoding", "Biological Context"],
    )
    st.dataframe(top_snps, use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="bio-card presentation-card">
            <div class="bio-card-title">SHAP explanation</div>
            <div class="bio-small">
                SHAP highlights which SNP features contributed most to the model output.
                It explains model behavior, not medical diagnosis and not biological causality.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    shap_img = _first_existing_asset([
        "XGB_Additive_SHAP_Top20.png",
        "XGB_Additive_SHAP_beeswarm_Top20.png",
    ])
    string_img = _first_existing_asset(["string_network.png"])

    img_col1, img_col2 = st.columns(2)
    with img_col1:
        if shap_img:
            st.image(shap_img, caption="Top SNPs by SHAP importance", use_container_width=True)
        else:
            st.info("SHAP figure will appear here when added to assets/.")

    with img_col2:
        if string_img:
            st.image(string_img, caption="Candidate gene/protein network", use_container_width=True)
        else:
            st.info("STRING network figure will appear here when added to assets/.")

    st.markdown("#### Prioritized SNPs → Candidate Genes → Proteins/STRING Network → Immune Pathways → RA Risk Interpretation")
    st.plotly_chart(build_biological_sankey(), use_container_width=True)

    st.markdown(
        """
        <div class="bio-card presentation-card">
            <div class="bio-card-title">Biological meaning</div>
            <div class="bio-small">
                The interpretation layer points to immune-related biology, including:
                <br><br>
                <span class="bio-pill">antigen processing and presentation</span>
                <span class="bio-pill">T-cell receptor signaling</span>
                <span class="bio-pill">interferon-gamma pathway</span>
                <span class="bio-pill">immune-related biology</span>
                <br><br>
                This supports a pitch-friendly message: the model produces a genomic risk estimate,
                then connects important additive SNP patterns to biologically meaningful immune mechanisms.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="bio-disclaimer presentation-card">
            This is a research and educational prototype, not a medical diagnosis.
        </div>
        """,
        unsafe_allow_html=True,
    )

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

    render_disclaimer()

    st.markdown("""
    <div class="simple-intro presentation-card">
      <div class="simple-kicker">The idea in simple words</div>
      <p>
        Rheumatoid arthritis can be discovered late, and delays can lead to joint damage.
        This project uses small genetic markers to estimate whether a person's genetic
        pattern looks closer to higher-risk or lower-risk examples learned by the program.
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

                # Biological Interpretation appears directly after the risk gauge for the pitch video.
                render_biological_interpretation(
                    prob=prob,
                    sample_name=selected_label,
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

                        # For one uploaded sample: show current sample context.
                        # For multiple samples: show project-level interpretation only.
                        if len(res) == 1:
                            render_biological_interpretation(
                                prob=float(probs[0]),
                                sample_name=str(res["Sample"].iloc[0]),
                            )
                        else:
                            render_biological_interpretation(
                                prob=None,
                                sample_name="uploaded batch",
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
