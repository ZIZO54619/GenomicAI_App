import streamlit as st
import pandas as pd
import joblib, os

CLASSIFICATION_THRESHOLD = 0.50
DISCLAIMER_TEXT = (
    "This application is a research prototype for educational and academic "
    "demonstration purposes. It provides a model-based genomic risk estimate "
    "and does not replace physician evaluation, laboratory testing, imaging, "
    "or clinical diagnosis."
)
LABEL_COLUMNS = ["label", "ExpectedLabel"]

st.set_page_config(
    page_title="GenomicAI — RA Genomic Risk Prediction",
    page_icon="🧬",
    layout="centered",          # CENTERED — fixes the extremes issue
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────
def load_css():
    p = os.path.join(os.path.dirname(__file__), "assets", "style.css")
    if os.path.exists(p):
        with open(p) as f:
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
        st.error(f"Missing required SNP features: {len(missing)}")
        st.write(missing[:20])
        st.stop()

    if extra:
        st.warning(f"Ignoring extra columns: {len(extra)}")

    X = df.reindex(columns=required_cols)
    values = pd.unique(X.values.ravel())
    invalid = [v for v in values if v not in [0, 1, 2]]

    if invalid:
        st.error(f"Invalid genotype values detected: {invalid}. Expected only 0, 1, or 2.")
        st.stop()

    return X

def predict(artifact, df):
    m, cols = artifact["model"], get_required_feature_columns(artifact)
    probs = m.predict_proba(df.reindex(columns=cols))[:, 1]
    return (probs >= CLASSIFICATION_THRESHOLD).astype(int), probs

def render_disclaimer():
    st.markdown(f"""
    <div class="disclaimer-card">
      <div class="dc-kicker">Research prototype notice</div>
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
  <div class="nb-tagline">AI · Genomics · Precision Medicine</div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    if st.button("🏠  Home", use_container_width=True,
                 type="primary" if st.session_state.page == "Home" else "secondary"):
        go("Home")
with c2:
    if st.button("🔬  Predict", use_container_width=True,
                 type="primary" if st.session_state.page == "Predict" else "secondary"):
        go("Predict")
with c3:
    if st.button("📄  About", use_container_width=True,
                 type="primary" if st.session_state.page == "About" else "secondary"):
        go("About")

st.markdown('<div class="nb-line"></div>', unsafe_allow_html=True)

page = st.session_state.page

# ══════════════════════════════════════════════════════════════
#  HOME
# ══════════════════════════════════════════════════════════════
if page == "Home":

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
        <h1 class="pred-title">Prediction Center</h1>
        <p class="pred-sub">Run SNP-based RA genomic risk estimation using the trained XGBoost model</p>
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
              <div class="ms-title">XGBoost Model · Loaded</div>
              <div class="ms-sub">212-feature additive encoder · Demo threshold {CLASSIFICATION_THRESHOLD:.2f}</div>
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
              <div class="ms-title">Model Not Found</div>
              <div class="ms-sub">Place xgboost_model.joblib in models/ directory</div>
            </div>
          </div>
          <div class="ms-badge ms-badge-warn">OFFLINE</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Mode selector
    st.markdown('<div class="mode-label">SELECT PREDICTION MODE</div>', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["📋  Demo — Predefined Samples", "📤  Upload — Your CSV File"])

    # ── TAB 1 ─────────────────────────────────────────────────
    with tab1:
        st.markdown("""
        <div class="tab-desc">
          Choose a sample from the NARAC dataset to explore the risk-estimation pipeline
          without uploading any file. Great for demos and presentations.
        </div>
        """, unsafe_allow_html=True)

        demo_path = os.path.join(os.path.dirname(__file__), "data", "demo_samples.csv")
        if os.path.exists(demo_path):
            demo_df = pd.read_csv(demo_path, index_col=0)

            col_s, col_b = st.columns([3, 1])
            with col_s:
                selected = st.selectbox("", demo_df.index.tolist(),
                                        label_visibility="collapsed")
            with col_b:
                run = st.button("▶  Analyze", type="primary",
                                use_container_width=True, key="dr")

            if run and model_artifact:
                row   = demo_df.loc[[selected]]
                label_col = next((c for c in LABEL_COLUMNS if c in row.columns), None)
                exp   = row[label_col].values[0] if label_col else None
                required_cols = get_required_feature_columns(model_artifact)
                feat  = validate_features(row, required_cols)
                preds, probs = predict(model_artifact, feat)
                cls   = "Higher RA-like genomic risk pattern" if preds[0] == 1 else "Lower RA-like genomic risk pattern"
                prob  = float(probs[0])
                is_higher = preds[0] == 1

                # Big result card
                card_cls = "result-ra" if is_higher else "result-ctrl"
                icon     = "▲" if is_higher else "●"
                st.markdown(f"""
                <div class="result-card {card_cls}">
                  <div class="rc-left">
                    <div class="rc-status">{icon} {'Higher risk pattern' if is_higher else 'Lower risk pattern'}</div>
                    <div class="rc-class">{cls}</div>
                    <div class="rc-sample">Sample ID: {selected}</div>
                  </div>
                  <div class="rc-right">
                    <div class="rc-plabel">RA-like genomic risk probability</div>
                    <div class="rc-prob {'rp-ra' if is_higher else 'rp-ctrl'}">{prob:.1%}</div>
                    <div class="rc-sublabel">Model-based genomic risk score</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="threshold-note">
                  Classification threshold used in this demo: {CLASSIFICATION_THRESHOLD:.2f}.
                  The probability score should be interpreted as a model-based risk score,
                  not as a clinical diagnosis.
                </div>
                """, unsafe_allow_html=True)

                # Risk gauge bar
                bar_pct = int(prob * 100)
                bar_color = "#b45309" if is_higher else "#0d9488"
                st.markdown(f"""
                <div class="gauge-wrap">
                  <div class="gauge-label">
                    <span>Model-based risk score</span>
                    <span style="color:{bar_color};font-weight:700">{prob:.4f}</span>
                  </div>
                  <div class="gauge-track">
                    <div class="gauge-fill" style="width:{bar_pct}%;background:{bar_color};">
                      <div class="gauge-shine"></div>
                    </div>
                  </div>
                  <div class="gauge-marks">
                    <span>Low</span><span>Medium</span><span>High</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                if exp is not None:
                    ok = is_higher == (str(exp).upper() in ["1", "RA", "CASE"])
                    st.markdown(f"""
                    <div class="verify-row">
                      <span class="vr-badge {'vr-ok' if ok else 'vr-err'}">
                        {'Reference label matched' if ok else 'Reference label mismatch'}
                      </span>
                      <span class="vr-exp">Expected label: <strong>{exp}</strong></span>
                    </div>
                    """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="empty-state">
              <div class="es-icon">📂</div>
              <div class="es-title">No Demo Samples Found</div>
              <div class="es-body">Add <code>demo_samples.csv</code> to the <code>data/</code> folder.<br>
              Columns = SNP features (0/1/2), optional <code>label</code> column.</div>
            </div>
            """, unsafe_allow_html=True)

    # ── TAB 2 ─────────────────────────────────────────────────
    with tab2:
        st.markdown("""
        <div class="tab-desc">
          Upload an additive-encoded CSV file. Each row is one sample.
          Columns must match the 212 SNP features used during model training.
          An optional <code>label</code> column enables reference-label comparison.
        </div>
        """, unsafe_allow_html=True)

        f = st.file_uploader("", type=["csv"], label_visibility="collapsed")
        if f:
            try:
                df = pd.read_csv(f, index_col=0)
                st.markdown(f"""
                <div class="file-info">
                  <span class="fi-icon">📊</span>
                  <span><strong>{len(df)}</strong> sample(s) loaded · <strong>{df.shape[1]}</strong> columns detected</span>
                </div>
                """, unsafe_allow_html=True)

                if st.button("▶  Run Predictions on All Samples",
                             type="primary", key="ur"):
                    if model_artifact:
                        label_col = next((c for c in LABEL_COLUMNS if c in df.columns), None)
                        lc   = df[label_col] if label_col else None
                        required_cols = get_required_feature_columns(model_artifact)
                        feat = validate_features(df, required_cols)
                        preds, probs = predict(model_artifact, feat)

                        res = pd.DataFrame({
                            "Sample": feat.index,
                            "Risk Pattern": [
                                "Higher RA-like genomic risk pattern" if p == 1
                                else "Lower RA-like genomic risk pattern" for p in preds
                            ],
                            "RA-like Genomic Risk Probability": [round(float(p), 4) for p in probs],
                            "Risk Stratum": [
                                "Higher risk pattern" if p >= .7 else
                                "Intermediate risk pattern" if p >= .4 else
                                "Lower risk pattern" for p in probs
                            ],
                        })
                        if lc is not None:
                            res["Expected"] = lc.values

                        def style_r(r):
                            if r["Risk Pattern"] == "Higher RA-like genomic risk pattern":
                                return ["background:rgba(180,83,9,.10)"]*len(r)
                            return ["background:rgba(20,184,166,.08)"]*len(r)

                        st.markdown(f"""
                        <div class="threshold-note">
                          Classification threshold used in this demo: {CLASSIFICATION_THRESHOLD:.2f}.
                          The probability score should be interpreted as a model-based risk score,
                          not as a clinical diagnosis.
                        </div>
                        """, unsafe_allow_html=True)

                        st.dataframe(
                            res.style.apply(style_r, axis=1),
                            use_container_width=True,
                            height=min(480, 60 + len(res)*38),
                        )
            except Exception as e:
                st.error(f"Error reading file: {e}")


# ══════════════════════════════════════════════════════════════
#  ABOUT
# ══════════════════════════════════════════════════════════════
elif page == "About":

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
