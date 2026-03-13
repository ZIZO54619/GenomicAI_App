import streamlit as st
import pandas as pd
import joblib, os

st.set_page_config(
    page_title="GenomicAI — Rheumatoid Arthritis Prediction",
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

def predict(artifact, df):
    m, cols = artifact["model"], artifact["feature_columns"]
    probs = m.predict_proba(df.reindex(columns=cols, fill_value=0))[:, 1]
    return (probs >= 0.5).astype(int), probs

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
        AI-Powered Early Detection &nbsp;·&nbsp; Precision Genomics &nbsp;·&nbsp; Clinical Research
      </div>

      <h1 class="hero-title">
        Detect Rheumatoid Arthritis<br>
        Risk Before It Strikes
      </h1>

      <p class="hero-sub">
        GenomicAI transforms raw chromosomal genotype files into an
        interpretable AI risk score — catching high-risk individuals
        <em>before</em> irreversible joint damage occurs.
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

    # ── Why it matters ────────────────────────────────────────
    st.markdown("""
    <div class="why-grid">
      <div class="why-card wc-animate" style="animation-delay:0s">
        <div class="why-icon">⏱️</div>
        <div class="why-title">Catch It Early</div>
        <div class="why-body">RA damages joints silently. A genomic risk score flags at-risk individuals years before clinical symptoms — when intervention is most effective.</div>
      </div>
      <div class="why-card wc-animate" style="animation-delay:0.1s">
        <div class="why-icon">🧬</div>
        <div class="why-title">Biology-Driven AI</div>
        <div class="why-body">Not black-box statistics. Every feature is a validated RA-associated SNP marker — selected from peer-reviewed genomic literature, not a generic filter.</div>
      </div>
      <div class="why-card wc-animate" style="animation-delay:0.2s">
        <div class="why-icon">🌍</div>
        <div class="why-title">Built for Scale</div>
        <div class="why-body">Designed for populations with limited specialist access. In Egypt, RA onset averages 38.4 years — a working-age demographic that cannot wait for late diagnosis.</div>
      </div>
      <div class="why-card wc-animate" style="animation-delay:0.3s">
        <div class="why-icon">🔍</div>
        <div class="why-title">Explainable Output</div>
        <div class="why-body">SHAP explainability reveals <em>which SNPs</em> drove each prediction. Clinicians see the evidence, not just a number — building trust in AI-assisted diagnosis.</div>
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
        <div class="ps-body">Nested CV · hyperparameter tuning · SHAP feature explainability</div>
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
        <div class="tc-item"><span class="tc-k">SHAP</span><span class="tc-v">Feature explainability</span></div>
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
        <p class="pred-sub">Run RA genomic risk scoring on SNP profiles using the trained XGBoost model</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    model_artifact = load_model()

    # Model status card
    if model_artifact:
        st.markdown("""
        <div class="model-status ms-ok">
          <div class="ms-left">
            <div class="ms-dot ms-dot-ok"></div>
            <div>
              <div class="ms-title">XGBoost Model · Loaded</div>
              <div class="ms-sub">212-feature additive encoder · Binary classification</div>
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
          Choose a sample from the NARAC dataset to explore the prediction pipeline
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
                exp   = row["label"].values[0] if "label" in row.columns else None
                feat  = row.drop(columns=["label"]) if "label" in row.columns else row
                preds, probs = predict(model_artifact, feat)
                cls   = "RA" if preds[0] == 1 else "Control"
                prob  = float(probs[0])
                is_ra = cls == "RA"

                # Big result card
                card_cls = "result-ra" if is_ra else "result-ctrl"
                icon     = "⚠️" if is_ra else "✅"
                st.markdown(f"""
                <div class="result-card {card_cls}">
                  <div class="rc-left">
                    <div class="rc-status">{icon} {'RA Risk Detected' if is_ra else 'No Significant Risk'}</div>
                    <div class="rc-class">{cls}</div>
                    <div class="rc-sample">Sample ID: {selected}</div>
                  </div>
                  <div class="rc-right">
                    <div class="rc-plabel">RA Probability Score</div>
                    <div class="rc-prob {'rp-ra' if is_ra else 'rp-ctrl'}">{prob:.1%}</div>
                    <div class="rc-sublabel">XGBoost model confidence</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # Risk gauge bar
                bar_pct = int(prob * 100)
                bar_color = "#ef4444" if is_ra else "#14b8a6"
                st.markdown(f"""
                <div class="gauge-wrap">
                  <div class="gauge-label">
                    <span>Risk Level</span>
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
                    ok = is_ra == (str(exp).upper() in ["1", "RA", "CASE"])
                    st.markdown(f"""
                    <div class="verify-row">
                      <span class="vr-badge {'vr-ok' if ok else 'vr-err'}">
                        {'✅ Prediction Correct' if ok else '❌ Mismatch'}
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
          An optional <code>label</code> column enables accuracy comparison.
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
                        lc   = df["label"] if "label" in df.columns else None
                        feat = df.drop(columns=["label"]) if "label" in df.columns else df
                        preds, probs = predict(model_artifact, feat)

                        res = pd.DataFrame({
                            "Sample": feat.index,
                            "Prediction": ["RA" if p == 1 else "Control" for p in preds],
                            "RA Probability": [round(float(p), 4) for p in probs],
                            "Risk Level": [
                                "🔴 High" if p >= .7 else
                                "🟡 Medium" if p >= .4 else
                                "🟢 Low" for p in probs
                            ],
                        })
                        if lc is not None:
                            res["Expected"] = lc.values

                        def style_r(r):
                            if r["Prediction"] == "RA":
                                return ["background:rgba(239,68,68,.12)"]*len(r)
                            return ["background:rgba(20,184,166,.08)"]*len(r)

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
        "If deployed at scale, GenomicAI could shift RA management from
        reactive treatment to proactive prevention — giving clinicians
        a genomic early-warning system before a single joint is damaged."
      </blockquote>
    </div>
    """, unsafe_allow_html=True)

    # Impact grid
    st.markdown('<div class="sec-label">WHY THIS PROJECT MATTERS</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="impact-grid">
      <div class="ig-card ig-animate" style="animation-delay:0s">
        <div class="ig-icon">🩺</div>
        <div class="ig-title">Clinical Impact</div>
        <div class="ig-body">RA affects 1% of the global population with lifelong consequences.
        In Egypt, onset averages <strong>38.4 years</strong> — a working-age demographic.
        Early genomic prediction converts a destructive disease into a manageable one.</div>
      </div>
      <div class="ig-card ig-animate" style="animation-delay:0.1s">
        <div class="ig-icon">🌍</div>
        <div class="ig-title">Egyptian Healthcare Context</div>
        <div class="ig-body">Egypt's Universal Health Insurance expansion and Digital Egypt 2030
        create a direct institutional path for AI screening tools.
        With specialists concentrated in Cairo and Alexandria,
        a scalable genomic tool fills a critical geographic gap.</div>
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
        <div class="ig-body">SHAP explainability reveals which exact SNPs drove each prediction.
        Clinicians see evidence, not just a score — establishing trust
        in AI-assisted genomic diagnosis where it matters most.</div>
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
        <div class="sl-item ok-item">SHAP explainability — every prediction is interpretable</div>
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
      <div class="rm-step"><div class="rm-num">2</div><div class="rm-content"><div class="rm-title">Institutional Pilot</div><div class="rm-body">Deploy as a research decision-support tool in 2–3 tertiary rheumatology centers under IRB oversight.</div></div></div>
      <div class="rm-conn"></div>
      <div class="rm-step"><div class="rm-num">3</div><div class="rm-content"><div class="rm-title">Digital Egypt 2030 Integration</div><div class="rm-body">Align with UHI infrastructure and EHR interoperability standards for national health system embedding.</div></div></div>
      <div class="rm-conn"></div>
      <div class="rm-step"><div class="rm-num">4</div><div class="rm-content"><div class="rm-title">Regional Scale-Up</div><div class="rm-body">Expand to Upper Egypt and rural governorates where specialist density is lowest and early detection value is highest.</div></div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
      ⚕️ <strong>Research Disclaimer:</strong> GenomicAI is a research prototype.
      Outputs are model-generated risk scores, not clinical diagnoses.
      Any clinical deployment requires full regulatory review and validation.
    </div>
    """, unsafe_allow_html=True)
