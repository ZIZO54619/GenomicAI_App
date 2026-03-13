import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="GenomicAI — RA Prediction",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",   # collapsed by default — we use on-page nav
)

# ── CSS ───────────────────────────────────────────────────────────────────────
def load_css():
    p = os.path.join(os.path.dirname(__file__), "assets", "style.css")
    if os.path.exists(p):
        with open(p) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css()

# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    p = os.path.join(os.path.dirname(__file__), "models", "xgboost_model.joblib")
    return joblib.load(p) if os.path.exists(p) else None

def predict(artifact, df):
    m, cols = artifact["model"], artifact["feature_columns"]
    probs = m.predict_proba(df.reindex(columns=cols, fill_value=0))[:, 1]
    return (probs >= 0.5).astype(int), probs

# ── Session state ─────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "Home"

def go(p):
    st.session_state.page = p
    st.rerun()

# ══════════════════════════════════════════════════════════════
#  TOP NAV BAR  (visible on all pages, works on mobile)
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="topnav">
  <div class="topnav-brand">🧬 GenomicAI</div>
</div>
""", unsafe_allow_html=True)

n1, n2, n3, n4 = st.columns([2, 1, 1, 1])
with n1:
    st.markdown('<div style="height:1px"></div>', unsafe_allow_html=True)
with n2:
    if st.button("🏠 Home", use_container_width=True,
                 type="primary" if st.session_state.page == "Home" else "secondary"):
        go("Home")
with n3:
    if st.button("🔬 Predict", use_container_width=True,
                 type="primary" if st.session_state.page == "Prediction" else "secondary"):
        go("Prediction")
with n4:
    if st.button("📄 About", use_container_width=True,
                 type="primary" if st.session_state.page == "About" else "secondary"):
        go("About")

st.markdown('<div class="nav-divider"></div>', unsafe_allow_html=True)

page = st.session_state.page

# ══════════════════════════════════════════════════════════════
#  HOME
# ══════════════════════════════════════════════════════════════
if page == "Home":

    # ── Hero ──────────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
      <div class="hero-chip"><span class="dot"></span>AI · Genomics · Precision Medicine</div>
      <h1 class="hero-h1">Detect Rheumatoid<br>Arthritis Risk from<br>
        <span class="grad">Genomic SNP Data</span></h1>
      <p class="hero-p">
        GenomicAI converts raw chromosome files into a clinical-grade AI risk score —
        531,689 SNPs filtered to 212 RA-validated features, powering an XGBoost
        classifier that can flag high-risk individuals <em>before</em> symptoms appear.
      </p>
      <div class="chip-row">
        <span class="chip">531,689 Raw SNPs</span>
        <span class="chip-sep">→</span>
        <span class="chip">313 RA Markers</span>
        <span class="chip-sep">→</span>
        <span class="chip">212 Features</span>
        <span class="chip-sep">→</span>
        <span class="chip chip-teal">XGBoost Prediction</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Why it matters ────────────────────────────────────────
    st.markdown("""
    <div class="impact-bar">
      <div class="impact-item">
        <div class="impact-icon">⏱️</div>
        <div class="impact-text"><strong>Earlier diagnosis</strong><br>RA caught before joint damage is irreversible</div>
      </div>
      <div class="impact-item">
        <div class="impact-icon">💊</div>
        <div class="impact-text"><strong>Smarter treatment</strong><br>Risk stratification guides therapy decisions</div>
      </div>
      <div class="impact-item">
        <div class="impact-icon">🌍</div>
        <div class="impact-text"><strong>Scalable in Egypt</strong><br>Low-cost genomic screening for high-burden regions</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Numbers (responsive HTML grid — not st.columns) ───────
    st.markdown('<div class="sec-label">BY THE NUMBERS</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="kpi-grid">
      <div class="kpi kpi-p"><div class="kv">2,062</div><div class="kn">Individuals</div><div class="ks">NARAC cohort</div></div>
      <div class="kpi"><div class="kv" style="color:var(--tx2)">531K</div><div class="kn">Raw SNPs</div><div class="ks">22 autosomes</div></div>
      <div class="kpi kpi-t"><div class="kv">313</div><div class="kn">RA Markers</div><div class="ks">Knowledge-driven</div></div>
      <div class="kpi kpi-p"><div class="kv">212</div><div class="kn">Features</div><div class="ks">Additive 0/1/2</div></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Pipeline (HTML grid) ───────────────────────────────────
    st.markdown('<div class="sec-label">PIPELINE</div>', unsafe_allow_html=True)
    st.markdown('<p class="sec-sub">Six stages from raw genome to deployable prediction</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="pipe-grid">
      <div class="pc" style="--acc:#7c3aed"><div class="pt-row"><span class="pnum">01</span><span>🔍</span></div><div class="ptitle">Data Description</div><div class="pdesc">PED/MAP inspection · SNP counts · missing data profiling</div></div>
      <div class="pc" style="--acc:#6d28d9"><div class="pt-row"><span class="pnum">02</span><span>🛡️</span></div><div class="ptitle">QC Validation</div><div class="pdesc">Consistency checks · phenotype & token validity · anomaly detection</div></div>
      <div class="pc" style="--acc:#0d9488"><div class="pt-row"><span class="pnum">03</span><span>🎯</span></div><div class="ptitle">SNP Refinement</div><div class="pdesc">RA-SNP intersection · 531,689 → 313 biologically validated markers</div></div>
      <div class="pc" style="--acc:#0f766e"><div class="pt-row"><span class="pnum">04</span><span>🔄</span></div><div class="ptitle">Imputation</div><div class="pdesc">PLINK → VCF → Beagle · missing-only overlay preserves original calls</div></div>
      <div class="pc" style="--acc:#7c3aed"><div class="pt-row"><span class="pnum">05</span><span>⚙️</span></div><div class="ptitle">Encoding</div><div class="pdesc">Additive 0/1/2 · filter monomorphic SNPs · 313 → 212 features</div></div>
      <div class="pc" style="--acc:#0d9488"><div class="pt-row"><span class="pnum">06</span><span>🤖</span></div><div class="ptitle">XGBoost Model</div><div class="pdesc">Nested CV · hyperparameter tuning · SHAP explainability</div></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Flow (vertical on mobile via CSS) ─────────────────────
    st.markdown('<div class="sec-label">DATA FLOW</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="flow">
      <div class="fn fn-s"><div class="fi">📁</div><div class="fl">Raw PED/MAP</div><div class="fs">531,689 SNPs</div></div>
      <div class="fa">→</div>
      <div class="fn"><div class="fi">🛡️</div><div class="fl">QC Check</div><div class="fs">Validation</div></div>
      <div class="fa">→</div>
      <div class="fn fn-t"><div class="fi">🎯</div><div class="fl">RA Panel</div><div class="fs">313 markers</div></div>
      <div class="fa">→</div>
      <div class="fn"><div class="fi">🔄</div><div class="fl">Imputation</div><div class="fs">Beagle</div></div>
      <div class="fa">→</div>
      <div class="fn"><div class="fi">⚙️</div><div class="fl">Encoding</div><div class="fs">212 features</div></div>
      <div class="fa">→</div>
      <div class="fn fn-e"><div class="fi">🤖</div><div class="fl">XGBoost</div><div class="fs">RA Risk Score</div></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tech stack (HTML grid) ─────────────────────────────────
    st.markdown('<div class="sec-label">STACK</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="tech-grid">
      <div class="tc"><div class="th">🧬 Bioinformatics</div>
        <div class="tr"><span class="tk">PLINK</span><span class="tv">Binary genotype conversion</span></div>
        <div class="tr"><span class="tk">Beagle</span><span class="tv">Genotype imputation</span></div>
        <div class="tr"><span class="tk">VCF / PED-MAP</span><span class="tv">Standard genomic formats</span></div>
      </div>
      <div class="tc"><div class="th">🤖 ML</div>
        <div class="tr"><span class="tk">XGBoost</span><span class="tv">Final deployed classifier</span></div>
        <div class="tr"><span class="tk">Scikit-learn</span><span class="tv">Preprocessing & nested CV</span></div>
        <div class="tr"><span class="tk">SHAP</span><span class="tv">Feature explainability</span></div>
      </div>
      <div class="tc"><div class="th">🚀 Deployment</div>
        <div class="tr"><span class="tk">Streamlit</span><span class="tv">Interactive web interface</span></div>
        <div class="tr"><span class="tk">Joblib</span><span class="tv">Model serialization</span></div>
        <div class="tr"><span class="tk">Pandas / NumPy</span><span class="tv">Data handling</span></div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PREDICTION
# ══════════════════════════════════════════════════════════════
elif page == "Prediction":

    st.markdown("""
    <div class="page-hdr">
      <div class="ph-ic">🔬</div>
      <div><h1 class="page-title">Prediction Center</h1>
      <p class="page-sub">Run RA risk prediction on additive-encoded SNP profiles</p></div>
    </div>
    """, unsafe_allow_html=True)

    model_artifact = load_model()

    if model_artifact is None:
        st.markdown('<div class="alert aw">⚠️ Model not loaded — place <code>xgboost_model.joblib</code> in <code>models/</code></div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert ao">✅ XGBoost model ready for inference</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["📋  Demo Samples", "📤  Upload CSV"])

    with tab1:
        st.markdown('<div class="tinfo">Select a predefined NARAC sample to test the pipeline.</div>',
                    unsafe_allow_html=True)
        demo_path = os.path.join(os.path.dirname(__file__), "data", "demo_samples.csv")
        if os.path.exists(demo_path):
            demo_df = pd.read_csv(demo_path, index_col=0)
            selected = st.selectbox("Sample", demo_df.index.tolist(), label_visibility="collapsed")
            if st.button("▶  Run Prediction", type="primary", key="dr"):
                row = demo_df.loc[[selected]]
                expected = row["label"].values[0] if "label" in row.columns else None
                feat = row.drop(columns=["label"]) if "label" in row.columns else row
                if model_artifact:
                    preds, probs = predict(model_artifact, feat)
                    cls = "RA" if preds[0] == 1 else "Control"
                    prob = float(probs[0])
                    is_ra = cls == "RA"
                    st.markdown(f"""
                    <div class="rbanner {'r-ra' if is_ra else 'r-ctrl'}">
                      <div class="rl">
                        <div class="rchip">{'⚠ RA Risk Detected' if is_ra else '✓ Low Risk'}</div>
                        <div class="rmain">{cls}</div>
                        <div class="rsub">Sample: {selected}</div>
                      </div>
                      <div class="rr">
                        <div class="rplbl">RA Probability</div>
                        <div class="rpval">{prob:.1%}</div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(prob, text=f"Risk score: {prob:.4f}")
                    if expected is not None:
                        ok = is_ra == (str(expected).upper() in ["1","RA","CASE"])
                        st.markdown(f'<div class="mbadge {"mok" if ok else "merr"}">{"✅ Correct" if ok else "❌ Mismatch"} · Expected: {expected}</div>',
                                    unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert ai">📂 Add <code>demo_samples.csv</code> to <code>data/</code></div>',
                        unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="tinfo">Upload a CSV — rows = samples, columns = SNP features (0/1/2). Optional <code>label</code> column for comparison.</div>',
                    unsafe_allow_html=True)
        f = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
        if f:
            try:
                df = pd.read_csv(f, index_col=0)
                st.markdown(f'<div class="alert ai">📊 <strong>{len(df)} sample(s)</strong> · <strong>{df.shape[1]} columns</strong></div>',
                            unsafe_allow_html=True)
                if st.button("▶  Run Predictions", type="primary", key="ur"):
                    if model_artifact:
                        lc = df["label"] if "label" in df.columns else None
                        feat = df.drop(columns=["label"]) if "label" in df.columns else df
                        preds, probs = predict(model_artifact, feat)
                        res = pd.DataFrame({
                            "Sample": feat.index,
                            "Class": ["RA" if p == 1 else "Control" for p in preds],
                            "RA Probability": [round(float(p), 4) for p in probs],
                            "Risk": ["High" if p >= .7 else "Medium" if p >= .4 else "Low" for p in probs],
                        })
                        if lc is not None:
                            res["Expected"] = lc.values
                        def sr(r):
                            return ["background:rgba(139,92,246,.12)"]*len(r) if r["Class"]=="RA" \
                                   else ["background:rgba(20,184,166,.08)"]*len(r)
                        st.dataframe(res.style.apply(sr, axis=1), use_container_width=True,
                                     height=min(450, 60+len(res)*38))
            except Exception as e:
                st.error(f"Error: {e}")


# ══════════════════════════════════════════════════════════════
#  ABOUT
# ══════════════════════════════════════════════════════════════
elif page == "About":

    st.markdown("""
    <div class="page-hdr">
      <div class="ph-ic">📄</div>
      <div><h1 class="page-title">About GenomicAI</h1>
      <p class="page-sub">The research rationale, dataset, and real-world impact</p></div>
    </div>
    """, unsafe_allow_html=True)

    # Vision
    st.markdown("""
    <div class="vision">
      <div class="vq">
        "If deployed at scale, GenomicAI could shift RA management from
        <em>reactive treatment</em> to <em>proactive prevention</em> —
        giving clinicians a genomic early-warning system before a single joint is damaged."
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Impact section
    st.markdown('<div class="sec-label">WHY THIS PROJECT MATTERS</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="impact-grid">
      <div class="ig-card">
        <div class="ig-icon">🩺</div>
        <div class="ig-title">Clinical Impact</div>
        <div class="ig-body">RA affects 1% of the global population. In Egypt alone, a national ECR study found mean onset at <strong>38.4 years</strong> — a working-age demographic — with 73.7% RF-positive cases and moderate-to-high disease activity (DAS28 ≥ 4.4). Early prediction converts a progressive disease into a manageable one.</div>
      </div>
      <div class="ig-card">
        <div class="ig-icon">🌍</div>
        <div class="ig-title">Egyptian Context</div>
        <div class="ig-body">Egypt's Universal Health Insurance expansion and Digital Egypt 2030 strategy create a direct institutional path for genomic AI screening tools. With rheumatologist density concentrated in Cairo and Alexandria, a scalable digital risk tool addresses a critical geographic gap in specialist access across Upper Egypt and rural governorates.</div>
      </div>
      <div class="ig-card">
        <div class="ig-icon">🔬</div>
        <div class="ig-title">Scientific Contribution</div>
        <div class="ig-body">The pipeline's knowledge-driven SNP selection — reducing 531,689 raw markers to 313 RA-validated ones — is reproducible, biologically grounded, and extensible. It provides a framework applicable to other autoimmune diseases and other MENA-region cohorts, contributing to the global precision medicine evidence base.</div>
      </div>
      <div class="ig-card">
        <div class="ig-icon">💡</div>
        <div class="ig-title">Innovation Edge</div>
        <div class="ig-body">Unlike black-box genome-wide association studies, GenomicAI produces an <strong>interpretable</strong> output via SHAP explainability. Clinicians can see <em>which SNPs</em> drove the risk score. Combined with the missing-only imputation overlay that preserves original observed calls, the system maintains scientific integrity end-to-end.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Dataset
    st.markdown('<div class="sec-label">NARAC COHORT</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="kpi-grid">
      <div class="kpi kpi-p"><div class="kv">2,062</div><div class="kn">Individuals</div><div class="ks">Full cohort</div></div>
      <div class="kpi"><div class="kv" style="color:#f87171">868</div><div class="kn">RA Cases</div><div class="ks">42.1%</div></div>
      <div class="kpi kpi-t"><div class="kv">1,194</div><div class="kn">Controls</div><div class="ks">57.9%</div></div>
      <div class="kpi kpi-p"><div class="kv">22</div><div class="kn">Chromosomes</div><div class="ks">Autosomal</div></div>
    </div>
    """, unsafe_allow_html=True)

    # Strengths + Limits (HTML grid)
    st.markdown('<div class="sec-label">STRENGTHS & LIMITS</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sl-grid">
      <div class="sl-card">
        <div class="sl-h sl-ok">✅ Strengths</div>
        <div class="fi fi-ok">End-to-end: raw files → live web prototype</div>
        <div class="fi fi-ok">Biology-driven SNP selection — not statistical shortcutting</div>
        <div class="fi fi-ok">Missing-only imputation preserves observed genotype integrity</div>
        <div class="fi fi-ok">SHAP explainability — clinically interpretable output</div>
        <div class="fi fi-ok">Nested CV ensures unbiased generalization estimates</div>
      </div>
      <div class="sl-card">
        <div class="sl-h sl-warn">⚠️ Limitations</div>
        <div class="fi fi-warn">Research prototype — not clinically certified</div>
        <div class="fi fi-warn">Genomic features only — no imaging or lab values</div>
        <div class="fi fi-warn">Trained on North American cohort — needs Egyptian validation</div>
        <div class="fi fi-warn">Production healthcare use requires security & governance hardening</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Future
    st.markdown('<div class="sec-label">ROADMAP</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="road-grid">
      <div class="rcard"><div class="ric">🔗</div><div class="rt">Multi-modal</div><div class="rd">Genomic + clinical + imaging in one model</div></div>
      <div class="rcard"><div class="ric">🔍</div><div class="rt">In-app SHAP</div><div class="rd">Live feature importance plots per prediction</div></div>
      <div class="rcard"><div class="ric">🏥</div><div class="rt">Hospital Platform</div><div class="rd">Authenticated institutional deployment</div></div>
      <div class="rcard"><div class="ric">🌍</div><div class="rt">Egyptian Cohort</div><div class="rd">Retrain & validate on Arab genomic populations</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disc">
      ⚕️ <strong>Disclaimer:</strong> GenomicAI is a research prototype.
      Outputs are model-generated risk scores, not clinical diagnoses.
    </div>
    """, unsafe_allow_html=True)
