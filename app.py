import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="GenomicAI — RA Prediction",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "models", "xgboost_model.joblib")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

def predict_sample(model_artifact, df_input):
    model = model_artifact["model"]
    feature_cols = model_artifact["feature_columns"]
    df_aligned = df_input.reindex(columns=feature_cols, fill_value=0)
    probs = model.predict_proba(df_aligned)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return preds, probs

if "page" not in st.session_state:
    st.session_state.page = "Home"

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
        <div class="sb-dna">🧬</div>
        <div>
            <div class="sb-name">GenomicAI</div>
            <div class="sb-tagline">Precision · Genomics · AI</div>
        </div>
    </div>
    <div class="sb-divider"></div>
    <div class="sb-nav-label">NAVIGATION</div>
    """, unsafe_allow_html=True)

    for p_name, (p_icon, p_desc) in {
        "Home": ("🏠", "Overview & Pipeline"),
        "Prediction Center": ("🔬", "Run RA Predictions"),
        "About": ("📄", "Project & Research"),
    }.items():
        is_active = st.session_state.page == p_name
        btn_type = "primary" if is_active else "secondary"
        if st.button(f"{p_icon}  {p_name}", key=f"nav_{p_name}", use_container_width=True, help=p_desc, type=btn_type):
            st.session_state.page = p_name
            st.rerun()

    st.markdown('<div class="sb-divider" style="margin-top:8px;"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-section-label">DATASET</div>
    <div class="sb-stats">
        <div class="sb-stat"><span class="sb-slab">Cohort</span><span class="sb-sval">NARAC</span></div>
        <div class="sb-stat"><span class="sb-slab">Individuals</span><span class="sb-sval">2,062</span></div>
        <div class="sb-stat"><span class="sb-slab">RA Cases</span><span class="sb-sval">868</span></div>
        <div class="sb-stat"><span class="sb-slab">Controls</span><span class="sb-sval">1,194</span></div>
    </div>
    <div class="sb-divider"></div>
    <div class="sb-section-label">PIPELINE</div>
    <div class="sb-stats">
        <div class="sb-stat"><span class="sb-slab">Raw SNPs</span><span class="sb-sval">531,689</span></div>
        <div class="sb-stat"><span class="sb-slab">RA Markers</span><span class="sb-sval">313</span></div>
        <div class="sb-stat"><span class="sb-slab">Features</span><span class="sb-sval">212</span></div>
        <div class="sb-stat"><span class="sb-slab">Classifier</span><span class="sb-sval">XGBoost</span></div>
    </div>
    <div class="sb-divider"></div>
    <div class="sb-badge">Research Prototype · Not for Clinical Use</div>
    """, unsafe_allow_html=True)

page = st.session_state.page

# ══════════════════════ HOME ══════════════════════════════════════════════════
if page == "Home":

    st.markdown("""
    <div class="hero">
        <div class="hero-chip"><span class="hero-chip-dot"></span>AI-Powered Genomic Medicine</div>
        <h1 class="hero-h1">Predict Rheumatoid<br>Arthritis Risk from<br><span class="hero-accent">Genomic SNP Data</span></h1>
        <p class="hero-p">
            GenomicAI transforms raw genotype files into an actionable AI risk score —
            combining bioinformatics QC, RA-specific SNP selection, Beagle imputation,
            and an XGBoost classifier into one deployable pipeline.
        </p>
        <div class="hero-tags">
            <span class="tag">531,689 Raw SNPs</span>
            <span class="tag-sep">→</span>
            <span class="tag">313 RA Markers</span>
            <span class="tag-sep">→</span>
            <span class="tag">212 Features</span>
            <span class="tag-sep">→</span>
            <span class="tag tag-teal">XGBoost Prediction</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # CTA cards
    st.markdown("""
    <div class="cta-section">
        <div class="cta-heading">Explore the Application</div>
        <div class="cta-row">
            <div class="cta-card">
                <div class="cta-icon">🔬</div>
                <div class="cta-title">Prediction Center</div>
                <div class="cta-desc">Upload an additive-encoded SNP file or pick a demo sample and run the RA risk model instantly. Results include predicted class, RA probability score, and optional label comparison.</div>
                <div class="cta-hint">← Use the sidebar to navigate there</div>
            </div>
            <div class="cta-card cta-card-teal">
                <div class="cta-icon">📄</div>
                <div class="cta-title">About the Project</div>
                <div class="cta-desc">Explore the research rationale, NARAC dataset details, six-stage pipeline breakdown, technical strengths, limitations, and future directions for GenomicAI.</div>
                <div class="cta-hint">← Use the sidebar to navigate there</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    cola, colb, colc = st.columns([1,1,2])
    with cola:
        if st.button("🔬  Go to Prediction Center", use_container_width=True, type="primary"):
            st.session_state.page = "Prediction Center"; st.rerun()
    with colb:
        if st.button("📄  Read About the Project", use_container_width=True):
            st.session_state.page = "About"; st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # KPIs
    st.markdown('<div class="sec-label">BY THE NUMBERS</div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    for col,(val,name,sub,cls) in zip([c1,c2,c3,c4],[
        ("2,062","Individuals","NARAC genomic cohort","kpi-purple"),
        ("531,689","Raw SNPs","22 autosomal chromosomes","kpi-default"),
        ("313","RA Markers","Knowledge-driven selection","kpi-teal"),
        ("212","Final Features","Additive 0/1/2 encoding","kpi-purple"),
    ]):
        with col:
            st.markdown(f'<div class="kpi-card {cls}"><div class="kpi-val">{val}</div><div class="kpi-name">{name}</div><div class="kpi-sub">{sub}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Pipeline
    st.markdown('<div class="sec-label">PROCESSING PIPELINE</div>', unsafe_allow_html=True)
    st.markdown('<p class="sec-sub">Six structured stages transforming raw chromosomal files into a deployable prediction system</p>', unsafe_allow_html=True)

    steps = [
        ("01","Data Description","Chromosome-wise PED/MAP inspection. SNP counts per chromosome, missing data profiling, sample layout verification.","🔍","#7c3aed"),
        ("02","Discrepancy Detection","QC checks: PED/MAP consistency, duplicate IDs, phenotype/sex coding, genotype token validity, formatting anomalies.","🛡️","#6d28d9"),
        ("03","SNP Refinement","Intersect external RA-associated SNP list with NARAC data. Knowledge-driven reduction: 531,689 → 313 biologically validated markers.","🎯","#0d9488"),
        ("04","Imputation","PED/MAP → PLINK binary → VCF → Beagle imputation → back-conversion. Missing-only overlay preserves all original observed calls.","🔄","#0f766e"),
        ("05","Encoding","Token standardization, removal of monomorphic/non-biallelic SNPs, additive 0/1/2 encoding. 313 → 212 ML-ready features.","⚙️","#7c3aed"),
        ("06","XGBoost Modeling","Repeated nested cross-validation, hyperparameter tuning, threshold-aware evaluation, SHAP explainability analysis.","🤖","#0d9488"),
    ]
    c1,c2,c3 = st.columns(3)
    for i,(num,title,desc,icon,color) in enumerate(steps):
        with [c1,c2,c3][i%3]:
            st.markdown(f'<div class="pipe-card" style="--acc:{color};"><div class="pipe-top"><span class="pipe-num">{num}</span><span class="pipe-icon">{icon}</span></div><div class="pipe-title">{title}</div><div class="pipe-desc">{desc}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Flow
    st.markdown('<div class="sec-label">DATA FLOW</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="flow-wrap">
        <div class="flow-node fn-start"><div class="fn-icon">📁</div><div class="fn-title">Raw PED/MAP</div><div class="fn-sub">531,689 SNPs · 2,062 samples</div></div>
        <div class="flow-arr">→</div>
        <div class="flow-node"><div class="fn-icon">🛡️</div><div class="fn-title">QC & Validation</div><div class="fn-sub">Structural integrity</div></div>
        <div class="flow-arr">→</div>
        <div class="flow-node fn-teal"><div class="fn-icon">🎯</div><div class="fn-title">RA SNP Panel</div><div class="fn-sub">313 RA markers</div></div>
        <div class="flow-arr">→</div>
        <div class="flow-node"><div class="fn-icon">🔄</div><div class="fn-title">Imputation</div><div class="fn-sub">Beagle · Missing-only</div></div>
        <div class="flow-arr">→</div>
        <div class="flow-node"><div class="fn-icon">⚙️</div><div class="fn-title">Encoding</div><div class="fn-sub">212 features</div></div>
        <div class="flow-arr">→</div>
        <div class="flow-node fn-end"><div class="fn-icon">🤖</div><div class="fn-title">XGBoost</div><div class="fn-sub">RA Risk Score</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tech stack
    st.markdown('<div class="sec-label">TECHNOLOGY STACK</div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    for col,(grp,icon,items) in zip([c1,c2,c3],[
        ("Bioinformatics","🧬",[("PLINK","Binary genotype conversion"),("Beagle","Genotype imputation"),("VCF / PED-MAP","Standard genomic formats"),("22 Autosomes","Sex-chr excluded")]),
        ("Machine Learning","🤖",[("XGBoost","Final deployed classifier"),("Scikit-learn","Preprocessing & nested CV"),("SHAP","Feature explainability"),("Additive 0/1/2","Compact genotype encoding")]),
        ("Deployment","🚀",[("Streamlit","Interactive web interface"),("Joblib","Model artifact serialization"),("Pandas / NumPy","Data alignment & handling"),("Python 3.10+","Core runtime")]),
    ]):
        with col:
            rows="".join(f'<div class="tech-row"><span class="tech-key">{k}</span><span class="tech-val">{v}</span></div>' for k,v in items)
            st.markdown(f'<div class="tech-card"><div class="tech-head"><span>{icon}</span>{grp}</div>{rows}</div>', unsafe_allow_html=True)

# ══════════════════════ PREDICTION CENTER ════════════════════════════════════
elif page == "Prediction Center":

    st.markdown("""
    <div class="page-header">
        <div class="ph-icon">🔬</div>
        <div><h1 class="page-title">Prediction Center</h1>
        <p class="page-sub">Run RA risk prediction on additive-encoded SNP profiles using the trained XGBoost model</p></div>
    </div>
    """, unsafe_allow_html=True)

    model_artifact = load_model()
    if model_artifact is None:
        st.markdown('<div class="alert alert-warn">⚠️ <strong>Model artifact not found.</strong> Place <code>xgboost_model.joblib</code> in the <code>models/</code> directory.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert alert-ok">✅ XGBoost model loaded — ready for inference</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["📋  Demo Samples", "📤  Upload Your CSV"])

    with tab1:
        st.markdown('<div class="tab-info">Select a predefined NARAC sample to test the prediction pipeline without uploading any file.</div>', unsafe_allow_html=True)
        demo_path = os.path.join(os.path.dirname(__file__), "data", "demo_samples.csv")
        if os.path.exists(demo_path):
            demo_df = pd.read_csv(demo_path, index_col=0)
            col_sel, col_btn = st.columns([3,1])
            with col_sel:
                selected = st.selectbox("Sample", demo_df.index.tolist(), label_visibility="collapsed")
            with col_btn:
                run = st.button("▶  Run", key="demo_run", type="primary", use_container_width=True)
            if run:
                row = demo_df.loc[[selected]]
                expected = row["label"].values[0] if "label" in row.columns else None
                features = row.drop(columns=["label"]) if "label" in row.columns else row
                if model_artifact:
                    preds, probs = predict_sample(model_artifact, features)
                    pred_class = "RA" if preds[0]==1 else "Control"
                    prob_val = float(probs[0])
                    is_ra = pred_class == "RA"
                    st.markdown(f"""
                    <div class="result-banner {'result-ra' if is_ra else 'result-ctrl'}">
                        <div class="res-left">
                            <div class="res-chip">{'⚠ RA Risk Detected' if is_ra else '✓ Control — Low Risk'}</div>
                            <div class="res-main">{pred_class}</div>
                            <div class="res-sample">Sample: {selected}</div>
                        </div>
                        <div class="res-right">
                            <div class="res-prob-label">RA Probability</div>
                            <div class="res-prob">{prob_val:.1%}</div>
                            <div class="res-prob-sub">Model confidence score</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(prob_val, text=f"Risk Score: {prob_val:.4f}")
                    if expected is not None:
                        correct = is_ra == (str(expected).upper() in ["1","RA","CASE"])
                        st.markdown(f'<div class="match-row"><span class="match-badge {"match-ok" if correct else "match-err"}">{"✅ Correct" if correct else "❌ Mismatch"}</span><span class="match-exp">Expected: <strong>{expected}</strong></span></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert alert-info">📂 Add <code>demo_samples.csv</code> to the <code>data/</code> folder to enable demo mode.</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="tab-info">Upload a CSV file — rows = samples, columns = additive SNP features (0/1/2). Optional <code>label</code> column for comparison.</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
        if uploaded:
            try:
                df_up = pd.read_csv(uploaded, index_col=0)
                st.markdown(f'<div class="alert alert-info">📊 <strong>{len(df_up)} sample(s)</strong> loaded · <strong>{df_up.shape[1]} columns</strong></div>', unsafe_allow_html=True)
                if st.button("▶  Run Predictions", key="up_run", type="primary"):
                    if model_artifact:
                        label_col = df_up["label"] if "label" in df_up.columns else None
                        df_feat = df_up.drop(columns=["label"]) if "label" in df_up.columns else df_up
                        preds, probs = predict_sample(model_artifact, df_feat)
                        results = pd.DataFrame({
                            "Sample": df_feat.index,
                            "Predicted Class": ["RA" if p==1 else "Control" for p in preds],
                            "RA Probability": [round(float(p),4) for p in probs],
                            "Risk Level": ["High" if p>=0.7 else "Medium" if p>=0.4 else "Low" for p in probs],
                        })
                        if label_col is not None:
                            results["Expected"] = label_col.values
                        def style_row(r):
                            return ["background:rgba(139,92,246,0.12)"]*len(r) if r["Predicted Class"]=="RA" else ["background:rgba(20,184,166,0.08)"]*len(r)
                        st.dataframe(results.style.apply(style_row,axis=1), use_container_width=True, height=min(450,60+len(results)*38))
            except Exception as e:
                st.error(f"Error: {e}")

# ══════════════════════ ABOUT ═════════════════════════════════════════════════
elif page == "About":

    st.markdown("""
    <div class="page-header">
        <div class="ph-icon">📄</div>
        <div><h1 class="page-title">About GenomicAI</h1>
        <p class="page-sub">Research rationale, dataset details, technical strengths, and future directions</p></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="vision-banner">
        <div class="vision-quote">"Transforming raw chromosomal genotype data into actionable AI-driven risk predictions
        for early Rheumatoid Arthritis detection — bridging bioinformatics, machine learning, and precision medicine."</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="about-card">
            <div class="about-card-h">🎯 The Clinical Problem</div>
            <p>Rheumatoid Arthritis affects millions globally. Early diagnosis is hindered by non-specific symptoms,
            clinical overlap with other autoimmune conditions, and the absence of scalable genomic screening workflows.</p>
            <p>Traditional diagnosis depends on clinical signs and lab markers that may not be informative at the earliest
            disease stage — when intervention is most impactful.</p>
            <p>GenomicAI addresses the core technical challenge: <em>how to convert high-dimensional raw SNP data
            into a reliable, biologically grounded AI prediction system.</em></p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="about-card">
            <div class="about-card-h">💡 The Innovation</div>
            <p>Rather than naive dimensionality reduction, GenomicAI applies <strong>knowledge-driven SNP selection</strong> —
            intersecting the NARAC dataset with externally validated RA-associated markers to build a biologically
            meaningful 212-feature panel.</p>
            <p>Combined with Beagle imputation and a missing-only overlay strategy, the pipeline maintains the integrity
            of original observed genotype calls throughout.</p>
            <p>The result: a complete research-to-deployment workflow from raw chromosomal files to a working prototype.</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="sec-label">NARAC COHORT</div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    for col,(v,l,s) in zip([c1,c2,c3,c4],[("2,062","Total Individuals","Full NARAC cohort"),("868","RA Cases","Confirmed disease"),("1,194","Controls","Healthy individuals"),("22","Chromosomes","Autosomal only")]):
        with col:
            st.markdown(f'<div class="kpi-card kpi-sm"><div class="kpi-val">{v}</div><div class="kpi-name">{l}</div><div class="kpi-sub">{s}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="about-card">
            <div class="about-card-h">✅ Technical Strengths</div>
            <div class="feat-list">
                <div class="feat-item fi-ok">End-to-end pipeline: raw genotype files → deployed web prototype</div>
                <div class="feat-item fi-ok">Biologically informed SNP selection (313 RA-validated markers)</div>
                <div class="feat-item fi-ok">Robust preprocessing: QC validation, Beagle imputation, additive encoding</div>
                <div class="feat-item fi-ok">Reproducible ML workflow with repeated nested cross-validation</div>
                <div class="feat-item fi-ok">Model artifact stores feature order — ensures training/inference alignment</div>
            </div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="about-card">
            <div class="about-card-h">⚠️ Current Limitations</div>
            <div class="feat-list">
                <div class="feat-item fi-warn">Research prototype — not a clinically certified diagnostic system</div>
                <div class="feat-item fi-warn">Genomic features only — no clinical symptoms, imaging, or lab values</div>
                <div class="feat-item fi-warn">Single cohort — requires external validation before broader use</div>
                <div class="feat-item fi-warn">Production deployment would need security hardening & governance</div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-label">FUTURE DIRECTIONS</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    for col,(ic,title,desc) in zip(cols,[("🔗","Multi-modal Integration","Combine genomic + clinical + imaging data in a unified prediction model"),("🔍","In-app Explainability","SHAP waterfall plots and feature importance directly in the interface"),("🏥","Institutional Deployment","Authenticated, secure platform for hospital genomics research teams"),("🌍","Cohort Expansion","Validation across diverse genomic populations and RA subtypes")]):
        with col:
            st.markdown(f'<div class="future-card"><div class="fut-ic">{ic}</div><div class="fut-title">{title}</div><div class="fut-desc">{desc}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="disclaimer">
        <strong>⚕️ Research & Clinical Disclaimer</strong><br>
        GenomicAI is a research demonstration prototype. Predictions are not intended to replace clinical diagnosis,
        medical advice, or professional healthcare judgment. The model is based on a single genomic cohort and has not
        undergone clinical validation. All outputs are model-generated risk scores within the current experimental context only.
    </div>""", unsafe_allow_html=True)
