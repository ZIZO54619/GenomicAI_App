# GenomicAI — RA Genomic Machine-Learning Research Dashboard

A professional Streamlit dashboard aligned to the final GenomicAI graduation book.

## What the application contains

- **Overview:** end-to-end analytical workflow and final project snapshot.
- **Prediction:** strict 212-SNP additive-input validation, frozen XGBoost prediction, downloadable template, batch output, and local XGBoost SHAP contributions.
- **Model Validation:** complete 16-combination benchmark, interactive ROC-AUC/PR-AUC model map, repeated nested cross-validation + Optuna explanation, ranking table, and final model card.
- **Explainable AI:** global SHAP ranking, K-selection analysis, final 22 stable SNPs, genomic distribution, and top K25 SHAP interactions.
- **Biological Interpretation:** direct GWAS mapping, HLA/non-HLA gene sets, enrichment counts, representative terms, and final biological interpretation.
- **About:** team, advisors, scope, limitations, technology stack, and responsible-use statement.

## Scientific scope

The deployed model is the final **XGBoost Additive** classifier using **212 additive SNP features**. Reported performance is internal to the NARAC cohort:

- ROC-AUC: **0.9220 ± 0.0142**
- PR-AUC: **0.8914 ± 0.0203**
- F1: **0.8161**
- MCC: **0.6765**

The final interpretation layer uses **K = 25**, an outer-fold selection-frequency threshold of **0.80**, and **22 stable SNPs**. The focused interaction screen contains 231 possible pairs; the dashboard displays the top 15. Direct GWAS Catalog mapping yielded 29 genes: 5 HLA and 24 non-HLA.

## Responsible-use boundary

This is a **research prototype**, not a medical device. Its outputs must not be used for clinical diagnosis, treatment selection, or validated individual genetic-risk communication. SHAP values, interaction magnitudes, mapped genes, and enriched terms describe model behavior and annotation context; they do not prove causality.

## Local run

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud deployment

1. Push this repository to GitHub.
2. In Streamlit Community Cloud, select the repository and set the main file to `app.py`.
3. The included `runtime.txt` requests Python 3.12.
4. No secrets are required.

## Input CSV format

- One row per sample.
- All 212 required rsID columns must be present.
- Additive genotype values must be `0`, `1`, or `2`.
- Optional identifier columns: `sample_id`, `sample`, or `id`.
- Optional expected-label columns: `label`, `ExpectedLabel`, `phenotype`, or `target`.
- Download a valid template directly from the Prediction page.

## Model files

- `models/xgboost_model.json`: native XGBoost model used by default.
- `models/model_metadata.json`: ordered 212-feature list and deployment metadata.
- `models/xgboost_model.joblib`: retained as a backward-compatible archive/fallback.

## Data provenance

Dashboard tables are aligned to the final graduation book. See `data/DATA_PROVENANCE.md` for the K25 metadata note and the relationship between the final-book tables and the original K25 provenance package retained under `data/source_k25/`.
