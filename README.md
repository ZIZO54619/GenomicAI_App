# GenomicAI: Rheumatoid Arthritis Genomic Risk Demo

Streamlit research demo for SNP-based rheumatoid arthritis risk prediction using a pre-trained XGBoost model and additive-encoded genomic features.

## Why this project matters

Rheumatoid arthritis (RA) can cause progressive and irreversible joint damage if detected late. This project presents a genomics-focused machine learning demo that converts selected SNP features into a research-grade RA risk score through an interactive web interface.

This repository should be interpreted as a **research and portfolio demo**, not as a clinical diagnostic system.

## Dataset / data source

The app is built around a genotype-based RA prediction workflow referenced in the project as a NARAC-derived cohort.

Public repository scope:
- Streamlit inference app
- model-loading interface
- optional demo inputs
- dependency file for local setup

Not redistributed here:
- raw genotype files
- patient-level genomic source data
- restricted clinical metadata
- full training pipeline documentation

## Methods / workflow

The current app reflects the following high-level workflow:

1. Start from genome-wide genotype data.
2. Filter or intersect with RA-relevant SNP markers.
3. Encode selected variants into additive features (0/1/2).
4. Align user input to the trained feature schema.
5. Run a pre-trained XGBoost classifier.
6. Return an RA probability score and class prediction.

The app content references the following reduction path:
- 531,689 raw SNPs
- 313 RA-associated markers after refinement
- 212 final features used for model inference

## Repository structure

```text
GenomicAI_App/
├─ app.py                    # Main Streamlit app
├─ requirements.txt          # Python dependencies
├─ assets/
│  └─ style.css             # App styling
├─ data/
│  └─ demo_samples.csv      # Example inputs for demo mode
├─ models/
│  └─ xgboost_model.joblib  # Expected trained model artifact
└─ README.md
