# GenomicAI: Rheumatoid Arthritis Genomic Risk Demo

Interactive Streamlit research demo for SNP-based rheumatoid arthritis risk prediction using a pre-trained XGBoost model.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Open%20App-2ea44f?style=for-the-badge)](https://genomicaiapp1.streamlit.app/)

<p align="center">
  <img src="https://github.com/user-attachments/assets/cd94c426-5f2b-4850-87ef-42a23794f6be" alt="GenomicAI overview banner showing the user workflow from SNP file upload to AI analysis, RA risk score, and user-friendly result" width="100%">
</p>
<img width="1376" height="768" alt="genomicai-overview" src="https://github.com/user-attachments/assets/cd94c426-5f2b-4850-87ef-42a23794f6be" />

## Why this project matters

Rheumatoid arthritis can cause progressive and irreversible joint damage if detected late.  
This project presents a genomics-focused machine learning demo that transforms selected SNP features into an accessible research-grade risk score through a simple web interface.

This repository is presented as a **research demo**, not as a clinical diagnostic tool.

## Dataset / data source

This application is based on a genotype-driven rheumatoid arthritis prediction workflow.

Public repository scope:
- Streamlit inference app
- demo interface
- model-loading workflow
- local setup files

Not redistributed here:
- raw genotype files
- patient-level genomic datasets
- restricted clinical metadata
- full raw preprocessing pipeline

## Methods / workflow

The current app reflects the following high-level workflow:

1. Upload a genomic/SNP-based input file.
2. Align the uploaded data to the trained feature schema.
3. Run a pre-trained XGBoost model.
4. Generate an RA risk score and class prediction.
5. Display the result through a simple, user-friendly dashboard.

The app content currently references the following feature reduction path:
- 531,689 raw SNPs
- 313 rheumatoid arthritis-associated markers after refinement
- 212 final model features used for inference

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
