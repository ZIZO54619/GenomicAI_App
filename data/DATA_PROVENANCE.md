# App data provenance

The dashboard tables are aligned to the **final graduation book** supplied on 12 July 2026.

- The final biological interpretation layer is **K = 25**, frequency threshold **>= 0.80**, with **22 stable SNPs**.
- `stable_snps_k25.csv` follows Table 36 in the final book so the website and submitted report remain consistent.
- The original `FINAL_K25_BOOK_ASSETS` package is retained under `data/source_k25/` for traceability. Its provenance note records fold/frequency metadata differences for six SNP rows, although the 22-SNP identifier set matches exactly.
- Interaction values are model-based SHAP interaction magnitudes and do not prove causal epistasis.
- Enrichment uses direct GWAS Catalog mapped genes without STRING/network expansion and provides annotation context rather than causal evidence.
