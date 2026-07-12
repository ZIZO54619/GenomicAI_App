# Final K25 Provenance

## Configuration

- Selected K: 25
- Stability threshold: 0.80
- Number of stable SNPs: 22
- Final model/encoding: XGBoost Additive
- Interpretation boundary: model-prioritized SNPs, not causal variants

## Source Paths

- Jaccard values: `03 Methodology/03 Processing (Modeling)/Output/stability/XGBoost__Additive__jaccard_vs_K.csv`
- Stable-SNP count across K: `03 Methodology/03 Processing (Modeling)/Output/stability/XGBoost__Additive__stable_snp_count_vs_K.csv`
- Existing final K25 stable SNP list: `03 Methodology/04 Postprocessing/4.2 Enrichment Analysis/Input/K25_stable_snps_22.csv`
- Fold-level SHAP importance used to reconstruct K25 selection frequency: `03 Methodology/03 Processing (Modeling)/Output/stability/XGBoost__Additive__all_folds_shap_importance.csv`
- K25 top interaction output: `03 Methodology/03 Processing (Modeling)/Output/GenomicAI_K25_Interaction_Rerun_Only_LOCAL/tables/xgboost_K25_top_15_interaction_pairs.csv`
- K25 interaction manifest: `03 Methodology/03 Processing (Modeling)/Output/GenomicAI_K25_Interaction_Rerun_Only_LOCAL/manifests/k25_interaction_rerun_manifest.csv`
- Matched GWAS rows: `03 Methodology/04 Postprocessing/4.2 Enrichment Analysis/Output/Tables/01_matched_22_stable_SNPs_GWAS_rows.csv`
- Direct SNP-to-gene mapping: `03 Methodology/04 Postprocessing/4.2 Enrichment Analysis/Output/Tables/03_direct_GWAS_SNP_to_gene_mapping.csv`
- Direct GWAS gene set summary: `03 Methodology/04 Postprocessing/4.2 Enrichment Analysis/Output/Tables/04_direct_GWAS_gene_set_summary.csv`
- RA enrichment terms: `03 Methodology/04 Postprocessing/4.2 Enrichment Analysis/Output/Tables/09_rheumatoid_arthritis_significant_terms.csv`
- K25 interaction notebook: `03 Methodology/03 Processing (Modeling)/Output/GenomicAI_K25_Interaction_Rerun_Only_LOCAL/GenomicAI_K25_Interaction_Rerun.ipynb`
- K25 enrichment script: `03 Methodology/04 Postprocessing/4.2 Enrichment Analysis/GenomicAI_K25_HLA_nonHLA_Enrichment_No_STRING.R`

## K20/K25 Status

K20 was the historical modeling configuration recorded in the original modeling run configuration. K25 was selected after evaluating candidate K values because it achieved the numerically highest mean Jaccard similarity among evaluated K settings. Therefore, K20 outputs are historical/exploratory and must not be used as the final biological interpretation layer.

## K25 Interaction Scope

The K25 interaction analysis reused the existing selected XGBoost Additive model and did not rerun full model benchmarking, Optuna, nested cross-validation, or model selection. The interaction analysis focused on the final 22 K25 stable SNPs, producing 231 possible SNP-SNP pairs.

## K25 Stable-List Consistency Check

The existing `K25_stable_snps_22.csv` file was compared against a K25 stable SNP list reconstructed from the saved fold-level XGBoost Additive SHAP-importance output using K=25 and frequency >= 0.80.

- SNP ID set exact match: True.
- Existing K25 stable count: 22.
- Recomputed-from-fold-output K25 stable count: 22.
- Missing from existing list: [].
- Extra in existing list: [].
- Fold/frequency metadata exact match: False.
- SNP rows with fold/frequency metadata differences: 6.

Result: the final SNP ID set matches exactly. Per-SNP fold/frequency metadata is not fully identical for every SNP, so the final package uses the existing `K25_stable_snps_22.csv` for the final stable-SNP table and uses the reconstructed fold-level selection frequencies for the K25 selection-frequency figure.

## SHA256 Hashes

- `03 Methodology/04 Postprocessing/4.2 Enrichment Analysis/Input/K25_stable_snps_22.csv`: `9e96d2ca3f5efe94dd9aa607f3be304b13d41d9e3ba678f18ec5cd98403cf783`
- `03 Methodology/03 Processing (Modeling)/Output/stability/XGBoost__Additive__jaccard_vs_K.csv`: `d3116e2e17e51e6f18cde78c56b96c36af7dbaa8d9282e04a504015ac9fc660c`
- `03 Methodology/03 Processing (Modeling)/Output/stability/XGBoost__Additive__stable_snp_count_vs_K.csv`: `a0dab2cd6756375bb81d9876dabf978c7ce8a9b3c026880d383be697684b5902`
- `03 Methodology/03 Processing (Modeling)/Output/GenomicAI_K25_Interaction_Rerun_Only_LOCAL/tables/xgboost_K25_top_15_interaction_pairs.csv`: `ab2dd88e5d9f5979eeea1e7fbfe0ee3f3d35b4e591e563ad6b6f94757f268655`
- `03 Methodology/04 Postprocessing/4.2 Enrichment Analysis/Output/Tables/01_matched_22_stable_SNPs_GWAS_rows.csv`: `da34b1b5e41772a1889d1ea5d49a3f82df3c6ed7f9e28d6506a0cfe39ff20083`
- `03 Methodology/04 Postprocessing/4.2 Enrichment Analysis/Output/Tables/03_direct_GWAS_SNP_to_gene_mapping.csv`: `5a881395a2c1d5f144bef26a086c0c9b75fafe39534c8411753dff5e3ffdef45`
- `03 Methodology/04 Postprocessing/4.2 Enrichment Analysis/Output/Tables/04_direct_GWAS_gene_set_summary.csv`: `2b7abe3675f17b607915bee1d546958a69925df7df1776e4e638ccf997e060d6`
- `03 Methodology/04 Postprocessing/4.2 Enrichment Analysis/FINAL_K25_BOOK_ASSETS/Table_Final_K25_Stability_Summary.csv`: `ee0e5f29e85e7b6ad16a187e976c384b299e83b656aedfc97a6d965f633626b5`
- `03 Methodology/04 Postprocessing/4.2 Enrichment Analysis/FINAL_K25_BOOK_ASSETS/Table_Final_K25_Stable_SNPs.csv`: `376602e757e395b81386c937ff8fde4f6fc0a3429b8e70db2651ae88b8c31c17`
- `03 Methodology/04 Postprocessing/4.2 Enrichment Analysis/FINAL_K25_BOOK_ASSETS/Table_Final_K25_Top_Interactions.csv`: `694a9cf229edd888c502a6f54d06645283c7cbbb075f290a3035d613391b3ce4`
- `03 Methodology/04 Postprocessing/4.2 Enrichment Analysis/FINAL_K25_BOOK_ASSETS/Table_Final_K25_Gene_Mapping_Summary.csv`: `c77330be25cedf10ec251c35a2e52a0c6997238912da174dea08b0665de0a2c9`
- `03 Methodology/04 Postprocessing/4.2 Enrichment Analysis/FINAL_K25_BOOK_ASSETS/FINAL_K25_BOOK_ASSET_MANIFEST.csv`: `6fe5ea72b40ac9fa24fe435e3df9e2243018e56d3dad4de77dc74073bcd00830`
- `03 Methodology/04 Postprocessing/4.2 Enrichment Analysis/FINAL_K25_BOOK_ASSETS/FINAL_FEATURE_SELECTION_FLOWCHART_CONTENT.csv`: `573dfc48f5e91865c241bec52f22b9119f46816e73ed2e1e6519f1e4faf040df`
- `03 Methodology/04 Postprocessing/4.2 Enrichment Analysis/FINAL_K25_BOOK_ASSETS/final_k25_biological_interpretation_config.json`: `823b2fee19b50e1418b773908761ad82ef6e8d0f1352603d5715f4b901db8975`
