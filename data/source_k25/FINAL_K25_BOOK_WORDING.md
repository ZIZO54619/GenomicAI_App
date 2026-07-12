# Final K25 Book Wording

## Methodology: SHAP Stability Selection

SHAP-based stability selection was performed for the selected XGBoost Additive modeling layer across candidate K values. For each candidate K, the top-ranked SNPs from each outer fold were compared across folds using Jaccard similarity. K=25 was selected as the final biological-interpretation setting because it achieved the numerically highest mean Jaccard similarity among the evaluated candidate K values. SNPs selected in at least 80% of folds were retained as K25 SHAP-stable SNPs.

## Results: K25 Stable SNP Selection

The final K25 stability rule retained 22 SHAP-stable SNPs. These SNPs define the final model-prioritized genomic importance map used for downstream biological interpretation. The K25 stable list should be interpreted as a reproducible model-prioritized feature set, not as a list of causal variants.

## Results: K25 Interaction Analysis

Pairwise SHAP interaction values were evaluated among the 22 K25 stable SNPs, yielding 231 possible SNP-SNP pairs. The reported interaction strengths are model-based SHAP interaction summaries and should not be interpreted as proof of causal epistasis.

## Results: Direct GWAS Mapping and Enrichment

The final 22 K25 SHAP-stable SNPs were mapped to GWAS Catalog mapped genes using direct mapped-gene annotations. The mapping produced 29 direct GWAS-mapped genes, including 5 HLA genes and 24 Non-HLA genes. These gene groups were evaluated separately in enrichment analysis to support biological annotation of the final model-prioritized SNP set.

## Figure Captions

Figure K25 Stability Across K. SHAP stability across evaluated candidate K values for the XGBoost Additive model. K=25 achieved the numerically highest mean Jaccard similarity among evaluated candidate K values and was selected as the final biological-interpretation setting.

Figure K25 Selection Frequency. SNP selection frequency under the K25 stability rule. The dashed line marks the frequency >= 0.80 stability threshold, and highlighted SNPs indicate the final 22 K25 SHAP-stable SNPs.

Figure K25 Genomic Importance Map. Genomic-coordinate view of the final 22 K25 SHAP-stable SNPs using cross-fold mean absolute SHAP values. This is not a GWAS Manhattan plot and does not display association p-values.

Figure K25 Top Interactions. Top model-based SHAP interactions among K25 stable SNPs. Interaction values summarize model behavior among the final 22 stable SNPs and do not prove causal epistasis.

## Table Captions

Table Final K25 Stability Summary. Candidate K settings, mean Jaccard similarity, Jaccard standard deviation, stable SNP count, and final selection decision for the XGBoost Additive SHAP-stability analysis.

Table Final K25 Stable SNPs. Final 22 K25 SHAP-stable SNPs retained at frequency >= 0.80 for biological interpretation.

Table Final K25 Top Interactions. Top K25 model-based SHAP SNP-SNP interaction pairs derived from the final 22 stable SNPs.

Table Final K25 Gene Mapping Summary. Direct GWAS SNP-to-gene mapping summary for the final K25 stable SNP set, including matched SNPs, matched GWAS rows, SNP-gene pairs, direct mapped genes, HLA genes, and Non-HLA genes.

## Discussion: Non-Causal Interpretation Boundaries

The final K25 SNP set should be interpreted as a model-prioritized genomic importance map rather than a causal variant list. SHAP stability identifies variants that repeatedly contributed to model predictions across validation folds, while GWAS mapping and enrichment provide biological annotation context. These analyses support hypothesis generation and pathway-level interpretation, but they do not establish causal mechanisms, clinical diagnostic rules, or experimentally validated epistasis.

## Safe GWAS Feature-Selection Wording

The GWAS Catalog-guided workflow produced a verified RA-unique list of 2,193 rsIDs, which was intersected with the NARAC genotype map to form a 314-SNP NARAC-Compatible RA-Informed SNP Panel. One low genotype-diversity SNP was removed, yielding a 313-SNP imputation panel. Among these 313 SNPs, 231 had at least one missing genotype call before imputation, with 10,086 missing calls in total; after Beagle imputation, no missing calls remained. Early raw GWAS counts such as 3,993 and 2,783 should be described only as document-supported unless the primary raw download and counting script are archived.
