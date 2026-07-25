# Day 6 validation-only selection and explanation report

## Scope and isolation

Day 6 reuses the three frozen fitted pipelines and the persisted grouped split. It materialises training and validation partitions through the established safe loader, but performs all Day 6 scoring, thresholding, uncertainty analysis, SHAP explanation and error analysis on the 6,000-row validation partition only. Final-test outcomes were not loaded, materialised, inspected, scored or summarised. These findings are not final generalisation performance.

## Provisional model selection and paired uncertainty

Random Forest remains the provisional primary candidate because its validation AP (0.539980) is numerically above XGBoost (0.536643). The paired AP difference is +0.003336. A 2,000-iteration fixed-seed paired stratified bootstrap gives a 95% percentile interval of [-0.007522, 0.013794], which includes zero. The difference remains negligible and uncertain; no superiority claim is made and no final production model is selected.

## Frozen operating-threshold rule

On validation only, the pre-approved rule requires default recall of at least 70%, maximises precision among qualifying observed thresholds, and resolves an exact precision tie by taking the highest threshold. It selects `0.191263843183`.

| Measure | Validation result |
|---|---:|
| Precision | 0.410880 |
| Recall | 0.700075 |
| F1 | 0.517837 |
| False-positive rate | 0.285042 |
| Proportion flagged | 0.376833 |

Confusion matrix: `[[3341, 1332], [398, 929]]`.

The 70% recall floor is an illustrative portfolio assumption, not a regulatory or real lending standard. This threshold is frozen from validation and must not be revised using the final test set.

## Explainability

Tree SHAP explains the frozen Random Forest. Global summaries use a deterministic class-stratified sample of 1,000 validation rows for bounded runtime. One-hot contributions are retained at encoded level and also summed to the 19 source variables. Exactly three post-threshold cases are shown: one true positive, one false positive and one false negative. They were selected deterministically after threshold selection and did not influence model or threshold choice.

SHAP values describe model associations, not causal effects. Correlated billing, payment and repayment-status fields can divide or mask contributions; one-hot aggregation aids readability but does not remove dependence or proxy risks. Individual explanations are examples, not representative customer narratives or lending reasons.

## Error analysis

At the selected threshold there are 1,332 false positives and 398 false negatives. Aggregate feature summaries compare those groups with the complete validation partition without publishing a general row-level prediction file. The analysis is descriptive and validation-specific.

False positives have a higher mean `PAY_0` code (0.272) and a lower median credit limit (£90,000) than the full validation sample (-0.014 and £140,000, respectively). False negatives have a lower mean `PAY_0` code (-0.397) while their median first statement balance (£30,086) exceeds the validation median (£23,626). These are aggregate associations: raw repayment codes are ordinal labels with special values, and correlated financial fields prevent causal or individual-level conclusions.

## Artefacts

- Metrics: `reports/day6_validation_analysis_metrics.json`
- Threshold: `reports/day6_selected_threshold.json`
- Paired uncertainty: `reports/day6_paired_ap_uncertainty.json`
- Model comparison: `reports/day6_validation_model_comparison.csv`
- Global SHAP: `reports/day6_random_forest_global_shap_encoded.csv`, `reports/day6_random_forest_global_shap_source.csv`
- Case explanations: `reports/day6_case_level_metadata.json`, `reports/day6_case_level_shap.csv`
- Error summaries: `reports/day6_error_group_summary.csv`, `reports/day6_error_feature_summary.csv`
- Figures: `reports/figures/day6_*.png`
