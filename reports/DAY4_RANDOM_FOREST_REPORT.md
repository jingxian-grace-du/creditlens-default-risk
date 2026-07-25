# Day 4 Random Forest validation report

## Scope and isolation

The persisted grouped split was reused without regeneration. The bounded search used 18,000 training rows only. Hyperparameters were selected by four-fold `StratifiedGroupKFold` using exact group identifiers formed from all 23 original predictors. Audit-only variables contributed only to group identifiers and never entered the 19-field model matrix. Zero predictor-profile groups crossed internal folds.

The selected pipeline was evaluated once on 6,000 validation rows. Final-test outcomes were not materialised, inspected, scored or summarised. These validation results are not final generalisation estimates.

## Preprocessing and search

Six repayment-status variables were one-hot encoded with `handle_unknown="ignore"`. Thirteen monetary fields passed through without scaling. No imputation, resampling, capping, winsorisation or outlier removal was performed. All learned encoding was fitted inside the pipeline.

The deterministic search sampled 12 configurations and performed 48 training-only CV fits. Average Precision was the refit metric and ROC-AUC was recorded secondarily. Search time was 1144.966 seconds; total workflow time including validation permutation importance was 1187.872 seconds.

Selected classifier hyperparameters:

```json
{
  "class_weight": null,
  "max_depth": 20,
  "max_features": "sqrt",
  "min_samples_leaf": 10,
  "min_samples_split": 2,
  "n_estimators": 300
}
```

## Validation metrics

| Metric | Random Forest | Logistic Regression |
|---|---:|---:|
| ROC-AUC | 0.771917 | 0.754405 |
| Average Precision (AP) | 0.539980 | 0.519471 |
| Precision at 0.5 | 0.661829 | 0.669617 |
| Recall at 0.5 | 0.343632 | 0.342125 |
| F1-score at 0.5 | 0.452381 | 0.452868 |
| Proportion flagged at 0.5 | 0.114833 | 0.113000 |

Random Forest confusion matrix at reporting threshold 0.5:

```text
[4440, 233]
[871, 456]
```

Threshold 0.5 remains a reporting convention, not the selected operating threshold. Metric differences on one validation partition do not by themselves establish model superiority or uncertainty.

## Importance limitations

Impurity importance can favour continuous or high-cardinality predictors and can distribute or inflate importance unpredictably among correlated variables. Source-level repayment importance aggregates its one-hot indicators. Validation permutation importance reports the change in Average Precision after shuffling each source field, but correlated predictors can mask one another and the result remains validation-specific. Neither measure is causal. SHAP was not performed.

## Artefacts

- Pipeline: `models/day4_random_forest_pipeline.joblib`
- Search results: `reports/day4_random_forest_search_results.csv`
- Selected parameters and CV audit: `reports/day4_random_forest_selected_params.json`
- Metrics: `reports/day4_random_forest_validation_metrics.json`
- Encoded importance: `reports/day4_random_forest_encoded_importance.csv`
- Source importance: `reports/day4_random_forest_source_importance.csv`
- Model comparison: `reports/day4_validation_model_comparison.csv`
- ROC curve: `reports/figures/day4_random_forest_validation_roc.png`
- Precision–recall curve: `reports/figures/day4_random_forest_validation_precision_recall.png`

Warnings captured during search: []

## Resource-control notes

An initial sparse-input attempt was interrupted before validation because tree fitting was excessively slow. The completed search used dense one-hot output, which is memory-safe at 76 encoded features and does not change the encoded values or search space. After the completed search and saved validation evaluation, sandbox restrictions blocked process creation for parallel permutation importance; interpretation resumed from the saved selected model with serial permutation importance. The search was not repeated, hyperparameters and stored metrics remained fixed, and no final-test data were accessed.
