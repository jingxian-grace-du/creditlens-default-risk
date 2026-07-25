# Day 5 XGBoost validation report

## Scope and isolation

The persisted grouped split was reused without regeneration. The bounded search used 18,000 training rows only. Four-fold shuffled `StratifiedGroupKFold` used exact group identifiers formed from all 23 original predictors, with audit-only variables serving solely as grouping metadata. No group crossed an internal fold.

The selected pipeline was evaluated on 6,000 validation rows only after training-CV selection. Final-test target values were not materialised, inspected, scored or summarised. Validation results are not final generalisation estimates.

## Preprocessing and bounded search

The model matrix contains exactly the approved 19 predictors. Six repayment-status fields were one-hot encoded with `handle_unknown="ignore"` inside the pipeline. Thirteen monetary fields passed through without scaling, capping or outlier removal. No resampling or validation-led early stopping was used.

`RandomizedSearchCV` sampled 12 configurations and performed 48 training-only fits. Average Precision was the refit metric and ROC-AUC was recorded secondarily. Search time was 669.438 seconds and total workflow time was 708.390 seconds.

Fixed XGBoost configuration:

```json
{
  "eval_metric": "logloss",
  "n_jobs": -1,
  "objective": "binary:logistic",
  "random_state": 42,
  "tree_method": "hist"
}
```

Selected hyperparameters:

```json
{
  "colsample_bytree": 0.85,
  "gamma": 0.0,
  "learning_rate": 0.03,
  "max_depth": 3,
  "min_child_weight": 1,
  "n_estimators": 600,
  "reg_lambda": 10.0,
  "scale_pos_weight": 3.5,
  "subsample": 0.85
}
```

Best training-only grouped-CV Average Precision: 0.562699

Best configuration's grouped-CV ROC-AUC: 0.786319

## Validation comparison

| Metric | Logistic Regression | Random Forest | XGBoost |
|---|---:|---:|---:|
| ROC-AUC | 0.754405 | 0.771917 | 0.771792 |
| Average Precision (AP) | 0.519471 | 0.539980 | 0.536643 |
| Precision at 0.5 | 0.669617 | 0.661829 | 0.451595 |
| Recall at 0.5 | 0.342125 | 0.343632 | 0.618689 |
| F1-score at 0.5 | 0.452868 | 0.452381 | 0.522099 |
| Proportion flagged at 0.5 | 0.113000 | 0.114833 | 0.303000 |

XGBoost confusion matrix at diagnostic threshold 0.5:

```text
[3676, 997]
[506, 821]
```

Relative to Random Forest, the numerical AP change is -0.003336, described as negligible under the predeclared descriptive bands in code. This is not an uncertainty or significance conclusion and does not select a final model.

Threshold 0.5 is diagnostic only. No final model or business threshold was selected.

## Gain importance

Encoded-feature gain and source-aggregated gain are reported. Gain measures how much fitted tree splits improve the objective; correlated inputs can divide or mask importance, and importance is not causal. No permutation importance or SHAP analysis was performed on Day 5.

## Artefacts

- Pipeline: `models/day5_xgboost_pipeline.joblib`
- Search results: `reports/day5_xgboost_search_results.csv`
- Selected parameters and CV audit: `reports/day5_xgboost_selected_params.json`
- Metrics: `reports/day5_xgboost_validation_metrics.json`
- Encoded gain: `reports/day5_xgboost_encoded_importance.csv`
- Source gain: `reports/day5_xgboost_source_importance.csv`
- Model comparison: `reports/day5_validation_model_comparison.csv`
- ROC curve: `reports/figures/day5_xgboost_validation_roc.png`
- Precision–recall curve: `reports/figures/day5_xgboost_validation_precision_recall.png`

Warnings captured during search: []
