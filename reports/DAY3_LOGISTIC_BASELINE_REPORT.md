# Day 3 Logistic Regression baseline

## Scope and leakage controls

The persisted group-aware split was loaded without regeneration. Preprocessing and Logistic Regression were fitted on 18,000 training rows only. The 6,000-row validation partition was used only for baseline evaluation. Final-test rows were skipped before target-bearing data were materialised, and no final-test outcome was scored, summarised or inspected.

## Preprocessing

- Six repayment-status fields were treated as categorical and one-hot encoded with `handle_unknown="ignore"`; distinct raw codes, including `-2` and `0`, were retained.
- Thirteen monetary fields were standardised using training-fitted means and scales.
- The pipeline used exactly 19 approved source fields and produced 76 encoded features.
- No imputation, resampling, winsorisation, capping, outlier removal or outcome-informed category collapsing was performed.

## Fixed model configuration

Logistic Regression used L2 regularisation, `C=1.0`, `solver="liblinear"`, `max_iter=1000`, `random_state=42` and no class weighting. No validation-led hyperparameter tuning was performed.

## Validation results

| Metric | Value |
|---|---:|
| ROC-AUC | 0.754405 |
| Average Precision (AP) | 0.519471 |
| Precision at 0.5 | 0.669617 |
| Recall at 0.5 | 0.342125 |
| F1-score at 0.5 | 0.452868 |
| Proportion flagged at 0.5 | 0.113000 |

Confusion matrix, with rows as actual `[0, 1]` and columns as predicted `[0, 1]`:

```text
[4449, 224]
[873, 454]
```

The 0.5 threshold is for baseline reporting only. It is not the final business threshold, and the approved 70%-recall threshold rule has not been applied.

## Artefacts

- Pipeline: `models/day3_logistic_regression_baseline.joblib`
- Metrics: `reports/day3_logistic_validation_metrics.json`
- Coefficients: `reports/day3_logistic_coefficients.csv`
- ROC curve: `reports/figures/day3_logistic_validation_roc.png`
- Precision–recall curve: `reports/figures/day3_logistic_validation_precision_recall.png`

Convergence warnings: []
