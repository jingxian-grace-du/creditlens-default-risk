# Day 7 one-time final-test evaluation

## Decision lock

Before any final-test outcomes were loaded, `reports/day7_final_decision_lock.json` froze `models/day4_random_forest_pipeline.joblib` at SHA-256 `1d52eed5ae717e3e0bc09d319362263fcebcfb9f691a23951df32ac5c935037e`, its embedded preprocessing, and threshold `0.1912638431828811`. Random Forest was selected because it had the highest numerical validation Average Precision. The paired Random-Forest-minus-XGBoost AP interval was `[-0.0075220545, 0.0137943885]`; no superiority claim was made. The lock prohibits model, preprocessing or threshold changes after viewing these results.

## Final-test isolation

The dedicated loader skipped training and validation rows before materialising the target-bearing final-test frame. Exactly 6,000 final-test rows were scored once using only the frozen Random Forest pipeline. No fitting, refitting, tuning, recalibration, retraining or alternative-model comparison was performed. No general row-level prediction file was saved.

## Aggregate final-test results

| Measure | Result |
|---|---:|
| Class prevalence | 0.221167 |
| ROC-AUC | 0.779397 |
| Average Precision (AP) | 0.557033 |
| Precision at frozen threshold | 0.406195 |
| Recall at frozen threshold | 0.691786 |
| F1 | 0.511848 |
| False-positive rate | 0.287182 |
| Proportion flagged | 0.376667 |

Confusion matrix: `[[3331, 1342], [409, 918]]`.

## Validation-to-test comparison

| metric                        |   validation |   final_test |   final_minus_validation |
|:------------------------------|-------------:|-------------:|-------------------------:|
| roc_auc                       |     0.771917 |     0.779397 |                 0.007480 |
| average_precision             |     0.539980 |     0.557033 |                 0.017054 |
| precision_at_frozen_threshold |     0.410880 |     0.406195 |                -0.004685 |
| recall_at_frozen_threshold    |     0.700075 |     0.691786 |                -0.008289 |
| f1_at_frozen_threshold        |     0.517837 |     0.511848 |                -0.005989 |
| false_positive_rate           |     0.285042 |     0.287182 |                 0.002140 |
| proportion_flagged            |     0.376833 |     0.376667 |                -0.000167 |

Differences are described rather than used to revise any decision. The 70% recall requirement was a validation threshold-selection target, not a guarantee that final-test recall would reach 70%. Any decline or increase observed here leaves the frozen pipeline and threshold unchanged.

## Limitations

This is one held-out assessment on a historical public dataset, not evidence of production lending suitability, causality, regulatory compliance, fairness certification or stability under population drift. The audit-only demographic variables were excluded from the model, but that does not guarantee fairness. The model remains decision-support research rather than an automated lending system.

## Artefacts

- Decision lock: `reports/day7_final_decision_lock.json`
- Metrics: `reports/day7_final_test_metrics.json`
- Validation comparison: `reports/day7_validation_test_comparison.csv`
- ROC: `reports/figures/day7_final_test_roc.png`
- Precision–recall: `reports/figures/day7_final_test_precision_recall.png`
- Confusion matrix: `reports/figures/day7_final_test_confusion_matrix.png`
