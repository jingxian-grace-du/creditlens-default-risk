# Leakage register

This register is defined before exploratory analysis or modelling and must be updated if new risks are discovered.

| ID | Risk | Stage | Planned control | Verification evidence |
|---|---|---|---|---|
| L01 | `Y` or an outcome-derived field enters the feature matrix. | Ingestion | Explicit feature allow-list (`X1`–`X23`); keep `Y` separate. | Schema assertion and feature list. |
| L02 | October 2005 payment, collections or account-status information is used to predict the October outcome. | Ingestion/feature engineering | Fix the cut-off at end September 2005; reject all post-cut-off fields and aggregates. | Feature timing review against the data dictionary. |
| L03 | `ID` memorises rows or encodes collection order. | Feature selection | Exclude `ID` from every model; use it only for uniqueness and split-integrity checks. | Pipeline feature assertion. |
| L04 | Duplicate predictor profiles or repeated customers cross data partitions. | Splitting | Define profiles from all 23 original predictors, excluding `ID` and target; keep each group wholly within one partition. The remediated split has 29,944 groups, of which 52 are repeated, and zero cross partitions. | `test_no_predictor_profile_crosses_partitions` passes; summary reports zero crossings. **Closed.** |
| L05 | Imputation, scaling, encoding, outlier rules or feature selection learns from validation/test data. | Preprocessing | Fit every learned transformation on the training partition or training folds only; use pipelines. | Fitted-pipeline review and tests. |
| L06 | Resampling or class balancing occurs before the split or outside training folds. | Training | Apply any justified resampling only inside training folds. Never resample validation or test data. | Sampling counts by fold. |
| L07 | Hyperparameters or model choice are optimised on the test set. | Model selection | Use training/cross-validation and validation only; evaluate test once after choices are frozen. | Experiment log and final-evaluation timestamp. |
| L08 | The operating threshold is selected or revised using test labels. | Thresholding | Apply the approved recall/precision rule on validation only and freeze the chosen threshold before test evaluation. | Saved threshold decision record. |
| L09 | SHAP case selection influences modelling or threshold decisions. | Explainability | Choose the three outcome examples only after final predictions, solely for explanation. | Analysis order and selection record. |
| L10 | Manual inspection of test outcomes indirectly changes preprocessing or feature engineering. | Evaluation | Keep test labels out of exploratory and iterative notebooks; predeclare final checks. | Notebook and workflow review. |
| L11 | Undocumented repayment or demographic codes are silently interpreted using full-data outcome patterns. | Data quality | Label codes as undocumented, determine handling without reference to validation/test outcomes, and fit any mapping on training data only. | Code-frequency report and decision log. |
| L12 | A random split is misrepresented as out-of-time validation. | Reporting | State that all rows share one historical observation window and that the split estimates within-sample-era generalisation only. | Report wording review. |
| L13 | Sensitive attributes or proxies create harmful or misleading explanations. | Governance/explainability | Clearly label `SEX`, `AGE`, `MARRIAGE` and `EDUCATION`; report subgroup limitations and avoid deployment claims. | Model card/limitations section. |
| L14 | Sensitive/audit attributes enter the primary model despite the approved scope. | Feature selection | Enforce a primary feature allow-list that excludes `SEX`, `AGE`, `MARRIAGE` and `EDUCATION`; retain them only for descriptive and subgroup audits. | Automated feature-list test. |

The final test partition is an evaluation asset, not a source of design feedback. If a defect is discovered after test evaluation, document it and restart the evaluation protocol with a newly justified hold-out strategy rather than repeatedly consulting the same test results.

Excluding the four audit variables is a responsible modelling choice for this demonstration, but does not guarantee fairness or regulatory compliance. Extensive fairness certification is outside the MVP.
