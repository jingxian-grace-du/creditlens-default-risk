# CreditLens: Leakage-Aware Credit Default Prediction and Explainable Risk Modelling

## 1. Objective and business context

CreditLens will develop a reproducible binary-classification workflow that estimates a borrower's risk of default from information available at a defined credit decision point. The business aim is to support responsible lending, portfolio risk management and consistent manual review while explicitly preventing information from the future leaking into model development.

## 2. Main research question

How accurately can publicly available, decision-time borrower and repayment information predict subsequent credit default, and which factors drive those predictions without introducing target leakage?

## 3. Target users and practical value

The primary users are credit-risk analysts, lending policy teams and model reviewers. CreditLens should help them rank cases by risk, select an operating threshold suited to business costs, understand key risk drivers and audit whether each feature would genuinely have been available when a decision was made. It is decision support, not an automated approval or rejection system.

## 4. Public dataset options

| Dataset | Outcome and strengths | Limitations for this project |
|---|---|---|
| UCI Default of Credit Card Clients | Predicts default in the following month from customer, credit-limit, bill and repayment history. Public, manageable and well suited to a one-week tabular-classification MVP. | Covers Taiwanese credit-card customers from 2005; variable names and repayment-status codes need careful documentation, and findings may not generalise to current UK lending. |
| FICO HELOC Explainable Machine Learning Challenge | Predicts a binary credit-performance outcome using anonymised home-equity credit-line attributes. Designed for interpretable credit-risk work and contains no direct personal identifiers. | Anonymised feature meanings and special-value codes complicate business interpretation; the decision point and outcome window require careful confirmation from the data documentation. |
| Give Me Some Credit | Predicts serious delinquency within two years and offers realistic class imbalance for ranking and threshold analysis. It is widely used for credit-risk exercises. | Usually accessed through Kaggle, may involve additional access steps, and includes missing values and outliers requiring explicit treatment. Competition conventions can also encourage choices that are less suitable for a leakage-focused study. |

## 5. Recommended dataset and justification

The approved primary dataset is the **UCI Default of Credit Card Clients** dataset. It has a clearly defined next-month outcome, a useful mix of demographic, credit-limit, billing and repayment-history variables, and a scale appropriate for comparing three models within one week. Its temporal structure also makes the feature cut-off and leakage audit concrete. The project must state its age and geographic limitations and must not present the model as deployable in a current UK lending setting.

## 6. Target variable

The binary target is **default payment in the following month**: `1` for default and `0` for no default. Before modelling, the dataset documentation will be used to record the exact observation date, prediction horizon and meaning of default.

## 7. Potential data leakage risks

- Using payments, collections activity, account status or other events recorded after the prediction cut-off.
- Applying imputation, scaling, encoding, feature selection or resampling before the train/test split or outside cross-validation folds.
- Selecting features, hyperparameters or thresholds using the final test set.
- Retaining identifiers or proxy variables that memorise customers or encode the outcome.
- Allowing the same customer or linked records to appear in more than one split, if repeated entities exist.
- Using random splitting where timestamps reveal that a temporal split is required.
- Misreading repayment-status codes or engineered aggregates so that they include the target month.

Each candidate feature will therefore receive an availability-at-decision-time check. Preprocessing will be fitted on training data only, with the final test set held back until evaluation.

For version 1, `SEX`, `AGE`, `MARRIAGE` and `EDUCATION` are retained only for descriptive and subgroup auditing and are excluded from the primary model feature set. This is a responsible modelling choice for a portfolio demonstration; it does not guarantee fairness or regulatory compliance. Extensive fairness certification is outside the MVP.

## 8. Planned models

- **Logistic Regression:** interpretable baseline, with appropriate scaling and regularisation.
- **Random Forest:** non-linear tree ensemble for interactions and a robust feature-importance comparison.
- **XGBoost:** gradient-boosted tree model for stronger tabular predictive performance, with controlled tuning and class-imbalance handling where justified.

All models will use the same leakage-safe split and comparable validation procedure. No performance outcome is assumed in advance.

The approved split is deterministic and group-aware, using fixed seed `42`. A predictor-profile group is defined by all 23 original explanatory variables, including the four audit-only demographics but excluding `ID` and `DEFAULT_NEXT_MONTH`. Groups are indivisible and are assigned to produce an approximately stratified 60% training, 20% validation and 20% final test allocation, jointly balancing total rows and default-class counts. Small allocation deviations are acceptable. Persisted row assignments will be identical for every model.

For the Day 3 Logistic Regression baseline, the six raw repayment-status fields are categorical and one-hot encoded with unknown-category handling. The 13 monetary fields are standardised. Both transformations are fitted on training data only. Raw repayment codes remain distinct, while outlier treatment, imputation, resampling and class weighting are deliberately excluded. The fixed model uses L2 regularisation, `C=1.0`, `solver="liblinear"`, `max_iter=1000` and `random_state=42`; it is not tuned on validation data.

Day 3 validation evaluation is distinct from final test evaluation. The validation partition is used for baseline metrics, while the final test partition remains untouched. Threshold 0.5 is reported only as a conventional baseline reference and is not the final business threshold. The approved 70%-recall selection rule is deferred to the model-comparison stage.

For Day 4, Random Forest uses the same 19 model fields. Repayment codes are one-hot encoded inside the pipeline and monetary fields pass through without scaling or outlier treatment. A bounded 12-configuration `RandomizedSearchCV` uses four-fold shuffled `StratifiedGroupKFold`, seed `42`, Average Precision as the refit metric and ROC-AUC as a secondary metric. Groups use all 23 original predictors, including audit-only fields solely as grouping metadata. Validation is evaluated only after training-only selection; threshold 0.5 remains descriptive. Impurity and validation permutation importance are reported with explicit correlation and bias limitations; SHAP remains deferred.

## 9. Evaluation metrics

Model comparison will report:

- ROC-AUC for overall ranking discrimination.
- Average Precision (AP), calculated with `average_precision_score`, to summarise precision–recall performance for the less frequent default class.
- Precision, recall and F1-score at the selected operating threshold.
- A confusion matrix showing true positives, false positives, true negatives and false negatives.

Results will include validation uncertainty where feasible and a final, one-time test-set assessment.

## 10. Threshold selection and business trade-offs

The default threshold of 0.5 will not be assumed to be optimal. On the validation set, default-class recall must be at least 70%; among thresholds meeting that constraint, the threshold with the highest precision will be selected. Ties will be resolved deterministically by choosing the highest threshold. The resulting false-positive rate and proportion of customers flagged for review will also be reported. The 70% recall requirement is an illustrative portfolio assumption, not a regulatory or real lending standard. The test set must not be used to select or revise the threshold.

Day 6 applies this frozen rule to the provisional validation candidate without refitting any model. Paired, class-stratified bootstrap resampling quantifies the validation AP difference between Random Forest and XGBoost; a small difference or an interval spanning zero will not be described as superiority. The resulting validation threshold must not be revised using final-test outcomes.

Before Day 7 final-test access, a machine-readable decision lock freezes the selected model artefact and its SHA-256 hash, embedded preprocessing, validation-selected threshold, selection basis and paired uncertainty statement. The one-time final-test evaluation scores only that frozen pipeline and saves aggregate metrics and figures. Final-test results cannot trigger model, preprocessing, calibration or threshold changes. The validation recall floor is a selection target rather than a promise of final-test recall.

Missing a likely default may create credit loss; incorrectly flagging a reliable borrower may reduce approvals, harm customer experience or increase manual-review costs. The chosen threshold and its assumptions will be documented before the one-time final test evaluation.

## 11. Explainability

Global SHAP explanations and exactly three clearly labelled validation case-level SHAP examples will be produced for the provisional model: one true positive, one false positive and one false negative. The examples will be selected only after applying the validation-chosen threshold; they will not influence model or threshold selection. If a required error category is absent, that fact will be reported rather than substituting or fabricating an example. Logistic coefficients and permutation or model-native feature importance may provide complementary checks. Explanations will describe associations rather than causal effects, include feature-direction context, and be checked for instability, correlated-feature effects and potentially sensitive proxies. Day 6 global Tree SHAP uses a fixed-seed, class-stratified validation sample for bounded runtime. Final-test explanations remain deferred.

## 12. One-week implementation schedule

| Day | Deliverable |
|---|---|
| 1 | Confirm dataset and outcome definition; document provenance, licence, schema and decision-time feature cut-off. |
| 2 | Run data-quality checks and exploratory analysis; define the split strategy and leakage register. |
| 3 | Build training-only preprocessing and the Logistic Regression baseline. |
| 4 | Train and validate Random Forest and XGBoost using bounded, reproducible tuning. |
| 5 | Compare validation metrics and choose a threshold from stated business trade-offs. |
| 6 | Perform validation-only uncertainty, threshold selection, explainability and error analysis. |
| 7 | Record the immutable decision lock, run the authorised one-time final-test evaluation and consolidate the final report. |

## 13. MVP scope

Version 1 will include the approved UCI dataset, a documented data dictionary and leakage audit, reproducible preprocessing, the three specified models, validation-based comparison, the approved recall-constrained threshold rule, final test metrics, a confusion matrix, global SHAP explainability and the three specified case-level SHAP explanations where those outcome categories exist.

Version 1 will exclude live or proprietary data, data collection, deployment, an API or user interface, real-time scoring, automated lending decisions, causal claims, production monitoring, fairness certification, regulatory validation, extensive hyperparameter optimisation, multiple-dataset benchmarking and model results presented as suitable for real lending decisions.
