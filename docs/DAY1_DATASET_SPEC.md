# Day 1 dataset specification

## Provenance and scope

The approved dataset is UCI Machine Learning Repository dataset 350, **Default of Credit Card Clients**, attributed to I-Cheng Yeh. UCI assigns dataset DOI `10.24432/C55S3H`, reports 30,000 instances and 23 explanatory features, and distributes one Excel workbook. The data concern credit-card clients of a Taiwanese issuer and historical activity in 2005. UCI donated the repository entry on 25 January 2016; its suggested dataset citation uses 2009, matching the publication year of the introductory paper.

The official source and licence record are documented in `data/README.md`. The raw archive and extracted workbook are retained unchanged in `data/raw/` with SHA-256 checksums.

This historical, geographically specific sample is suitable for a methodological MVP, not for estimating current UK portfolio risk or making real lending decisions.

## Outcome and prediction timing

The prediction point is fixed at the end of September 2005, after September repayment status, bill amount and previous payment amount are known. The target, raw field `Y`, is whether the customer defaults on payment in the following month, October 2005 (`1` yes, `0` no). April–September histories and static/customer attributes are candidate predictors. October outcome information and anything generated after the cut-off are prohibited features.

The source is a single cross-section: it does not provide a row-level application date or multiple observation dates suitable for an out-of-time split.

## Split strategy

After schema, target-value, identifier-uniqueness and duplicate checks, create one deterministic, group-aware, approximately stratified partition using fixed seed `42`:

- Approximately 60% training data for fitting model parameters.
- Approximately 20% validation data for model comparison and the approved threshold rule.
- Approximately 20% test data, held untouched for one final evaluation.

An identical predictor profile is defined by all 23 original explanatory variables. This includes `SEX`, `AGE`, `MARRIAGE` and `EDUCATION` for grouping, although they remain excluded from the primary model. `ID`, `DEFAULT_NEXT_MONTH` and derived category labels are excluded from the key. Each profile group is indivisible. A deterministic greedy allocation jointly minimises relative deviations from the target row counts and the target default/non-default counts. Persisted row assignments are used for every model. No resampling may be applied to validation or test data.

A temporal split is not defensible because all records describe the same historical window. The remediated Day 2 split contains 29,944 predictor-profile groups, including 52 repeated groups covering 108 rows. Automated validation confirms that zero predictor-profile groups cross partitions. Preprocessing and hyperparameter tuning must use training data only, with folds internal to training where required.

The accepted absolute tolerance for each row proportion is one percentage point around 60%/20%/20%. Default prevalence in each partition must remain within 0.5 percentage points of the full-dataset prevalence. The realised allocation is exactly 18,000/6,000/6,000 with default prevalence of 22.1222%/22.1167%/22.1167%; exact counts are an outcome of this dataset and seed, not a general requirement.

## Frozen threshold rule

On validation data, consider thresholds at which default-class recall is at least 70%. Select the threshold with the highest precision; if precision ties, choose the highest threshold deterministically. Record validation recall, precision, false-positive rate and proportion of customers flagged for review. Freeze the threshold before applying it to test predictions. The 70% target is an illustrative portfolio assumption, not a regulatory or lending standard.

## Frozen explainability scope

Produce global SHAP explainability and three clearly labelled case-level examples: one true positive, one false positive and one false negative, based on predictions at the frozen threshold. Case selection must not feed back into model or threshold choice. If an outcome category does not occur, report its absence rather than inventing or substituting an example.
