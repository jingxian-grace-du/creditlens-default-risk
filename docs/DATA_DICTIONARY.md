# Data dictionary

Source: UCI Machine Learning Repository dataset 350, **Default of Credit Card Clients**. Monetary values are in New Taiwan dollars (NT$). The raw workbook uses `X1`–`X23` and `Y`; the descriptive names below follow UCI's variable table.

The prediction cut-off is the end of September 2005. “Available” below means present in the historical dataset by that cut-off, not necessarily lawful or appropriate for use in a real lending decision.

| Raw field | Descriptive name | Type | Definition | Timing | Prediction-time status |
|---|---|---|---|---|---|
| `ID` | Customer identifier | Identifier | Record identifier; not a predictive feature. | Static | Exclude; retain only for integrity and split checks. |
| `X1` | `LIMIT_BAL` | Numeric | Granted credit amount, including individual and supplementary/family credit. | At or before September 2005 | Available. |
| `X2` | `SEX` | Categorical | Sex: 1 male, 2 female. | Static | Audit only; excluded from primary version-1 models. |
| `X3` | `EDUCATION` | Categorical | Education: 1 graduate school, 2 university, 3 high school, 4 other. | Static | Audit only; excluded from primary version-1 models. |
| `X4` | `MARRIAGE` | Categorical | Marital status: 1 married, 2 single, 3 other. | Static | Audit only; excluded from primary version-1 models. |
| `X5` | `AGE` | Numeric | Age in years. | At observation time | Audit only; excluded from primary version-1 models. |
| `X6` | `PAY_0` | Ordinal categorical | Repayment status in September 2005. | September 2005 | Available at end-of-September cut-off. |
| `X7` | `PAY_2` | Ordinal categorical | Repayment status in August 2005. | August 2005 | Available. |
| `X8` | `PAY_3` | Ordinal categorical | Repayment status in July 2005. | July 2005 | Available. |
| `X9` | `PAY_4` | Ordinal categorical | Repayment status in June 2005. | June 2005 | Available. |
| `X10` | `PAY_5` | Ordinal categorical | Repayment status in May 2005. | May 2005 | Available. |
| `X11` | `PAY_6` | Ordinal categorical | Repayment status in April 2005. | April 2005 | Available. |
| `X12` | `BILL_AMT1` | Numeric | Bill statement amount in September 2005. | September 2005 | Available at end-of-September cut-off. |
| `X13` | `BILL_AMT2` | Numeric | Bill statement amount in August 2005. | August 2005 | Available. |
| `X14` | `BILL_AMT3` | Numeric | Bill statement amount in July 2005. | July 2005 | Available. |
| `X15` | `BILL_AMT4` | Numeric | Bill statement amount in June 2005. | June 2005 | Available. |
| `X16` | `BILL_AMT5` | Numeric | Bill statement amount in May 2005. | May 2005 | Available. |
| `X17` | `BILL_AMT6` | Numeric | Bill statement amount in April 2005. | April 2005 | Available. |
| `X18` | `PAY_AMT1` | Numeric | Previous payment amount in September 2005. | September 2005 | Available at end-of-September cut-off. |
| `X19` | `PAY_AMT2` | Numeric | Previous payment amount in August 2005. | August 2005 | Available. |
| `X20` | `PAY_AMT3` | Numeric | Previous payment amount in July 2005. | July 2005 | Available. |
| `X21` | `PAY_AMT4` | Numeric | Previous payment amount in June 2005. | June 2005 | Available. |
| `X22` | `PAY_AMT5` | Numeric | Previous payment amount in May 2005. | May 2005 | Available. |
| `X23` | `PAY_AMT6` | Numeric | Previous payment amount in April 2005. | April 2005 | Available. |
| `Y` | `default payment next month` | Binary target | Default payment: 1 yes, 0 no. | October 2005, after cut-off | Never available as a feature. |

For `PAY_0` and `PAY_2`–`PAY_6`, UCI documents `-1` as paid duly and positive values `1`–`9` as months of payment delay, with `9` meaning nine months or more. Any additional codes observed in the workbook are not assigned a meaning here; they must be reported as undocumented and handled only after a documented data-quality decision.

All availability judgements are conditional on a retrospective end-of-September 2005 scoring point. They do not establish legal permissibility, operational availability in another institution, or fairness.

## Observed undocumented-code mappings

Raw columns are preserved unchanged in `data/processed/credit_default_validated.csv`. Additional `*_CATEGORY` columns apply the following mappings without consulting the target:

| Field(s) | Undocumented raw value | Full-dataset frequency | Derived category |
|---|---:|---:|---|
| `EDUCATION` | 0 | 14 | `Unknown/Other` |
| `EDUCATION` | 5 | 280 | `Unknown/Other` |
| `EDUCATION` | 6 | 51 | `Unknown/Other` |
| `MARRIAGE` | 0 | 54 | `Unknown/Other` |
| `PAY_0` | -2 / 0 | 2,759 / 14,737 | `Unknown/Other` |
| `PAY_2` | -2 / 0 | 3,782 / 15,730 | `Unknown/Other` |
| `PAY_3` | -2 / 0 | 4,085 / 15,764 | `Unknown/Other` |
| `PAY_4` | -2 / 0 | 4,348 / 16,455 | `Unknown/Other` |
| `PAY_5` | -2 / 0 | 4,546 / 16,947 | `Unknown/Other` |
| `PAY_6` | -2 / 0 | 4,895 / 16,286 | `Unknown/Other` |

UCI does not define these values in its catalogue description. Their meaning is therefore not inferred from outcome rates or unofficial conventions. Collapsing them into a derived label does not overwrite or discard their distinct raw values.

## Approved primary feature set

The primary version-1 models may use `LIMIT_BAL`, `PAY_0`, `PAY_2`–`PAY_6`, `BILL_AMT1`–`BILL_AMT6` and `PAY_AMT1`–`PAY_AMT6` (19 fields). `ID` and the target are never predictors. `SEX`, `AGE`, `MARRIAGE` and `EDUCATION` are audit-only. This exclusion is a responsible portfolio-demonstration choice, not proof of fairness or regulatory compliance.

For split grouping only, the predictor-profile key uses all 23 original explanatory fields (`X1`–`X23`). It therefore includes the four audit-only fields even though they are not primary model inputs. It excludes `ID`, `DEFAULT_NEXT_MONTH` and every derived `*_CATEGORY` field.
