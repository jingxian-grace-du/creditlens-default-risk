# Day 2 data-quality report

## Integrity and structure

- The official ZIP and extracted XLS checksums match the Day 1 provenance record.
- The ZIP integrity test passes.
- The workbook contains sheet `Data`, two header rows and 30,000 data rows.
- The canonical dataset has 25 raw fields: `ID`, 23 explanatory variables and `DEFAULT_NEXT_MONTH`.
- All raw fields load as numeric after removing the two workbook header rows.
- IDs are the unique integers 1–30,000.
- There are no missing values and no exact duplicates when `ID` is included.
- Excluding `ID` but retaining the target, there are 35 exact duplicate row pairs (70 rows). This is a data-quality statistic, not the leakage-control grouping rule.
- Using all 23 original explanatory variables and excluding both `ID` and the target produces 29,944 predictor-profile groups. Fifty-two groups are repeated, covering 108 rows. The group-aware split keeps every group intact, and zero groups cross partition boundaries.

## Target balance

| Target | Meaning | Rows | Share |
|---:|---|---:|---:|
| 0 | No default | 23,364 | 77.88% |
| 1 | Default | 6,636 | 22.12% |

The default class is the minority class, at roughly one default to 3.5 non-defaults. No resampling has been performed.

## Missing, invalid and undocumented values

No missing values were detected. `SEX` contains only documented codes 1 and 2. The following UCI-undocumented codes were observed and mapped to derived `Unknown/Other` labels while preserving every original value:

- `EDUCATION`: 0 (14), 5 (280), 6 (51).
- `MARRIAGE`: 0 (54).
- `PAY_0`: -2 (2,759), 0 (14,737).
- `PAY_2`: -2 (3,782), 0 (15,730).
- `PAY_3`: -2 (4,085), 0 (15,764).
- `PAY_4`: -2 (4,348), 0 (16,455).
- `PAY_5`: -2 (4,546), 0 (16,947).
- `PAY_6`: -2 (4,895), 0 (16,286).

No meaning was inferred from target outcomes. The mapping is fully defined in the data dictionary and preparation script.

## Numerical ranges and anomalies

- `LIMIT_BAL` ranges from NT$10,000 to NT$1,000,000; there are no non-positive values.
- `AGE` ranges from 21 to 79; no values fall outside the broad 18–100 validity check.
- Bill statement fields contain negative values: between 590 and 688 records per monthly field. These may represent credits or adjustments, so they are flagged but not altered.
- Previous-payment amounts are never negative. Zero-payment counts range from 5,249 in `PAY_AMT1` to 7,173 in `PAY_AMT6`.
- Large monetary maxima are present, including `BILL_AMT3` of NT$1,664,089 and `PAY_AMT2` of NT$1,684,259. They are potential genuine extremes and are preserved for training-only preprocessing decisions.

No anomaly has been removed, capped or imputed during Day 2.

## Split integrity

Seed `42` produced the approved deterministic, group-aware, approximately stratified allocation. The algorithm jointly balances total rows and default/non-default counts without splitting predictor-profile groups:

| Split | No default | Default | Total | Default share |
|---|---:|---:|---:|---:|
| Training | 14,018 | 3,982 | 18,000 | 22.1222% |
| Validation | 4,673 | 1,327 | 6,000 | 22.1167% |
| Final test | 4,673 | 1,327 | 6,000 | 22.1167% |

The realised row proportions are 60.0000%, 20.0000% and 20.0000%; deviations from the targets are therefore zero for this run. Exact counts are not required by validation: each proportion may deviate by up to one absolute percentage point to preserve group integrity. Partition default prevalence may deviate by no more than 0.5 percentage points from the full-dataset prevalence of 22.1200%.

Test labels were consulted only to optimise stratification and mechanically verify balance. All exploratory figures use training rows only.

## Feature scope

The version-1 primary feature set contains 19 fields: credit limit, six repayment-status fields, six bill amounts and six previous-payment amounts. `SEX`, `AGE`, `MARRIAGE` and `EDUCATION` are retained only for descriptive and subgroup auditing. `ID` and the target are excluded. This does not establish fairness or regulatory compliance, and extensive fairness certification remains outside the MVP.
