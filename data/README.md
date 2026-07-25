# Data acquisition and provenance

## Approved dataset

The primary dataset is **Default of Credit Card Clients**, created by I-Cheng Yeh and distributed by the UCI Machine Learning Repository.

- Official catalogue: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
- Dataset DOI: https://doi.org/10.24432/C55S3H
- Official archive: https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip
- UCI citation: Yeh, I. (2009). *Default of Credit Card Clients* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H
- Introductory paper: Yeh, I.-C. and Lien, C.-h. (2009), *The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients*, DOI: https://doi.org/10.1016/j.eswa.2007.12.020

UCI reports 30,000 instances, 23 explanatory features, no missing values in its catalogue metadata, and a binary default outcome. The records concern credit-card clients in Taiwan. Explanatory repayment, bill and payment histories cover April to September 2005; the outcome is default payment in the following month (October 2005).

## Licence and attribution

The official UCI catalogue marks the dataset as **Creative Commons Attribution 4.0 International (CC BY 4.0)**: https://creativecommons.org/licenses/by/4.0/

The licence permits sharing and adaptation, including commercial use, provided appropriate attribution is given, a link to the licence is supplied, and changes are indicated. This repository must retain the dataset citation and licence notice when redistributing either the raw data or adaptations. This is a project record of the published terms, not legal advice.

## Raw files

The following files were retrieved directly from the official UCI archive on 22 July 2026:

| File | Purpose | SHA-256 |
|---|---|---|
| `raw/default-of-credit-card-clients.zip` | Unmodified official download archive | `56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602` |
| `raw/default of credit card clients.xls` | Workbook extracted from the official archive | `30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933` |

Do not edit these files in place. Derived tables must be written under `data/processed/` and their transformation documented.

## Reproducible acquisition instructions

From the repository root:

```sh
curl --fail --location \
  --output data/raw/default-of-credit-card-clients.zip \
  'https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip'

shasum -a 256 data/raw/default-of-credit-card-clients.zip
unzip data/raw/default-of-credit-card-clients.zip -d data/raw
shasum -a 256 'data/raw/default of credit card clients.xls'
```

Compare the output with the checksums above. If UCI replaces the official file and a checksum changes, do not silently update this record: verify the UCI catalogue, record the retrieval date and new checksum, and describe the change.

No package installation is required to acquire the data. The UCI page also documents a `ucimlrepo` method, but this project retains the original archive for provenance.

## Supporting documentation

- `docs/DATA_DICTIONARY.md` defines the raw variables and prediction-time availability.
- `docs/LEAKAGE_REGISTER.md` records leakage risks and controls.
- `docs/DAY1_DATASET_SPEC.md` fixes the outcome timing and split design before modelling.

## Day 2 derived artefacts

Run `Rscript scripts/day2_prepare.R` from the repository root. The script verifies both raw checksums before reading the workbook and creates:

- `processed/credit_default_validated.csv`: canonical numeric raw fields plus separately labelled audit categories; original raw values are preserved.
- `processed/split_assignments.csv`: fixed-seed (`42`), group-aware, approximately stratified row assignments. All rows sharing the same values across the 23 original predictors are kept together; `ID` and the target do not form part of that profile key.
- `reports/day2_data_quality_summary.json`: machine-readable validation findings.
- Three training-only exploratory figures under `reports/figures/`.

The final test labels are used only to create and mechanically verify stratification. They are not used for exploratory feature or design decisions.
