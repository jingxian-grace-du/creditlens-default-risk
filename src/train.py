"""Day 3 leakage-safe Logistic Regression training workflow."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from src.data import APPROVED_PRIMARY_FEATURES, TARGET, load_split_assignments
from src.evaluate import (
    calculate_validation_metrics,
    save_coefficient_table,
    save_validation_curves,
    save_validation_metrics,
)
from src.preprocessing import build_logistic_baseline_pipeline, validate_primary_fields

DAY3_PARTITIONS = frozenset({"train", "validation"})


@dataclass(frozen=True)
class Day3Partitions:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_validation: pd.DataFrame
    y_validation: pd.Series
    training_ids: tuple[int, ...]
    validation_ids: tuple[int, ...]


def load_day3_partitions(project_root: str | Path = ".") -> Day3Partitions:
    """Load training and validation data without materialising final-test outcomes."""

    project_root = Path(project_root)
    data_path = project_root / "data/processed/credit_default_validated.csv"
    split_path = project_root / "data/processed/split_assignments.csv"
    assignments = load_split_assignments(split_path)

    identifiers = pd.read_csv(data_path, usecols=["ID"])["ID"].astype(int)
    if len(identifiers) != 30_000 or identifiers.nunique() != 30_000:
        raise ValueError("Validated dataset must contain exactly 30,000 unique IDs")
    if set(identifiers) != set(assignments):
        raise ValueError("Validated-data and assignment ID sets do not match exactly")

    # CSV row numbers start at 1 after the header. Rows assigned to final test are
    # skipped by the parser before the target-bearing modelling frame is created.
    skipped_test_rows = {
        row_number
        for row_number, identifier in enumerate(identifiers, start=1)
        if assignments[int(identifier)] == "test"
    }
    columns = ["ID", *APPROVED_PRIMARY_FEATURES, TARGET]
    modelling_data = pd.read_csv(
        data_path,
        usecols=columns,
        skiprows=lambda row_number: row_number in skipped_test_rows,
    )
    modelling_data["partition"] = modelling_data["ID"].astype(int).map(assignments)
    if set(modelling_data["partition"]) != DAY3_PARTITIONS:
        raise RuntimeError("Day 3 may materialise training and validation partitions only")
    validate_primary_fields(modelling_data.columns)

    train = modelling_data.loc[modelling_data["partition"] == "train"]
    validation = modelling_data.loc[modelling_data["partition"] == "validation"]
    return Day3Partitions(
        X_train=train.loc[:, list(APPROVED_PRIMARY_FEATURES)].copy(),
        y_train=train[TARGET].astype(int).copy(),
        X_validation=validation.loc[:, list(APPROVED_PRIMARY_FEATURES)].copy(),
        y_validation=validation[TARGET].astype(int).copy(),
        training_ids=tuple(train["ID"].astype(int)),
        validation_ids=tuple(validation["ID"].astype(int)),
    )


def fit_logistic_baseline(partitions: Day3Partitions):
    """Fit the fixed pipeline on training data only and return fit warnings."""

    pipeline = build_logistic_baseline_pipeline()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipeline.fit(partitions.X_train, partitions.y_train)
    convergence_warnings = [str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)]
    return pipeline, convergence_warnings


def run_day3_baseline(project_root: str | Path = ".") -> dict:
    """Train once, evaluate validation only, and persist approved Day 3 artefacts."""

    project_root = Path(project_root)
    partitions = load_day3_partitions(project_root)
    pipeline, convergence_warnings = fit_logistic_baseline(partitions)
    validation_probabilities = pipeline.predict_proba(partitions.X_validation)[:, 1]
    metrics = calculate_validation_metrics(partitions.y_validation, validation_probabilities)
    encoded_feature_count = len(pipeline.named_steps["preprocessor"].get_feature_names_out())
    metrics.update(
        {
            "model": "Logistic Regression",
            "training_rows": len(partitions.X_train),
            "source_feature_count": len(APPROVED_PRIMARY_FEATURES),
            "encoded_feature_count": encoded_feature_count,
            "configuration": {
                "penalty": "l2",
                "C": 1.0,
                "solver": "liblinear",
                "max_iter": 1000,
                "random_state": 42,
                "class_weight": None,
            },
            "convergence_warnings": convergence_warnings,
            "test_partition_evaluated": False,
        }
    )

    model_path = project_root / "models/day3_logistic_regression_baseline.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    metrics_path = project_root / "reports/day3_logistic_validation_metrics.json"
    save_validation_metrics(metrics, metrics_path)
    coefficient_path = project_root / "reports/day3_logistic_coefficients.csv"
    save_coefficient_table(pipeline, coefficient_path)
    roc_path, pr_path = save_validation_curves(
        partitions.y_validation,
        validation_probabilities,
        project_root / "reports/figures",
    )

    matrix = metrics["confusion_matrix_at_0_5"]["values"]
    report = f"""# Day 3 Logistic Regression baseline

## Scope and leakage controls

The persisted group-aware split was loaded without regeneration. Preprocessing and Logistic Regression were fitted on {len(partitions.X_train):,} training rows only. The {len(partitions.X_validation):,}-row validation partition was used only for baseline evaluation. Final-test rows were skipped before target-bearing data were materialised, and no final-test outcome was scored, summarised or inspected.

## Preprocessing

- Six repayment-status fields were treated as categorical and one-hot encoded with `handle_unknown=\"ignore\"`; distinct raw codes, including `-2` and `0`, were retained.
- Thirteen monetary fields were standardised using training-fitted means and scales.
- The pipeline used exactly 19 approved source fields and produced {encoded_feature_count} encoded features.
- No imputation, resampling, winsorisation, capping, outlier removal or outcome-informed category collapsing was performed.

## Fixed model configuration

Logistic Regression used L2 regularisation, `C=1.0`, `solver=\"liblinear\"`, `max_iter=1000`, `random_state=42` and no class weighting. No validation-led hyperparameter tuning was performed.

## Validation results

| Metric | Value |
|---|---:|
| ROC-AUC | {metrics['roc_auc']:.6f} |
| Average Precision (AP) | {metrics['average_precision']:.6f} |
| Precision at 0.5 | {metrics['precision_at_0_5']:.6f} |
| Recall at 0.5 | {metrics['recall_at_0_5']:.6f} |
| F1-score at 0.5 | {metrics['f1_at_0_5']:.6f} |
| Proportion flagged at 0.5 | {metrics['proportion_flagged_at_0_5']:.6f} |

Confusion matrix, with rows as actual `[0, 1]` and columns as predicted `[0, 1]`:

```text
{matrix[0]}
{matrix[1]}
```

The 0.5 threshold is for baseline reporting only. It is not the final business threshold, and the approved 70%-recall threshold rule has not been applied.

## Artefacts

- Pipeline: `models/day3_logistic_regression_baseline.joblib`
- Metrics: `reports/day3_logistic_validation_metrics.json`
- Coefficients: `reports/day3_logistic_coefficients.csv`
- ROC curve: `{roc_path.relative_to(project_root)}`
- Precision–recall curve: `{pr_path.relative_to(project_root)}`

Convergence warnings: {json.dumps(convergence_warnings)}
"""
    report_path = project_root / "reports/DAY3_LOGISTIC_BASELINE_REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    return metrics


if __name__ == "__main__":
    print(json.dumps(run_day3_baseline(), indent=2, sort_keys=True))
