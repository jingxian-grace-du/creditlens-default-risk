"""Day 4 bounded, group-aware Random Forest workflow."""

from __future__ import annotations

import csv
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold

from src.data import APPROVED_PRIMARY_FEATURES, ORIGINAL_PREDICTORS, TARGET, load_split_assignments
from src.evaluate import (
    calculate_validation_metrics,
    save_model_validation_curves,
    save_validation_metrics,
)
from src.preprocessing import (
    MONETARY_FIELDS,
    REPAYMENT_STATUS_FIELDS,
    build_random_forest_pipeline,
    validate_primary_fields,
)

DAY4_PARTITIONS = frozenset({"train", "validation"})
CV_FOLDS = 4
SEARCH_ITERATIONS = 12
PARAMETER_DISTRIBUTIONS = {
    "classifier__n_estimators": [300, 500],
    "classifier__max_depth": [None, 10, 20],
    "classifier__min_samples_split": [2, 10],
    "classifier__min_samples_leaf": [1, 5, 10],
    "classifier__max_features": ["sqrt", 0.5],
    "classifier__class_weight": [None, "balanced_subsample"],
}


@dataclass(frozen=True)
class Day4Partitions:
    X_train: pd.DataFrame
    y_train: pd.Series
    training_groups: pd.Series
    X_validation: pd.DataFrame
    y_validation: pd.Series
    training_ids: tuple[int, ...]
    validation_ids: tuple[int, ...]


def _profile_identifiers(frame: pd.DataFrame) -> pd.Series:
    """Create exact group labels from all 23 original predictor values."""

    missing = [field for field in ORIGINAL_PREDICTORS if field not in frame]
    if missing:
        raise ValueError(f"Missing grouping fields: {', '.join(missing)}")
    return frame.loc[:, list(ORIGINAL_PREDICTORS)].astype(str).agg("\x1f".join, axis=1)


def load_day4_partitions(project_root: str | Path = ".") -> Day4Partitions:
    """Load train/validation data while skipping final-test target rows."""

    project_root = Path(project_root)
    data_path = project_root / "data/processed/credit_default_validated.csv"
    assignments = load_split_assignments(project_root / "data/processed/split_assignments.csv")
    identifiers = pd.read_csv(data_path, usecols=["ID"])["ID"].astype(int)
    if len(identifiers) != 30_000 or identifiers.nunique() != 30_000:
        raise ValueError("Validated dataset must contain exactly 30,000 unique IDs")
    if set(identifiers) != set(assignments):
        raise ValueError("Validated-data and assignment ID sets do not match exactly")

    skipped_test_rows = {
        row_number
        for row_number, identifier in enumerate(identifiers, start=1)
        if assignments[int(identifier)] == "test"
    }
    loaded = pd.read_csv(
        data_path,
        usecols=["ID", *ORIGINAL_PREDICTORS, TARGET],
        skiprows=lambda row_number: row_number in skipped_test_rows,
    )
    loaded["partition"] = loaded["ID"].astype(int).map(assignments)
    if set(loaded["partition"]) != DAY4_PARTITIONS:
        raise RuntimeError("Day 4 may materialise training and validation partitions only")
    validate_primary_fields(loaded.columns)

    train = loaded.loc[loaded["partition"] == "train"].copy()
    validation = loaded.loc[loaded["partition"] == "validation"].copy()
    groups = _profile_identifiers(train)
    return Day4Partitions(
        X_train=train.loc[:, list(APPROVED_PRIMARY_FEATURES)].copy(),
        y_train=train[TARGET].astype(int).copy(),
        training_groups=groups.copy(),
        X_validation=validation.loc[:, list(APPROVED_PRIMARY_FEATURES)].copy(),
        y_validation=validation[TARGET].astype(int).copy(),
        training_ids=tuple(train["ID"].astype(int)),
        validation_ids=tuple(validation["ID"].astype(int)),
    )


def build_grouped_cv() -> StratifiedGroupKFold:
    return StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)


def audit_grouped_cv(cv, X, y, groups) -> list[dict[str, int]]:
    """Assert zero group overlap and return fold-level mechanical counts."""

    audits: list[dict[str, int]] = []
    validation_group_appearances: dict[str, int] = {}
    for fold, (fit_indices, validation_indices) in enumerate(cv.split(X, y, groups), start=1):
        fit_groups = set(groups.iloc[fit_indices])
        validation_groups = set(groups.iloc[validation_indices])
        overlap = fit_groups & validation_groups
        if overlap:
            raise RuntimeError(f"Predictor-profile groups cross CV fold {fold}")
        for group in validation_groups:
            validation_group_appearances[group] = validation_group_appearances.get(group, 0) + 1
        audits.append(
            {
                "fold": fold,
                "fit_rows": len(fit_indices),
                "validation_rows": len(validation_indices),
                "fit_groups": len(fit_groups),
                "validation_groups": len(validation_groups),
                "overlapping_groups": 0,
            }
        )
    if any(count != 1 for count in validation_group_appearances.values()):
        raise RuntimeError("Each training group must occur in exactly one internal validation fold")
    return audits


def build_search() -> RandomizedSearchCV:
    """Return the predeclared, bounded training-only search."""

    return RandomizedSearchCV(
        estimator=build_random_forest_pipeline(),
        param_distributions=PARAMETER_DISTRIBUTIONS,
        n_iter=SEARCH_ITERATIONS,
        scoring={"average_precision": "average_precision", "roc_auc": "roc_auc"},
        refit="average_precision",
        cv=build_grouped_cv(),
        random_state=42,
        n_jobs=1,
        return_train_score=False,
        verbose=1,
    )


def _encoded_sources(pipeline) -> list[str]:
    encoder = pipeline.named_steps["preprocessor"].named_transformers_["repayment_status"]
    sources = [
        field
        for field, categories in zip(REPAYMENT_STATUS_FIELDS, encoder.categories_)
        for _ in categories
    ]
    return [*sources, *MONETARY_FIELDS]


def save_importance_tables(pipeline, partitions: Day4Partitions, report_directory: Path) -> tuple[Path, Path]:
    """Save encoded impurity and source-level impurity/permutation importance."""

    report_directory.mkdir(parents=True, exist_ok=True)
    encoded_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    encoded_sources = _encoded_sources(pipeline)
    impurity = pipeline.named_steps["classifier"].feature_importances_
    if not (len(encoded_names) == len(encoded_sources) == len(impurity)):
        raise RuntimeError("Encoded feature-importance dimensions do not align")

    encoded_records = sorted(
        zip(encoded_names, encoded_sources, impurity),
        key=lambda item: float(item[2]),
        reverse=True,
    )
    encoded_path = report_directory / "day4_random_forest_encoded_importance.csv"
    with encoded_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination)
        writer.writerow(["encoded_feature", "source_feature", "impurity_importance"])
        for encoded, source, value in encoded_records:
            writer.writerow([encoded, source, f"{float(value):.12g}"])

    aggregated_impurity = {field: 0.0 for field in APPROVED_PRIMARY_FEATURES}
    for _, source, value in encoded_records:
        aggregated_impurity[source] += float(value)
    permutation = permutation_importance(
        pipeline,
        partitions.X_validation,
        partitions.y_validation,
        scoring="average_precision",
        n_repeats=5,
        random_state=42,
        n_jobs=1,
    )
    source_records = sorted(
        (
            (
                field,
                aggregated_impurity[field],
                float(permutation.importances_mean[position]),
                float(permutation.importances_std[position]),
            )
            for position, field in enumerate(APPROVED_PRIMARY_FEATURES)
        ),
        key=lambda item: item[2],
        reverse=True,
    )
    source_path = report_directory / "day4_random_forest_source_importance.csv"
    with source_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination)
        writer.writerow(
            ["source_feature", "aggregated_impurity_importance", "permutation_ap_mean", "permutation_ap_std"]
        )
        for record in source_records:
            writer.writerow([record[0], *(f"{value:.12g}" for value in record[1:])])
    return encoded_path, source_path


def _serialisable_parameter(value: Any) -> Any:
    return value if value is None or isinstance(value, (str, int, float, bool)) else str(value)


def run_day4_random_forest(
    project_root: str | Path = ".",
    *,
    resume_completed_search: bool = False,
) -> dict[str, Any]:
    """Run bounded training-only search, then one validation evaluation."""

    project_root = Path(project_root)
    reports = project_root / "reports"
    figures = reports / "figures"
    models = project_root / "models"
    reports.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)

    total_start = time.perf_counter()
    partitions = load_day4_partitions(project_root)
    cv_audit = audit_grouped_cv(
        build_grouped_cv(),
        partitions.X_train,
        partitions.y_train,
        partitions.training_groups,
    )
    model_path = models / "day4_random_forest_pipeline.joblib"
    metrics_path = reports / "day4_random_forest_validation_metrics.json"
    selected_params_path = reports / "day4_random_forest_selected_params.json"
    if resume_completed_search:
        required = [
            model_path,
            metrics_path,
            selected_params_path,
            reports / "day4_random_forest_search_results.csv",
        ]
        if not all(path.exists() for path in required):
            raise FileNotFoundError("Cannot resume: completed-search artefacts are missing")
        selected = joblib.load(model_path)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        selected_payload = json.loads(selected_params_path.read_text(encoding="utf-8"))
        if metrics.get("validation_used_for_hyperparameter_selection") is not False:
            raise RuntimeError("Saved search does not satisfy validation isolation")
        search_seconds = float(metrics["search_seconds"])
        search_warnings = list(metrics["search_warnings"])
        metrics["resource_notes"] = [
            "Initial sparse-input search attempt was interrupted for excessive runtime before validation.",
            "Completed dense-input search was retained; post-search permutation importance resumed serially after sandbox process creation was denied.",
        ]
    else:
        search = build_search()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            search_start = time.perf_counter()
            search.fit(
                partitions.X_train,
                partitions.y_train,
                groups=partitions.training_groups,
            )
            search_seconds = time.perf_counter() - search_start
        search_warnings = sorted({f"{item.category.__name__}: {item.message}" for item in caught})
        selected = search.best_estimator_

        validation_probabilities = selected.predict_proba(partitions.X_validation)[:, 1]
        metrics = calculate_validation_metrics(partitions.y_validation, validation_probabilities)
        metrics.update(
            {
                "model": "Random Forest",
                "training_rows": len(partitions.X_train),
                "source_feature_count": len(APPROVED_PRIMARY_FEATURES),
                "encoded_feature_count": len(selected.named_steps["preprocessor"].get_feature_names_out()),
                "search_iterations": SEARCH_ITERATIONS,
                "cv_folds": CV_FOLDS,
                "fitted_candidates": SEARCH_ITERATIONS * CV_FOLDS,
                "primary_search_metric": "average_precision",
                "search_seconds": search_seconds,
                "selected_hyperparameters": {
                    key.removeprefix("classifier__"): _serialisable_parameter(value)
                    for key, value in search.best_params_.items()
                },
                "search_warnings": search_warnings,
                "test_partition_evaluated": False,
                "validation_used_for_hyperparameter_selection": False,
            }
        )
        joblib.dump(selected, model_path)
        save_validation_metrics(metrics, metrics_path)

        results = pd.DataFrame(search.cv_results_)
        result_columns = [
            "rank_test_average_precision", "mean_test_average_precision",
            "std_test_average_precision", "mean_test_roc_auc", "std_test_roc_auc",
            "mean_fit_time", "std_fit_time",
            *sorted(column for column in results if column.startswith("param_")),
        ]
        results.loc[:, result_columns].sort_values("rank_test_average_precision").to_csv(
            reports / "day4_random_forest_search_results.csv", index=False
        )
        selected_payload = {
            "random_state": 42,
            "search_iterations": SEARCH_ITERATIONS,
            "cv_folds": CV_FOLDS,
            "fitted_candidates": SEARCH_ITERATIONS * CV_FOLDS,
            "search_partition": "training",
            "validation_used_for_selection": False,
            "primary_scoring": "average_precision",
            "secondary_scoring": "roc_auc",
            "best_cv_average_precision": float(search.best_score_),
            "search_seconds": search_seconds,
            "selected_hyperparameters": metrics["selected_hyperparameters"],
            "cv_group_audit": cv_audit,
            "search_warnings": search_warnings,
        }
        selected_params_path.write_text(
            json.dumps(selected_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    # Recompute probabilities only to complete curves after an infrastructure
    # recovery; model selection and the stored validation metrics stay fixed.
    validation_probabilities = selected.predict_proba(partitions.X_validation)[:, 1]

    encoded_path, source_path = save_importance_tables(selected, partitions, reports)
    roc_path, pr_path = save_model_validation_curves(
        partitions.y_validation,
        validation_probabilities,
        figures,
        file_prefix="day4_random_forest_validation",
        model_label="Random Forest",
    )

    logistic_metrics = json.loads(
        (reports / "day3_logistic_validation_metrics.json").read_text(encoding="utf-8")
    )
    comparison_fields = [
        "roc_auc", "average_precision", "precision_at_0_5", "recall_at_0_5",
        "f1_at_0_5", "proportion_flagged_at_0_5",
    ]
    comparison_path = reports / "day4_validation_model_comparison.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination)
        writer.writerow(["model", *comparison_fields])
        writer.writerow(["Logistic Regression", *(logistic_metrics[field] for field in comparison_fields)])
        writer.writerow(["Random Forest", *(metrics[field] for field in comparison_fields)])

    total_seconds = search_seconds + (time.perf_counter() - total_start)
    metrics["total_workflow_seconds"] = total_seconds
    save_validation_metrics(metrics, metrics_path)
    matrix = metrics["confusion_matrix_at_0_5"]["values"]
    params = metrics["selected_hyperparameters"]
    report = f"""# Day 4 Random Forest validation report

## Scope and isolation

The persisted grouped split was reused without regeneration. The bounded search used {len(partitions.X_train):,} training rows only. Hyperparameters were selected by four-fold `StratifiedGroupKFold` using exact group identifiers formed from all 23 original predictors. Audit-only variables contributed only to group identifiers and never entered the 19-field model matrix. Zero predictor-profile groups crossed internal folds.

The selected pipeline was evaluated once on {len(partitions.X_validation):,} validation rows. Final-test outcomes were not materialised, inspected, scored or summarised. These validation results are not final generalisation estimates.

## Preprocessing and search

Six repayment-status variables were one-hot encoded with `handle_unknown=\"ignore\"`. Thirteen monetary fields passed through without scaling. No imputation, resampling, capping, winsorisation or outlier removal was performed. All learned encoding was fitted inside the pipeline.

The deterministic search sampled {SEARCH_ITERATIONS} configurations and performed {SEARCH_ITERATIONS * CV_FOLDS} training-only CV fits. Average Precision was the refit metric and ROC-AUC was recorded secondarily. Search time was {search_seconds:.3f} seconds; total workflow time including validation permutation importance was {total_seconds:.3f} seconds.

Selected classifier hyperparameters:

```json
{json.dumps(params, indent=2, sort_keys=True)}
```

## Validation metrics

| Metric | Random Forest | Logistic Regression |
|---|---:|---:|
| ROC-AUC | {metrics['roc_auc']:.6f} | {logistic_metrics['roc_auc']:.6f} |
| Average Precision (AP) | {metrics['average_precision']:.6f} | {logistic_metrics['average_precision']:.6f} |
| Precision at 0.5 | {metrics['precision_at_0_5']:.6f} | {logistic_metrics['precision_at_0_5']:.6f} |
| Recall at 0.5 | {metrics['recall_at_0_5']:.6f} | {logistic_metrics['recall_at_0_5']:.6f} |
| F1-score at 0.5 | {metrics['f1_at_0_5']:.6f} | {logistic_metrics['f1_at_0_5']:.6f} |
| Proportion flagged at 0.5 | {metrics['proportion_flagged_at_0_5']:.6f} | {logistic_metrics['proportion_flagged_at_0_5']:.6f} |

Random Forest confusion matrix at reporting threshold 0.5:

```text
{matrix[0]}
{matrix[1]}
```

Threshold 0.5 remains a reporting convention, not the selected operating threshold. Metric differences on one validation partition do not by themselves establish model superiority or uncertainty.

## Importance limitations

Impurity importance can favour continuous or high-cardinality predictors and can distribute or inflate importance unpredictably among correlated variables. Source-level repayment importance aggregates its one-hot indicators. Validation permutation importance reports the change in Average Precision after shuffling each source field, but correlated predictors can mask one another and the result remains validation-specific. Neither measure is causal. SHAP was not performed.

## Artefacts

- Pipeline: `{model_path.relative_to(project_root)}`
- Search results: `reports/day4_random_forest_search_results.csv`
- Selected parameters and CV audit: `reports/day4_random_forest_selected_params.json`
- Metrics: `{metrics_path.relative_to(project_root)}`
- Encoded importance: `{encoded_path.relative_to(project_root)}`
- Source importance: `{source_path.relative_to(project_root)}`
- Model comparison: `{comparison_path.relative_to(project_root)}`
- ROC curve: `{roc_path.relative_to(project_root)}`
- Precision–recall curve: `{pr_path.relative_to(project_root)}`

Warnings captured during search: {json.dumps(search_warnings)}

## Resource-control notes

An initial sparse-input attempt was interrupted before validation because tree fitting was excessively slow. The completed search used dense one-hot output, which is memory-safe at 76 encoded features and does not change the encoded values or search space. After the completed search and saved validation evaluation, sandbox restrictions blocked process creation for parallel permutation importance; interpretation resumed from the saved selected model with serial permutation importance. The search was not repeated, hyperparameters and stored metrics remained fixed, and no final-test data were accessed.
"""
    (reports / "DAY4_RANDOM_FOREST_REPORT.md").write_text(report, encoding="utf-8")
    return metrics


if __name__ == "__main__":
    print(json.dumps(run_day4_random_forest(), indent=2, sort_keys=True))
