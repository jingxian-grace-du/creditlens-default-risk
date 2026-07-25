"""Day 5 bounded, leakage-aware XGBoost workflow."""

from __future__ import annotations

import csv
import json
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from src.data import APPROVED_PRIMARY_FEATURES
from src.evaluate import (
    calculate_validation_metrics,
    save_model_validation_curves,
    save_validation_metrics,
)
from src.preprocessing import MONETARY_FIELDS, REPAYMENT_STATUS_FIELDS
from src.random_forest import (
    audit_grouped_cv,
    build_grouped_cv,
    load_day4_partitions,
)

CV_FOLDS = 4
SEARCH_ITERATIONS = 12
PARAMETER_DISTRIBUTIONS = {
    "classifier__n_estimators": [200, 400, 600],
    "classifier__max_depth": [3, 5, 7],
    "classifier__learning_rate": [0.03, 0.05, 0.10],
    "classifier__min_child_weight": [1, 5, 10],
    "classifier__subsample": [0.70, 0.85, 1.00],
    "classifier__colsample_bytree": [0.70, 0.85, 1.00],
    "classifier__gamma": [0.0, 0.5, 1.0],
    "classifier__reg_lambda": [1.0, 5.0, 10.0],
    "classifier__scale_pos_weight": [1.0, 3.5],
}


def build_xgboost_pipeline() -> Pipeline:
    """Build the fixed preprocessing boundary and untuned XGBoost classifier."""

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "repayment_status",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(REPAYMENT_STATUS_FIELDS),
            ),
            ("monetary", "passthrough", list(MONETARY_FIELDS)),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    classifier = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def build_search() -> RandomizedSearchCV:
    """Return the predeclared 12-configuration grouped search."""

    return RandomizedSearchCV(
        estimator=build_xgboost_pipeline(),
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


def _serialisable(value: Any) -> Any:
    return value if value is None or isinstance(value, (str, int, float, bool)) else str(value)


def _encoded_source_names(pipeline: Pipeline) -> list[str]:
    encoder = pipeline.named_steps["preprocessor"].named_transformers_["repayment_status"]
    categorical_sources = [
        field
        for field, categories in zip(REPAYMENT_STATUS_FIELDS, encoder.categories_)
        for _ in categories
    ]
    return [*categorical_sources, *MONETARY_FIELDS]


def save_gain_importance_tables(
    pipeline: Pipeline,
    report_directory: Path,
) -> tuple[Path, Path]:
    """Save encoded and 19-source gain importance with an explicit mapping."""

    encoded_names = list(pipeline.named_steps["preprocessor"].get_feature_names_out())
    source_names = _encoded_source_names(pipeline)
    if len(encoded_names) != len(source_names):
        raise RuntimeError("Encoded XGBoost feature names do not align with source mapping")

    booster_scores = pipeline.named_steps["classifier"].get_booster().get_score(
        importance_type="gain"
    )
    raw_gains = []
    for position in range(len(encoded_names)):
        raw_gains.append(float(booster_scores.get(f"f{position}", 0.0)))
    gain_total = sum(raw_gains)
    normalised = [
        value / gain_total if gain_total else 0.0
        for value in raw_gains
    ]

    encoded_records = sorted(
        zip(encoded_names, source_names, raw_gains, normalised),
        key=lambda record: record[2],
        reverse=True,
    )
    encoded_path = report_directory / "day5_xgboost_encoded_importance.csv"
    with encoded_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination)
        writer.writerow(
            ["encoded_feature", "source_feature", "gain", "normalised_gain"]
        )
        for encoded, source, gain, normalised_gain in encoded_records:
            writer.writerow(
                [encoded, source, f"{gain:.12g}", f"{normalised_gain:.12g}"]
            )

    source_gain = {field: 0.0 for field in APPROVED_PRIMARY_FEATURES}
    for _, source, gain, _ in encoded_records:
        source_gain[source] += gain
    source_total = sum(source_gain.values())
    source_records = sorted(
        (
            (
                field,
                gain,
                gain / source_total if source_total else 0.0,
            )
            for field, gain in source_gain.items()
        ),
        key=lambda record: record[1],
        reverse=True,
    )
    source_path = report_directory / "day5_xgboost_source_importance.csv"
    with source_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination)
        writer.writerow(["source_feature", "aggregated_gain", "normalised_gain"])
        for source, gain, normalised_gain in source_records:
            writer.writerow(
                [source, f"{gain:.12g}", f"{normalised_gain:.12g}"]
            )
    return encoded_path, source_path


def _improvement_description(xgboost_ap: float, random_forest_ap: float) -> str:
    """Return a cautious descriptive label, not a significance conclusion."""

    difference = xgboost_ap - random_forest_ap
    magnitude = abs(difference)
    if magnitude < 0.005:
        return "negligible"
    direction = "improvement" if difference > 0 else "decline"
    if magnitude < 0.020:
        return f"modest {direction}"
    return f"meaningful {direction}"


def run_day5_xgboost(project_root: str | Path = ".") -> dict[str, Any]:
    """Search on training only, then evaluate the selected model on validation."""

    project_root = Path(project_root)
    reports = project_root / "reports"
    figures = reports / "figures"
    models = project_root / "models"
    reports.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)

    workflow_start = time.perf_counter()
    partitions = load_day4_partitions(project_root)
    cv_audit = audit_grouped_cv(
        build_grouped_cv(),
        partitions.X_train,
        partitions.y_train,
        partitions.training_groups,
    )
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
    search_warnings = sorted(
        {f"{item.category.__name__}: {item.message}" for item in caught}
    )
    selected = search.best_estimator_

    # Validation is first scored only after the training-only search and refit.
    validation_probabilities = selected.predict_proba(partitions.X_validation)[:, 1]
    metrics = calculate_validation_metrics(
        partitions.y_validation,
        validation_probabilities,
    )
    best_index = int(search.best_index_)
    best_cv_roc_auc = float(search.cv_results_["mean_test_roc_auc"][best_index])
    selected_hyperparameters = {
        key.removeprefix("classifier__"): _serialisable(value)
        for key, value in search.best_params_.items()
    }
    metrics.update(
        {
            "model": "XGBoost",
            "xgboost_version": xgboost.__version__,
            "training_rows": len(partitions.X_train),
            "source_feature_count": len(APPROVED_PRIMARY_FEATURES),
            "encoded_feature_count": len(
                selected.named_steps["preprocessor"].get_feature_names_out()
            ),
            "search_iterations": SEARCH_ITERATIONS,
            "cv_folds": CV_FOLDS,
            "fitted_candidates": SEARCH_ITERATIONS * CV_FOLDS,
            "primary_search_metric": "average_precision",
            "best_cv_average_precision": float(search.best_score_),
            "best_cv_roc_auc": best_cv_roc_auc,
            "search_seconds": search_seconds,
            "selected_hyperparameters": selected_hyperparameters,
            "fixed_configuration": {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "tree_method": "hist",
                "random_state": 42,
                "n_jobs": -1,
            },
            "search_warnings": search_warnings,
            "test_partition_evaluated": False,
            "validation_used_for_hyperparameter_selection": False,
            "threshold_selected": False,
        }
    )

    model_path = models / "day5_xgboost_pipeline.joblib"
    joblib.dump(selected, model_path)
    metrics_path = reports / "day5_xgboost_validation_metrics.json"
    save_validation_metrics(metrics, metrics_path)

    search_results = pd.DataFrame(search.cv_results_)
    search_columns = [
        "rank_test_average_precision",
        "mean_test_average_precision",
        "std_test_average_precision",
        "mean_test_roc_auc",
        "std_test_roc_auc",
        "mean_fit_time",
        "std_fit_time",
        *sorted(
            column
            for column in search_results
            if column.startswith("param_")
        ),
    ]
    search_results.loc[:, search_columns].sort_values(
        "rank_test_average_precision"
    ).to_csv(reports / "day5_xgboost_search_results.csv", index=False)

    selected_payload = {
        "xgboost_version": xgboost.__version__,
        "random_state": 42,
        "search_iterations": SEARCH_ITERATIONS,
        "cv_folds": CV_FOLDS,
        "fitted_candidates": SEARCH_ITERATIONS * CV_FOLDS,
        "search_partition": "training",
        "validation_used_for_selection": False,
        "primary_scoring": "average_precision",
        "secondary_scoring": "roc_auc",
        "best_cv_average_precision": float(search.best_score_),
        "best_cv_roc_auc": best_cv_roc_auc,
        "search_seconds": search_seconds,
        "fixed_configuration": metrics["fixed_configuration"],
        "selected_hyperparameters": selected_hyperparameters,
        "parameter_distributions": PARAMETER_DISTRIBUTIONS,
        "cv_group_audit": cv_audit,
        "search_warnings": search_warnings,
    }
    selected_path = reports / "day5_xgboost_selected_params.json"
    selected_path.write_text(
        json.dumps(selected_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    encoded_path, source_path = save_gain_importance_tables(selected, reports)
    roc_path, pr_path = save_model_validation_curves(
        partitions.y_validation,
        validation_probabilities,
        figures,
        file_prefix="day5_xgboost_validation",
        model_label="XGBoost",
    )

    logistic = json.loads(
        (reports / "day3_logistic_validation_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    forest = json.loads(
        (reports / "day4_random_forest_validation_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    comparison_fields = [
        "roc_auc",
        "average_precision",
        "precision_at_0_5",
        "recall_at_0_5",
        "f1_at_0_5",
        "proportion_flagged_at_0_5",
    ]
    comparison_path = reports / "day5_validation_model_comparison.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination)
        writer.writerow(["model", *comparison_fields])
        writer.writerow(
            ["Logistic Regression", *(logistic[field] for field in comparison_fields)]
        )
        writer.writerow(
            ["Random Forest", *(forest[field] for field in comparison_fields)]
        )
        writer.writerow(["XGBoost", *(metrics[field] for field in comparison_fields)])

    workflow_seconds = time.perf_counter() - workflow_start
    metrics["total_workflow_seconds"] = workflow_seconds
    metrics["ap_improvement_over_random_forest"] = (
        metrics["average_precision"] - forest["average_precision"]
    )
    metrics["ap_improvement_description"] = _improvement_description(
        metrics["average_precision"],
        forest["average_precision"],
    )
    save_validation_metrics(metrics, metrics_path)

    matrix = metrics["confusion_matrix_at_0_5"]["values"]
    report = f"""# Day 5 XGBoost validation report

## Scope and isolation

The persisted grouped split was reused without regeneration. The bounded search used {len(partitions.X_train):,} training rows only. Four-fold shuffled `StratifiedGroupKFold` used exact group identifiers formed from all 23 original predictors, with audit-only variables serving solely as grouping metadata. No group crossed an internal fold.

The selected pipeline was evaluated on {len(partitions.X_validation):,} validation rows only after training-CV selection. Final-test target values were not materialised, inspected, scored or summarised. Validation results are not final generalisation estimates.

## Preprocessing and bounded search

The model matrix contains exactly the approved 19 predictors. Six repayment-status fields were one-hot encoded with `handle_unknown="ignore"` inside the pipeline. Thirteen monetary fields passed through without scaling, capping or outlier removal. No resampling or validation-led early stopping was used.

`RandomizedSearchCV` sampled {SEARCH_ITERATIONS} configurations and performed {SEARCH_ITERATIONS * CV_FOLDS} training-only fits. Average Precision was the refit metric and ROC-AUC was recorded secondarily. Search time was {search_seconds:.3f} seconds and total workflow time was {workflow_seconds:.3f} seconds.

Fixed XGBoost configuration:

```json
{json.dumps(metrics['fixed_configuration'], indent=2, sort_keys=True)}
```

Selected hyperparameters:

```json
{json.dumps(selected_hyperparameters, indent=2, sort_keys=True)}
```

Best training-only grouped-CV Average Precision: {search.best_score_:.6f}

Best configuration's grouped-CV ROC-AUC: {best_cv_roc_auc:.6f}

## Validation comparison

| Metric | Logistic Regression | Random Forest | XGBoost |
|---|---:|---:|---:|
| ROC-AUC | {logistic['roc_auc']:.6f} | {forest['roc_auc']:.6f} | {metrics['roc_auc']:.6f} |
| Average Precision (AP) | {logistic['average_precision']:.6f} | {forest['average_precision']:.6f} | {metrics['average_precision']:.6f} |
| Precision at 0.5 | {logistic['precision_at_0_5']:.6f} | {forest['precision_at_0_5']:.6f} | {metrics['precision_at_0_5']:.6f} |
| Recall at 0.5 | {logistic['recall_at_0_5']:.6f} | {forest['recall_at_0_5']:.6f} | {metrics['recall_at_0_5']:.6f} |
| F1-score at 0.5 | {logistic['f1_at_0_5']:.6f} | {forest['f1_at_0_5']:.6f} | {metrics['f1_at_0_5']:.6f} |
| Proportion flagged at 0.5 | {logistic['proportion_flagged_at_0_5']:.6f} | {forest['proportion_flagged_at_0_5']:.6f} | {metrics['proportion_flagged_at_0_5']:.6f} |

XGBoost confusion matrix at diagnostic threshold 0.5:

```text
{matrix[0]}
{matrix[1]}
```

Relative to Random Forest, the numerical AP change is {metrics['ap_improvement_over_random_forest']:+.6f}, described as {metrics['ap_improvement_description']} under the predeclared descriptive bands in code. This is not an uncertainty or significance conclusion and does not select a final model.

Threshold 0.5 is diagnostic only. No final model or business threshold was selected.

## Gain importance

Encoded-feature gain and source-aggregated gain are reported. Gain measures how much fitted tree splits improve the objective; correlated inputs can divide or mask importance, and importance is not causal. No permutation importance or SHAP analysis was performed on Day 5.

## Artefacts

- Pipeline: `{model_path.relative_to(project_root)}`
- Search results: `reports/day5_xgboost_search_results.csv`
- Selected parameters and CV audit: `{selected_path.relative_to(project_root)}`
- Metrics: `{metrics_path.relative_to(project_root)}`
- Encoded gain: `{encoded_path.relative_to(project_root)}`
- Source gain: `{source_path.relative_to(project_root)}`
- Model comparison: `{comparison_path.relative_to(project_root)}`
- ROC curve: `{roc_path.relative_to(project_root)}`
- Precision–recall curve: `{pr_path.relative_to(project_root)}`

Warnings captured during search: {json.dumps(search_warnings)}
"""
    (reports / "DAY5_XGBOOST_REPORT.md").write_text(report, encoding="utf-8")
    return metrics


if __name__ == "__main__":
    print(json.dumps(run_day5_xgboost(), indent=2, sort_keys=True))
