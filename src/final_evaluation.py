"""One-time, locked final-test evaluation for CreditLens."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.data import APPROVED_PRIMARY_FEATURES, ORIGINAL_PREDICTORS, TARGET, load_split_assignments
from src.preprocessing import validate_primary_fields

LOCK_PATH = Path("reports/day7_final_decision_lock.json")
EXPECTED_MODEL_PATH = "models/day4_random_forest_pipeline.joblib"
EXPECTED_MODEL_SHA256 = "1d52eed5ae717e3e0bc09d319362263fcebcfb9f691a23951df32ac5c935037e"
FROZEN_THRESHOLD = 0.1912638431828811
EXPECTED_FINAL_TEST_ROWS = 6_000


@dataclass(frozen=True)
class FinalTestPartition:
    X_test: pd.DataFrame
    y_test: pd.Series
    test_ids: tuple[int, ...]


def sha256sum(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_decision_lock(project_root: str | Path = ".") -> dict[str, Any]:
    """Verify all immutable choices before any final-test outcome is loaded."""

    root = Path(project_root)
    lock = json.loads((root / LOCK_PATH).read_text(encoding="utf-8"))
    required = {
        "created_before_final_test_outcomes_loaded",
        "frozen_model_path",
        "frozen_model_sha256",
        "frozen_threshold",
        "selection_basis",
        "paired_rf_minus_xgboost_ap_uncertainty_interval",
        "no_superiority_claim",
        "post_test_change_prohibition",
    }
    missing = required - set(lock)
    if missing:
        raise ValueError(f"Decision lock is missing fields: {sorted(missing)}")
    if lock["created_before_final_test_outcomes_loaded"] is not True:
        raise RuntimeError("Decision lock was not recorded before final-test access")
    if lock["frozen_model_path"] != EXPECTED_MODEL_PATH:
        raise RuntimeError("Decision lock points to an unexpected model")
    if lock["frozen_threshold"] != FROZEN_THRESHOLD:
        raise RuntimeError("Decision lock threshold has changed")
    if lock["no_superiority_claim"] is not True:
        raise RuntimeError("Decision lock must prohibit a superiority claim")
    if lock["selection_basis"] != "highest numerical validation Average Precision":
        raise RuntimeError("Decision lock selection basis has changed")
    if lock["paired_rf_minus_xgboost_ap_uncertainty_interval"] != [
        -0.0075220545,
        0.0137943885,
    ]:
        raise RuntimeError("Decision lock uncertainty interval has changed")
    model_path = root / lock["frozen_model_path"]
    digest = sha256sum(model_path)
    if digest != EXPECTED_MODEL_SHA256 or digest != lock["frozen_model_sha256"]:
        raise RuntimeError("Frozen model hash does not match the decision lock")
    if "must not be changed" not in lock["post_test_change_prohibition"]:
        raise RuntimeError("Decision lock lacks the post-test change prohibition")
    return lock


def load_final_test_partition(project_root: str | Path = ".") -> FinalTestPartition:
    """Load exactly the persisted test partition after verifying the lock.

    The ID column and persisted assignments are read first. Training and
    validation rows are skipped before the target-bearing frame is materialised.
    """

    root = Path(project_root)
    verify_decision_lock(root)
    data_path = root / "data/processed/credit_default_validated.csv"
    assignments = load_split_assignments(
        root / "data/processed/split_assignments.csv"
    )
    identifiers = pd.read_csv(data_path, usecols=["ID"])["ID"].astype(int)
    if len(identifiers) != 30_000 or identifiers.nunique() != 30_000:
        raise ValueError("Validated dataset must contain 30,000 unique IDs")
    if set(identifiers) != set(assignments):
        raise ValueError("Validated-data and assignment ID sets do not match")

    skipped_non_test_rows = {
        row_number
        for row_number, identifier in enumerate(identifiers, start=1)
        if assignments[int(identifier)] != "test"
    }
    loaded = pd.read_csv(
        data_path,
        usecols=["ID", *ORIGINAL_PREDICTORS, TARGET],
        skiprows=lambda row_number: row_number in skipped_non_test_rows,
    )
    loaded["partition"] = loaded["ID"].astype(int).map(assignments)
    if len(loaded) != EXPECTED_FINAL_TEST_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_FINAL_TEST_ROWS} final-test rows, found {len(loaded)}"
        )
    if set(loaded["partition"]) != {"test"}:
        raise RuntimeError("The dedicated final loader materialised a non-test row")
    if loaded["ID"].nunique() != EXPECTED_FINAL_TEST_ROWS:
        raise RuntimeError("Final-test IDs are not unique")
    validate_primary_fields(loaded.columns)
    return FinalTestPartition(
        X_test=loaded.loc[:, list(APPROVED_PRIMARY_FEATURES)].copy(),
        y_test=loaded[TARGET].astype(int).copy(),
        test_ids=tuple(loaded["ID"].astype(int)),
    )


def calculate_final_metrics(y_true, probabilities) -> dict[str, Any]:
    """Calculate aggregate final-test metrics at the immutable threshold."""

    y = np.asarray(y_true, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    predictions = scores >= FROZEN_THRESHOLD
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    return {
        "partition": "final test",
        "rows": int(len(y)),
        "class_prevalence": float(np.mean(y)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "average_precision": float(average_precision_score(y, scores)),
        "frozen_threshold": FROZEN_THRESHOLD,
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions)),
        "f1": float(f1_score(y, predictions)),
        "false_positive_rate": float(fp / (fp + tn)),
        "proportion_flagged": float(np.mean(predictions)),
        "confusion_matrix": {
            "labels": [0, 1],
            "layout": "[[true_negative, false_positive], [false_negative, true_positive]]",
            "values": [[int(tn), int(fp)], [int(fn), int(tp)]],
        },
    }


def _save_figures(
    y_true,
    probabilities,
    metrics: dict[str, Any],
    figure_directory: Path,
) -> tuple[Path, Path, Path]:
    figure_directory.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(fpr, tpr, label=f"Random Forest (AUC = {metrics['roc_auc']:.3f})")
    axis.plot([0, 1], [0, 1], linestyle="--", color="grey")
    axis.set(
        xlabel="False-positive rate",
        ylabel="True-positive rate",
        title="Frozen Random Forest: final-test ROC",
    )
    axis.legend(loc="lower right")
    figure.tight_layout()
    roc_path = figure_directory / "day7_final_test_roc.png"
    figure.savefig(roc_path, dpi=160)
    plt.close(figure)

    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(
        recall,
        precision,
        label=f"Random Forest (AP = {metrics['average_precision']:.3f})",
    )
    axis.axhline(
        metrics["class_prevalence"],
        linestyle="--",
        color="grey",
        label="Class prevalence",
    )
    axis.set(
        xlabel="Recall",
        ylabel="Precision",
        title="Frozen Random Forest: final-test precision–recall",
    )
    axis.legend(loc="best")
    figure.tight_layout()
    pr_path = figure_directory / "day7_final_test_precision_recall.png"
    figure.savefig(pr_path, dpi=160)
    plt.close(figure)

    matrix = np.asarray(metrics["confusion_matrix"]["values"])
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["No default", "Default"],
    )
    display.plot(cmap="Blues", values_format="d")
    display.ax_.set_title(
        f"Final-test confusion matrix at threshold {FROZEN_THRESHOLD:.4f}"
    )
    display.figure_.tight_layout()
    confusion_path = figure_directory / "day7_final_test_confusion_matrix.png"
    display.figure_.savefig(confusion_path, dpi=160)
    plt.close(display.figure_)
    return roc_path, pr_path, confusion_path


def run_final_evaluation(project_root: str | Path = ".") -> dict[str, Any]:
    """Perform the authorised one-time evaluation of the frozen pipeline."""

    root = Path(project_root)
    lock = verify_decision_lock(root)
    model_path = root / lock["frozen_model_path"]
    model_hash_before = sha256sum(model_path)

    final_test = load_final_test_partition(root)
    frozen_model = joblib.load(model_path)
    probabilities = frozen_model.predict_proba(final_test.X_test)[:, 1]
    metrics = calculate_final_metrics(final_test.y_test, probabilities)
    model_hash_after = sha256sum(model_path)
    if model_hash_after != model_hash_before:
        raise RuntimeError("Frozen model changed during final evaluation")

    metrics.update(
        {
            "model": "Random Forest",
            "model_path": lock["frozen_model_path"],
            "model_sha256_before": model_hash_before,
            "model_sha256_after": model_hash_after,
            "decision_lock_verified_before_loading": True,
            "model_fitted_or_refitted": False,
            "model_tuned_or_recalibrated": False,
            "alternative_models_scored_on_final_test": False,
            "threshold_changed_after_test": False,
            "preprocessing_changed_after_test": False,
            "row_level_predictions_saved": False,
            "recall_floor_was_validation_target_not_test_guarantee": True,
            "post_test_decision_changed": False,
        }
    )
    reports = root / "reports"
    metrics_path = reports / "day7_final_test_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    validation_ranking = json.loads(
        (reports / "day4_random_forest_validation_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    validation_threshold = json.loads(
        (reports / "day6_selected_threshold.json").read_text(encoding="utf-8")
    )
    comparison_fields = {
        "roc_auc": (
            validation_ranking["roc_auc"],
            metrics["roc_auc"],
        ),
        "average_precision": (
            validation_ranking["average_precision"],
            metrics["average_precision"],
        ),
        "precision_at_frozen_threshold": (
            validation_threshold["precision"],
            metrics["precision"],
        ),
        "recall_at_frozen_threshold": (
            validation_threshold["recall"],
            metrics["recall"],
        ),
        "f1_at_frozen_threshold": (
            validation_threshold["f1"],
            metrics["f1"],
        ),
        "false_positive_rate": (
            validation_threshold["false_positive_rate"],
            metrics["false_positive_rate"],
        ),
        "proportion_flagged": (
            validation_threshold["proportion_flagged"],
            metrics["proportion_flagged"],
        ),
    }
    comparison = pd.DataFrame(
        [
            {
                "metric": field,
                "validation": values[0],
                "final_test": values[1],
                "final_minus_validation": values[1] - values[0],
            }
            for field, values in comparison_fields.items()
        ]
    )
    comparison_path = reports / "day7_validation_test_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    roc_path, pr_path, confusion_path = _save_figures(
        final_test.y_test,
        probabilities,
        metrics,
        reports / "figures",
    )

    matrix = metrics["confusion_matrix"]["values"]
    report = f"""# Day 7 one-time final-test evaluation

## Decision lock

Before any final-test outcomes were loaded, `reports/day7_final_decision_lock.json` froze `models/day4_random_forest_pipeline.joblib` at SHA-256 `{model_hash_before}`, its embedded preprocessing, and threshold `{FROZEN_THRESHOLD:.16f}`. Random Forest was selected because it had the highest numerical validation Average Precision. The paired Random-Forest-minus-XGBoost AP interval was `[-0.0075220545, 0.0137943885]`; no superiority claim was made. The lock prohibits model, preprocessing or threshold changes after viewing these results.

## Final-test isolation

The dedicated loader skipped training and validation rows before materialising the target-bearing final-test frame. Exactly {metrics['rows']:,} final-test rows were scored once using only the frozen Random Forest pipeline. No fitting, refitting, tuning, recalibration, retraining or alternative-model comparison was performed. No general row-level prediction file was saved.

## Aggregate final-test results

| Measure | Result |
|---|---:|
| Class prevalence | {metrics['class_prevalence']:.6f} |
| ROC-AUC | {metrics['roc_auc']:.6f} |
| Average Precision (AP) | {metrics['average_precision']:.6f} |
| Precision at frozen threshold | {metrics['precision']:.6f} |
| Recall at frozen threshold | {metrics['recall']:.6f} |
| F1 | {metrics['f1']:.6f} |
| False-positive rate | {metrics['false_positive_rate']:.6f} |
| Proportion flagged | {metrics['proportion_flagged']:.6f} |

Confusion matrix: `{matrix}`.

## Validation-to-test comparison

{comparison.to_markdown(index=False, floatfmt=".6f")}

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
"""
    report_path = reports / "DAY7_FINAL_EVALUATION_REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    return {
        "metrics": metrics,
        "comparison": comparison.to_dict(orient="records"),
        "artefacts": [
            str(metrics_path),
            str(comparison_path),
            str(report_path),
            str(roc_path),
            str(pr_path),
            str(confusion_path),
        ],
    }


if __name__ == "__main__":
    print(json.dumps(run_final_evaluation(), indent=2, sort_keys=True))
