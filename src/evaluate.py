"""Validation-only evaluation helpers for CreditLens models."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

REQUIRED_VALIDATION_METRICS = frozenset(
    {
        "roc_auc",
        "average_precision",
        "precision_at_0_5",
        "recall_at_0_5",
        "f1_at_0_5",
        "confusion_matrix_at_0_5",
        "proportion_flagged_at_0_5",
    }
)


def calculate_validation_metrics(y_true, probabilities, threshold: float = 0.5) -> dict[str, Any]:
    """Calculate predeclared validation metrics at the reporting threshold."""

    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    return {
        "partition": "validation",
        "reporting_threshold": threshold,
        "threshold_status": "Baseline reporting only; not an operating threshold.",
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "precision_at_0_5": float(precision_score(y_true, predictions, zero_division=0)),
        "recall_at_0_5": float(recall_score(y_true, predictions, zero_division=0)),
        "f1_at_0_5": float(f1_score(y_true, predictions, zero_division=0)),
        "confusion_matrix_at_0_5": {
            "labels": [0, 1],
            "values": matrix.tolist(),
            "layout": "[[true_negative, false_positive], [false_negative, true_positive]]",
        },
        "proportion_flagged_at_0_5": float(predictions.mean()),
        "validation_rows": int(len(y_true)),
    }


def save_validation_metrics(metrics: dict[str, Any], path: str | Path) -> None:
    """Write metrics as stable, machine-readable JSON."""

    missing = REQUIRED_VALIDATION_METRICS - metrics.keys()
    if missing:
        raise ValueError(f"Missing validation metrics: {sorted(missing)}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_validation_curves(y_true, probabilities, figure_directory: str | Path) -> tuple[Path, Path]:
    """Save ROC and precision–recall curves for validation data only."""

    return save_model_validation_curves(
        y_true,
        probabilities,
        figure_directory,
        file_prefix="day3_logistic_validation",
        model_label="Logistic Regression",
    )


def save_model_validation_curves(
    y_true,
    probabilities,
    figure_directory: str | Path,
    *,
    file_prefix: str,
    model_label: str,
) -> tuple[Path, Path]:
    """Save named ROC and precision–recall curves for validation data only."""

    figure_directory = Path(figure_directory)
    figure_directory.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, probabilities)
    roc_auc = roc_auc_score(y_true, probabilities)
    roc_path = figure_directory / f"{file_prefix}_roc.png"
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(fpr, tpr, label=f"{model_label} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#777777")
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate", title="Validation ROC curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(roc_path, dpi=160)
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    average_precision = average_precision_score(y_true, probabilities)
    pr_path = figure_directory / f"{file_prefix}_precision_recall.png"
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(recall, precision, label=f"{model_label} (AP = {average_precision:.3f})")
    ax.axhline(float(sum(y_true) / len(y_true)), linestyle="--", color="#777777", label="Default prevalence")
    ax.set(xlabel="Recall", ylabel="Precision", title="Validation precision–recall curve")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(pr_path, dpi=160)
    plt.close(fig)
    return roc_path, pr_path


def save_coefficient_table(pipeline, path: str | Path) -> int:
    """Save encoded names and fitted Logistic Regression coefficients."""

    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    coefficients = pipeline.named_steps["classifier"].coef_[0]
    records = sorted(
        zip(names, coefficients),
        key=lambda item: abs(float(item[1])),
        reverse=True,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination)
        writer.writerow(["encoded_feature", "coefficient", "absolute_coefficient"])
        for name, coefficient in records:
            writer.writerow([name, f"{float(coefficient):.12g}", f"{abs(float(coefficient)):.12g}"])
    return len(records)
