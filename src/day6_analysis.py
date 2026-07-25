"""Day 6 validation-only selection, thresholding, explanation and error analysis."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data import APPROVED_PRIMARY_FEATURES
from src.preprocessing import MONETARY_FIELDS, REPAYMENT_STATUS_FIELDS
from src.random_forest import _encoded_sources, load_day4_partitions

RANDOM_SEED = 42
RECALL_FLOOR = 0.70
BOOTSTRAP_ITERATIONS = 2_000
GLOBAL_SHAP_SAMPLE_SIZE = 1_000
CASE_LABELS = ("true_positive", "false_positive", "false_negative")


def paired_stratified_ap_bootstrap(
    y_true,
    first_probabilities,
    second_probabilities,
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    random_state: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Estimate paired AP-difference uncertainty while retaining class counts."""

    y = np.asarray(y_true, dtype=int)
    first = np.asarray(first_probabilities, dtype=float)
    second = np.asarray(second_probabilities, dtype=float)
    if not (len(y) == len(first) == len(second)):
        raise ValueError("Targets and both probability arrays must have equal length")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Paired AP bootstrap requires a binary target")
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    generator = np.random.default_rng(random_state)
    differences = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        sample = np.concatenate(
            [
                generator.choice(positive, len(positive), replace=True),
                generator.choice(negative, len(negative), replace=True),
            ]
        )
        differences[iteration] = average_precision_score(
            y[sample], first[sample]
        ) - average_precision_score(y[sample], second[sample])
    point_difference = average_precision_score(y, first) - average_precision_score(
        y, second
    )
    lower, upper = np.percentile(differences, [2.5, 97.5])
    return {
        "method": "paired stratified percentile bootstrap",
        "iterations": iterations,
        "random_state": random_state,
        "ap_difference_random_forest_minus_xgboost": float(point_difference),
        "confidence_level": 0.95,
        "confidence_interval_lower": float(lower),
        "confidence_interval_upper": float(upper),
        "proportion_bootstrap_differences_above_zero": float(
            np.mean(differences > 0)
        ),
        "interval_includes_zero": bool(lower <= 0 <= upper),
        "interpretation": (
            "uncertain/negligible; no superiority claim"
            if abs(point_difference) < 0.005 or lower <= 0 <= upper
            else "directionally separated on this validation bootstrap"
        ),
        "_differences": differences,
    }


def select_recall_constrained_threshold(
    y_true,
    probabilities,
    *,
    recall_floor: float = RECALL_FLOOR,
) -> dict[str, Any]:
    """Apply the frozen recall rule with a deterministic highest-threshold tie-break."""

    y = np.asarray(y_true, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    if len(y) != len(scores):
        raise ValueError("Targets and probabilities must have equal length")
    precision, recall, thresholds = precision_recall_curve(y, scores)
    qualifying = np.flatnonzero(recall[:-1] >= recall_floor)
    if not len(qualifying):
        raise RuntimeError("No observed threshold satisfies the recall floor")
    highest_precision = float(np.max(precision[qualifying]))
    tied = qualifying[precision[qualifying] == highest_precision]
    selected_index = int(tied[np.argmax(thresholds[tied])])
    threshold = float(thresholds[selected_index])
    predictions = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "recall_floor": recall_floor,
        "selection_partition": "validation",
        "selection_rule": (
            "Require default recall >= 70%; maximise precision; "
            "if precision ties, choose the highest threshold."
        ),
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
        "qualifying_threshold_count": int(len(qualifying)),
        "tie_count_at_highest_precision": int(len(tied)),
        "tie_break": "highest threshold",
        "illustrative_assumption": True,
        "regulatory_or_lending_standard": False,
        "test_partition_used": False,
    }


def _positive_class_shap_values(explainer, transformed) -> tuple[np.ndarray, float]:
    values = np.asarray(explainer.shap_values(transformed))
    if values.ndim == 3:
        values = values[:, :, 1]
    elif isinstance(values, list):
        values = np.asarray(values[1])
    expected = np.asarray(explainer.expected_value)
    base_value = float(expected[1] if expected.ndim else expected)
    return values, base_value


def _stratified_sample_indices(y_true, sample_size: int) -> np.ndarray:
    y = np.asarray(y_true, dtype=int)
    generator = np.random.default_rng(RANDOM_SEED)
    result: list[np.ndarray] = []
    for label in (0, 1):
        indices = np.flatnonzero(y == label)
        count = round(sample_size * len(indices) / len(y))
        result.append(generator.choice(indices, count, replace=False))
    return np.sort(np.concatenate(result))


def _aggregate_encoded_values(
    values: np.ndarray, sources: list[str]
) -> np.ndarray:
    positions = {
        field: [i for i, source in enumerate(sources) if source == field]
        for field in APPROVED_PRIMARY_FEATURES
    }
    return np.column_stack(
        [values[:, positions[field]].sum(axis=1) for field in APPROVED_PRIMARY_FEATURES]
    )


def _select_case_indices(y_true, probabilities, threshold: float) -> dict[str, int]:
    y = np.asarray(y_true, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    predicted = scores >= threshold
    masks = {
        "true_positive": (y == 1) & predicted,
        "false_positive": (y == 0) & predicted,
        "false_negative": (y == 1) & ~predicted,
    }
    selected: dict[str, int] = {}
    for label, mask in masks.items():
        candidates = np.flatnonzero(mask)
        if not len(candidates):
            continue
        if label in {"true_positive", "false_positive"}:
            selected[label] = int(candidates[np.argmax(scores[candidates])])
        else:
            selected[label] = int(candidates[np.argmin(scores[candidates])])
    return selected


def _save_threshold_figure(y, probabilities, threshold, output: Path) -> None:
    precision, recall, thresholds = precision_recall_curve(y, probabilities)
    qualifying = recall[:-1] >= RECALL_FLOOR
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(thresholds, precision[:-1], label="Precision")
    axis.plot(thresholds, recall[:-1], label="Recall")
    axis.axhline(RECALL_FLOOR, color="grey", linestyle="--", label="70% recall floor")
    axis.axvline(threshold, color="black", linestyle=":", label=f"Selected {threshold:.4f}")
    axis.scatter(
        thresholds[qualifying],
        precision[:-1][qualifying],
        s=4,
        alpha=0.15,
        label="Qualifying precision",
    )
    axis.set(xlabel="Probability threshold", ylabel="Validation metric", ylim=(0, 1))
    axis.set_title("Random Forest validation threshold analysis")
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _save_bootstrap_figure(differences: np.ndarray, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(differences, bins=40, color="#4F6D7A", alpha=0.85)
    axis.axvline(0, color="black", linestyle="--", label="No AP difference")
    axis.axvline(np.mean(differences), color="#C44E52", label="Bootstrap mean")
    axis.set(
        xlabel="AP difference: Random Forest − XGBoost",
        ylabel="Bootstrap samples",
        title="Paired validation AP uncertainty",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _save_error_outputs(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    reports: Path,
    figures: Path,
) -> tuple[Path, Path, Path]:
    predicted = probabilities >= threshold
    labels = np.select(
        [
            (y_true == 1) & predicted,
            (y_true == 0) & predicted,
            (y_true == 1) & ~predicted,
        ],
        ["true_positive", "false_positive", "false_negative"],
        default="true_negative",
    )
    summary_records = []
    for label in ("true_positive", "false_positive", "false_negative", "true_negative"):
        mask = labels == label
        summary_records.append(
            {
                "outcome_group": label,
                "rows": int(mask.sum()),
                "proportion_of_validation": float(mask.mean()),
                "mean_probability": float(probabilities[mask].mean()),
                "median_probability": float(np.median(probabilities[mask])),
            }
        )
    summary_path = reports / "day6_error_group_summary.csv"
    pd.DataFrame(summary_records).to_csv(summary_path, index=False)

    feature_records = []
    for label in ("false_positive", "false_negative"):
        mask = labels == label
        for field in APPROVED_PRIMARY_FEATURES:
            values = frame.loc[mask, field]
            feature_records.append(
                {
                    "error_group": label,
                    "source_feature": field,
                    "rows": int(mask.sum()),
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "most_frequent_value": float(values.mode().iloc[0]),
                    "validation_mean": float(frame[field].mean()),
                    "validation_median": float(frame[field].median()),
                }
            )
    feature_path = reports / "day6_error_feature_summary.csv"
    pd.DataFrame(feature_records).to_csv(feature_path, index=False)

    figure, axis = plt.subplots(figsize=(8, 5))
    counts = pd.Series(labels).value_counts().reindex(
        ["true_positive", "false_positive", "false_negative", "true_negative"]
    )
    axis.bar(counts.index, counts.values, color=["#4C72B0", "#DD8452", "#C44E52", "#55A868"])
    axis.set(ylabel="Validation rows", title="Outcomes at the selected validation threshold")
    axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    figure_path = figures / "day6_validation_error_counts.png"
    figure.savefig(figure_path, dpi=160)
    plt.close(figure)
    return summary_path, feature_path, figure_path


def run_day6(project_root: str | Path = ".") -> dict[str, Any]:
    """Run Day 6 from frozen models using validation data only."""

    started = time.perf_counter()
    root = Path(project_root)
    reports = root / "reports"
    figures = reports / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    partitions = load_day4_partitions(root)
    models = {
        "Logistic Regression": joblib.load(
            root / "models/day3_logistic_regression_baseline.joblib"
        ),
        "Random Forest": joblib.load(root / "models/day4_random_forest_pipeline.joblib"),
        "XGBoost": joblib.load(root / "models/day5_xgboost_pipeline.joblib"),
    }
    probabilities = {
        name: model.predict_proba(partitions.X_validation)[:, 1]
        for name, model in models.items()
    }
    y = partitions.y_validation.to_numpy()
    comparison = pd.DataFrame(
        [
            {
                "model": name,
                "roc_auc": roc_auc_score(y, scores),
                "average_precision": average_precision_score(y, scores),
                "selection_status": (
                    "provisional primary candidate"
                    if name == "Random Forest"
                    else "comparison model"
                ),
            }
            for name, scores in probabilities.items()
        ]
    ).sort_values("average_precision", ascending=False)
    comparison_path = reports / "day6_validation_model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    uncertainty = paired_stratified_ap_bootstrap(
        y, probabilities["Random Forest"], probabilities["XGBoost"]
    )
    bootstrap_values = uncertainty.pop("_differences")
    uncertainty_path = reports / "day6_paired_ap_uncertainty.json"
    uncertainty_path.write_text(
        json.dumps(uncertainty, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _save_bootstrap_figure(
        bootstrap_values, figures / "day6_paired_ap_bootstrap.png"
    )

    threshold = select_recall_constrained_threshold(
        y, probabilities["Random Forest"]
    )
    threshold.update(
        {
            "model": "Random Forest",
            "validation_rows": len(y),
            "final_test_outcomes_materialised": False,
            "final_test_evaluated": False,
        }
    )
    threshold_path = reports / "day6_selected_threshold.json"
    threshold_path.write_text(
        json.dumps(threshold, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _save_threshold_figure(
        y,
        probabilities["Random Forest"],
        threshold["threshold"],
        figures / "day6_threshold_tradeoff.png",
    )

    random_forest = models["Random Forest"]
    preprocessor = random_forest.named_steps["preprocessor"]
    classifier = random_forest.named_steps["classifier"]
    encoded_names = list(preprocessor.get_feature_names_out())
    sources = _encoded_sources(random_forest)
    sample_indices = _stratified_sample_indices(y, GLOBAL_SHAP_SAMPLE_SIZE)
    sample_frame = partitions.X_validation.iloc[sample_indices]
    transformed_sample = preprocessor.transform(sample_frame)
    explainer = shap.TreeExplainer(classifier)
    encoded_shap, base_value = _positive_class_shap_values(
        explainer, transformed_sample
    )
    source_shap = _aggregate_encoded_values(encoded_shap, sources)

    encoded_global = pd.DataFrame(
        {
            "encoded_feature": encoded_names,
            "source_feature": sources,
            "mean_absolute_shap": np.abs(encoded_shap).mean(axis=0),
            "mean_signed_shap": encoded_shap.mean(axis=0),
        }
    ).sort_values("mean_absolute_shap", ascending=False)
    encoded_global_path = reports / "day6_random_forest_global_shap_encoded.csv"
    encoded_global.to_csv(encoded_global_path, index=False)
    source_global = pd.DataFrame(
        {
            "source_feature": APPROVED_PRIMARY_FEATURES,
            "mean_absolute_shap": np.abs(source_shap).mean(axis=0),
            "mean_signed_shap": source_shap.mean(axis=0),
        }
    ).sort_values("mean_absolute_shap", ascending=False)
    source_global_path = reports / "day6_random_forest_global_shap_source.csv"
    source_global.to_csv(source_global_path, index=False)

    top = source_global.head(15).sort_values("mean_absolute_shap")
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.barh(top["source_feature"], top["mean_absolute_shap"], color="#4C72B0")
    axis.set(
        xlabel="Mean absolute SHAP value (default probability contribution)",
        title="Random Forest global SHAP importance",
    )
    figure.tight_layout()
    global_bar_path = figures / "day6_random_forest_global_shap_bar.png"
    figure.savefig(global_bar_path, dpi=160)
    plt.close(figure)

    explanation = shap.Explanation(
        values=source_shap,
        base_values=np.repeat(base_value, len(sample_frame)),
        data=sample_frame.loc[:, list(APPROVED_PRIMARY_FEATURES)].to_numpy(),
        feature_names=list(APPROVED_PRIMARY_FEATURES),
    )
    shap.plots.beeswarm(explanation, max_display=15, show=False)
    plt.title("Random Forest validation SHAP distribution")
    plt.tight_layout()
    beeswarm_path = figures / "day6_random_forest_global_shap_beeswarm.png"
    plt.savefig(beeswarm_path, dpi=160, bbox_inches="tight")
    plt.close()

    case_indices = _select_case_indices(
        y, probabilities["Random Forest"], threshold["threshold"]
    )
    if set(case_indices) != set(CASE_LABELS):
        raise RuntimeError("Validation does not contain all three required case types")
    case_frame = partitions.X_validation.iloc[list(case_indices.values())]
    case_transformed = preprocessor.transform(case_frame)
    case_encoded_shap, case_base = _positive_class_shap_values(
        explainer, case_transformed
    )
    case_source_shap = _aggregate_encoded_values(case_encoded_shap, sources)
    case_records = []
    case_metadata = []
    for position, (label, row_index) in enumerate(case_indices.items()):
        probability = float(probabilities["Random Forest"][row_index])
        actual = int(y[row_index])
        case_metadata.append(
            {
                "case_label": label,
                "actual_class": actual,
                "predicted_class": int(probability >= threshold["threshold"]),
                "predicted_probability": probability,
                "threshold": threshold["threshold"],
                "selection_method": (
                    "highest score in category"
                    if label != "false_negative"
                    else "lowest score in category"
                ),
            }
        )
        for field_position, field in enumerate(APPROVED_PRIMARY_FEATURES):
            case_records.append(
                {
                    "case_label": label,
                    "source_feature": field,
                    "raw_value": float(
                        partitions.X_validation.iloc[row_index][field]
                    ),
                    "shap_value": float(case_source_shap[position, field_position]),
                    "absolute_shap_value": float(
                        abs(case_source_shap[position, field_position])
                    ),
                }
            )
        encoded_explanation = shap.Explanation(
            values=case_encoded_shap[position],
            base_values=case_base,
            data=case_transformed[position],
            feature_names=encoded_names,
        )
        shap.plots.waterfall(encoded_explanation, max_display=12, show=False)
        plt.title(label.replace("_", " ").title())
        plt.tight_layout()
        plt.savefig(
            figures / f"day6_case_{label}_shap_waterfall.png",
            dpi=160,
            bbox_inches="tight",
        )
        plt.close()
    cases_path = reports / "day6_case_level_shap.csv"
    pd.DataFrame(case_records).sort_values(
        ["case_label", "absolute_shap_value"], ascending=[True, False]
    ).to_csv(cases_path, index=False)
    case_metadata_path = reports / "day6_case_level_metadata.json"
    case_metadata_path.write_text(
        json.dumps(case_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    error_summary, error_features, error_figure = _save_error_outputs(
        partitions.X_validation,
        y,
        probabilities["Random Forest"],
        threshold["threshold"],
        reports,
        figures,
    )
    error_feature_table = pd.read_csv(error_features).set_index(
        ["error_group", "source_feature"]
    )
    fp_pay0 = error_feature_table.loc[("false_positive", "PAY_0")]
    fn_pay0 = error_feature_table.loc[("false_negative", "PAY_0")]
    fp_limit = error_feature_table.loc[("false_positive", "LIMIT_BAL")]
    fn_bill1 = error_feature_table.loc[("false_negative", "BILL_AMT1")]

    metrics = {
        "analysis_partition": "validation",
        "provisional_primary_candidate": "Random Forest",
        "selection_basis": "highest validation Average Precision",
        "random_forest_validation_average_precision": float(
            comparison.loc[
                comparison["model"] == "Random Forest", "average_precision"
            ].iloc[0]
        ),
        "xgboost_validation_average_precision": float(
            comparison.loc[comparison["model"] == "XGBoost", "average_precision"].iloc[
                0
            ]
        ),
        "paired_uncertainty_interval_includes_zero": uncertainty[
            "interval_includes_zero"
        ],
        "superiority_claimed": False,
        "operating_threshold": threshold["threshold"],
        "threshold_precision": threshold["precision"],
        "threshold_recall": threshold["recall"],
        "threshold_f1": threshold["f1"],
        "threshold_false_positive_rate": threshold["false_positive_rate"],
        "threshold_proportion_flagged": threshold["proportion_flagged"],
        "threshold_confusion_matrix": threshold["confusion_matrix"],
        "global_shap_sample_rows": len(sample_frame),
        "global_shap_sampling": "deterministic class-stratified validation sample",
        "case_labels": list(case_indices),
        "shap_version": shap.__version__,
        "final_test_outcomes_materialised": False,
        "final_test_evaluated": False,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    metrics_path = reports / "day6_validation_analysis_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    report = f"""# Day 6 validation-only selection and explanation report

## Scope and isolation

Day 6 reuses the three frozen fitted pipelines and the persisted grouped split. It materialises training and validation partitions through the established safe loader, but performs all Day 6 scoring, thresholding, uncertainty analysis, SHAP explanation and error analysis on the {len(y):,}-row validation partition only. Final-test outcomes were not loaded, materialised, inspected, scored or summarised. These findings are not final generalisation performance.

## Provisional model selection and paired uncertainty

Random Forest remains the provisional primary candidate because its validation AP ({metrics['random_forest_validation_average_precision']:.6f}) is numerically above XGBoost ({metrics['xgboost_validation_average_precision']:.6f}). The paired AP difference is {uncertainty['ap_difference_random_forest_minus_xgboost']:+.6f}. A {uncertainty['iterations']:,}-iteration fixed-seed paired stratified bootstrap gives a 95% percentile interval of [{uncertainty['confidence_interval_lower']:.6f}, {uncertainty['confidence_interval_upper']:.6f}], which includes zero. The difference remains negligible and uncertain; no superiority claim is made and no final production model is selected.

## Frozen operating-threshold rule

On validation only, the pre-approved rule requires default recall of at least 70%, maximises precision among qualifying observed thresholds, and resolves an exact precision tie by taking the highest threshold. It selects `{threshold['threshold']:.12f}`.

| Measure | Validation result |
|---|---:|
| Precision | {threshold['precision']:.6f} |
| Recall | {threshold['recall']:.6f} |
| F1 | {threshold['f1']:.6f} |
| False-positive rate | {threshold['false_positive_rate']:.6f} |
| Proportion flagged | {threshold['proportion_flagged']:.6f} |

Confusion matrix: `{threshold['confusion_matrix']['values']}`.

The 70% recall floor is an illustrative portfolio assumption, not a regulatory or real lending standard. This threshold is frozen from validation and must not be revised using the final test set.

## Explainability

Tree SHAP explains the frozen Random Forest. Global summaries use a deterministic class-stratified sample of {len(sample_frame):,} validation rows for bounded runtime. One-hot contributions are retained at encoded level and also summed to the 19 source variables. Exactly three post-threshold cases are shown: one true positive, one false positive and one false negative. They were selected deterministically after threshold selection and did not influence model or threshold choice.

SHAP values describe model associations, not causal effects. Correlated billing, payment and repayment-status fields can divide or mask contributions; one-hot aggregation aids readability but does not remove dependence or proxy risks. Individual explanations are examples, not representative customer narratives or lending reasons.

## Error analysis

At the selected threshold there are {threshold['confusion_matrix']['values'][0][1]:,} false positives and {threshold['confusion_matrix']['values'][1][0]:,} false negatives. Aggregate feature summaries compare those groups with the complete validation partition without publishing a general row-level prediction file. The analysis is descriptive and validation-specific.

False positives have a higher mean `PAY_0` code ({fp_pay0['mean']:.3f}) and a lower median credit limit (£{fp_limit['median']:,.0f}) than the full validation sample ({fp_pay0['validation_mean']:.3f} and £{fp_limit['validation_median']:,.0f}, respectively). False negatives have a lower mean `PAY_0` code ({fn_pay0['mean']:.3f}) while their median first statement balance (£{fn_bill1['median']:,.0f}) exceeds the validation median (£{fn_bill1['validation_median']:,.0f}). These are aggregate associations: raw repayment codes are ordinal labels with special values, and correlated financial fields prevent causal or individual-level conclusions.

## Artefacts

- Metrics: `reports/day6_validation_analysis_metrics.json`
- Threshold: `reports/day6_selected_threshold.json`
- Paired uncertainty: `reports/day6_paired_ap_uncertainty.json`
- Model comparison: `reports/day6_validation_model_comparison.csv`
- Global SHAP: `reports/day6_random_forest_global_shap_encoded.csv`, `reports/day6_random_forest_global_shap_source.csv`
- Case explanations: `reports/day6_case_level_metadata.json`, `reports/day6_case_level_shap.csv`
- Error summaries: `{error_summary.relative_to(root)}`, `{error_features.relative_to(root)}`
- Figures: `reports/figures/day6_*.png`
"""
    report_path = reports / "DAY6_VALIDATION_SELECTION_EXPLAINABILITY_REPORT.md"
    report_path.write_text(report, encoding="utf-8")

    return {
        "metrics": metrics,
        "threshold": threshold,
        "uncertainty": uncertainty,
        "artefacts": [
            str(path)
            for path in (
                comparison_path,
                uncertainty_path,
                threshold_path,
                metrics_path,
                encoded_global_path,
                source_global_path,
                cases_path,
                case_metadata_path,
                error_summary,
                error_features,
                error_figure,
                global_bar_path,
                beeswarm_path,
                report_path,
            )
        ],
    }


if __name__ == "__main__":
    print(json.dumps(run_day6(), indent=2, sort_keys=True))
