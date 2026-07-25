import json
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.data import APPROVED_PRIMARY_FEATURES, load_split_assignments
from src.day6_analysis import (
    BOOTSTRAP_ITERATIONS,
    CASE_LABELS,
    GLOBAL_SHAP_SAMPLE_SIZE,
    RECALL_FLOOR,
    _aggregate_encoded_values,
    paired_stratified_ap_bootstrap,
    select_recall_constrained_threshold,
)
from src.random_forest import _encoded_sources, load_day4_partitions


class Day6AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.partitions = load_day4_partitions()
        cls.forest = joblib.load("models/day4_random_forest_pipeline.joblib")
        cls.xgboost = joblib.load("models/day5_xgboost_pipeline.joblib")
        cls.metrics = json.loads(
            Path("reports/day6_validation_analysis_metrics.json").read_text(
                encoding="utf-8"
            )
        )
        cls.threshold = json.loads(
            Path("reports/day6_selected_threshold.json").read_text(encoding="utf-8")
        )
        cls.uncertainty = json.loads(
            Path("reports/day6_paired_ap_uncertainty.json").read_text(
                encoding="utf-8"
            )
        )

    def test_only_validation_is_scored_and_final_test_is_excluded(self) -> None:
        assignments = load_split_assignments()
        validation_ids = {
            identifier
            for identifier, partition in assignments.items()
            if partition == "validation"
        }
        test_ids = {
            identifier
            for identifier, partition in assignments.items()
            if partition == "test"
        }
        self.assertEqual(set(self.partitions.validation_ids), validation_ids)
        self.assertTrue(set(self.partitions.validation_ids).isdisjoint(test_ids))
        self.assertEqual(self.metrics["analysis_partition"], "validation")
        self.assertFalse(self.metrics["final_test_outcomes_materialised"])
        self.assertFalse(self.metrics["final_test_evaluated"])
        self.assertFalse(self.threshold["test_partition_used"])

    def test_frozen_threshold_rule_is_reproducible(self) -> None:
        probabilities = self.forest.predict_proba(
            self.partitions.X_validation
        )[:, 1]
        result = select_recall_constrained_threshold(
            self.partitions.y_validation, probabilities
        )
        self.assertEqual(result["recall_floor"], RECALL_FLOOR)
        self.assertAlmostEqual(
            result["threshold"], self.threshold["threshold"], places=15
        )
        self.assertGreaterEqual(result["recall"], RECALL_FLOOR)
        self.assertEqual(
            result["confusion_matrix"], self.threshold["confusion_matrix"]
        )
        self.assertEqual(result["tie_break"], "highest threshold")

    def test_threshold_tie_break_chooses_highest_threshold(self) -> None:
        y = np.array([1, 1, 0, 0])
        probabilities = np.array([0.9, 0.8, 0.7, 0.1])
        result = select_recall_constrained_threshold(
            y, probabilities, recall_floor=0.5
        )
        self.assertEqual(result["threshold"], 0.9)
        self.assertEqual(result["precision"], 1.0)
        self.assertEqual(result["recall"], 0.5)

    def test_paired_bootstrap_is_deterministic_and_paired(self) -> None:
        y = self.partitions.y_validation.to_numpy()
        forest = self.forest.predict_proba(self.partitions.X_validation)[:, 1]
        xgboost = self.xgboost.predict_proba(self.partitions.X_validation)[:, 1]
        first = paired_stratified_ap_bootstrap(
            y, forest, xgboost, iterations=50, random_state=42
        )
        second = paired_stratified_ap_bootstrap(
            y, forest, xgboost, iterations=50, random_state=42
        )
        self.assertAlmostEqual(
            first["ap_difference_random_forest_minus_xgboost"],
            average_precision_score(y, forest)
            - average_precision_score(y, xgboost),
        )
        self.assertEqual(
            first["confidence_interval_lower"], second["confidence_interval_lower"]
        )
        self.assertEqual(
            first["confidence_interval_upper"], second["confidence_interval_upper"]
        )

    def test_observed_uncertainty_remains_negligible(self) -> None:
        self.assertEqual(self.uncertainty["iterations"], BOOTSTRAP_ITERATIONS)
        self.assertAlmostEqual(
            self.uncertainty["ap_difference_random_forest_minus_xgboost"],
            0.003336209474033791,
        )
        self.assertTrue(self.uncertainty["interval_includes_zero"])
        self.assertFalse(self.metrics["superiority_claimed"])

    def test_global_shap_tables_have_correct_feature_scope(self) -> None:
        encoded = pd.read_csv(
            "reports/day6_random_forest_global_shap_encoded.csv"
        )
        source = pd.read_csv("reports/day6_random_forest_global_shap_source.csv")
        self.assertEqual(len(encoded), 76)
        self.assertEqual(len(source), 19)
        self.assertEqual(
            set(source["source_feature"]), set(APPROVED_PRIMARY_FEATURES)
        )
        self.assertEqual(
            set(encoded["source_feature"]), set(APPROVED_PRIMARY_FEATURES)
        )
        self.assertEqual(
            self.metrics["global_shap_sample_rows"], GLOBAL_SHAP_SAMPLE_SIZE
        )

    def test_encoded_to_source_shap_aggregation(self) -> None:
        sources = _encoded_sources(self.forest)
        values = np.arange(2 * len(sources), dtype=float).reshape(2, len(sources))
        aggregated = _aggregate_encoded_values(values, sources)
        self.assertEqual(aggregated.shape, (2, 19))
        for position, field in enumerate(APPROVED_PRIMARY_FEATURES):
            expected = values[
                :, [i for i, source in enumerate(sources) if source == field]
            ].sum(axis=1)
            np.testing.assert_array_equal(aggregated[:, position], expected)

    def test_exactly_three_labelled_case_explanations_exist(self) -> None:
        metadata = json.loads(
            Path("reports/day6_case_level_metadata.json").read_text(
                encoding="utf-8"
            )
        )
        cases = pd.read_csv("reports/day6_case_level_shap.csv")
        self.assertEqual(
            [row["case_label"] for row in metadata], list(CASE_LABELS)
        )
        self.assertEqual(set(cases["case_label"]), set(CASE_LABELS))
        self.assertTrue(
            all(
                len(cases.loc[cases["case_label"] == label]) == 19
                for label in CASE_LABELS
            )
        )

    def test_error_analysis_matches_threshold_confusion_matrix(self) -> None:
        errors = pd.read_csv("reports/day6_error_group_summary.csv").set_index(
            "outcome_group"
        )
        matrix = self.threshold["confusion_matrix"]["values"]
        self.assertEqual(int(errors.loc["true_negative", "rows"]), matrix[0][0])
        self.assertEqual(int(errors.loc["false_positive", "rows"]), matrix[0][1])
        self.assertEqual(int(errors.loc["false_negative", "rows"]), matrix[1][0])
        self.assertEqual(int(errors.loc["true_positive", "rows"]), matrix[1][1])
        self.assertEqual(int(errors["rows"].sum()), 6000)

    def test_required_artefacts_exist_without_row_level_prediction_file(self) -> None:
        required = [
            "reports/DAY6_VALIDATION_SELECTION_EXPLAINABILITY_REPORT.md",
            "reports/day6_validation_analysis_metrics.json",
            "reports/day6_selected_threshold.json",
            "reports/day6_paired_ap_uncertainty.json",
            "reports/day6_validation_model_comparison.csv",
            "reports/day6_random_forest_global_shap_encoded.csv",
            "reports/day6_random_forest_global_shap_source.csv",
            "reports/day6_case_level_metadata.json",
            "reports/day6_case_level_shap.csv",
            "reports/day6_error_group_summary.csv",
            "reports/day6_error_feature_summary.csv",
            "reports/figures/day6_threshold_tradeoff.png",
            "reports/figures/day6_paired_ap_bootstrap.png",
            "reports/figures/day6_random_forest_global_shap_bar.png",
            "reports/figures/day6_random_forest_global_shap_beeswarm.png",
            "reports/figures/day6_case_true_positive_shap_waterfall.png",
            "reports/figures/day6_case_false_positive_shap_waterfall.png",
            "reports/figures/day6_case_false_negative_shap_waterfall.png",
            "reports/figures/day6_validation_error_counts.png",
        ]
        self.assertTrue(all(Path(path).is_file() for path in required))
        self.assertFalse(Path("reports/day6_validation_predictions.csv").exists())


if __name__ == "__main__":
    unittest.main()
