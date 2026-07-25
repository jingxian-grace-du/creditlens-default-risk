import inspect
import json
import unittest
from pathlib import Path

import pandas as pd

from src.final_evaluation import (
    EXPECTED_FINAL_TEST_ROWS,
    EXPECTED_MODEL_SHA256,
    FROZEN_THRESHOLD,
    load_final_test_partition,
    sha256sum,
    verify_decision_lock,
)


class Day7FinalEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lock = json.loads(
            Path("reports/day7_final_decision_lock.json").read_text(
                encoding="utf-8"
            )
        )
        cls.metrics = json.loads(
            Path("reports/day7_final_test_metrics.json").read_text(
                encoding="utf-8"
            )
        )
        cls.comparison = pd.read_csv(
            "reports/day7_validation_test_comparison.csv"
        )

    def test_pre_evaluation_lock_is_complete_and_verifiable(self) -> None:
        verified = verify_decision_lock()
        self.assertTrue(verified["created_before_final_test_outcomes_loaded"])
        self.assertEqual(
            verified["frozen_model_path"],
            "models/day4_random_forest_pipeline.joblib",
        )
        self.assertEqual(verified["frozen_threshold"], FROZEN_THRESHOLD)
        self.assertTrue(verified["no_superiority_claim"])
        self.assertEqual(
            verified["paired_rf_minus_xgboost_ap_uncertainty_interval"],
            [-0.0075220545, 0.0137943885],
        )

    def test_frozen_model_hash_is_unchanged(self) -> None:
        actual = sha256sum("models/day4_random_forest_pipeline.joblib")
        self.assertEqual(actual, EXPECTED_MODEL_SHA256)
        self.assertEqual(self.metrics["model_sha256_before"], actual)
        self.assertEqual(self.metrics["model_sha256_after"], actual)

    def test_dedicated_loader_is_lock_first_and_test_only_by_construction(self) -> None:
        source = inspect.getsource(load_final_test_partition)
        self.assertLess(
            source.index("verify_decision_lock"),
            source.index("pd.read_csv(data_path"),
        )
        self.assertIn('assignments[int(identifier)] != "test"', source)
        self.assertIn('set(loaded["partition"]) != {"test"}', source)
        self.assertIn("skiprows=", source)
        # Do not invoke the loader here: the authorised scoring pass is one-time.

    def test_final_metrics_are_aggregate_and_complete(self) -> None:
        required = {
            "roc_auc",
            "average_precision",
            "precision",
            "recall",
            "f1",
            "false_positive_rate",
            "proportion_flagged",
            "confusion_matrix",
            "class_prevalence",
        }
        self.assertTrue(required.issubset(self.metrics))
        self.assertEqual(self.metrics["partition"], "final test")
        self.assertEqual(self.metrics["rows"], EXPECTED_FINAL_TEST_ROWS)
        matrix = self.metrics["confusion_matrix"]["values"]
        self.assertEqual(sum(sum(row) for row in matrix), EXPECTED_FINAL_TEST_ROWS)
        self.assertAlmostEqual(
            self.metrics["class_prevalence"],
            (matrix[1][0] + matrix[1][1]) / EXPECTED_FINAL_TEST_ROWS,
        )

    def test_frozen_threshold_was_not_revised_for_test_recall(self) -> None:
        self.assertEqual(self.metrics["frozen_threshold"], FROZEN_THRESHOLD)
        self.assertLess(self.metrics["recall"], 0.70)
        self.assertTrue(
            self.metrics["recall_floor_was_validation_target_not_test_guarantee"]
        )
        self.assertFalse(self.metrics["threshold_changed_after_test"])

    def test_no_post_test_model_or_preprocessing_change(self) -> None:
        self.assertFalse(self.metrics["model_fitted_or_refitted"])
        self.assertFalse(self.metrics["model_tuned_or_recalibrated"])
        self.assertFalse(self.metrics["preprocessing_changed_after_test"])
        self.assertFalse(self.metrics["alternative_models_scored_on_final_test"])
        self.assertFalse(self.metrics["post_test_decision_changed"])

    def test_validation_comparison_differences_are_arithmetic_only(self) -> None:
        self.assertEqual(len(self.comparison), 7)
        calculated = (
            self.comparison["final_test"] - self.comparison["validation"]
        )
        self.assertTrue(
            (
                abs(calculated - self.comparison["final_minus_validation"])
                < 1e-12
            ).all()
        )

    def test_required_artefacts_exist_and_no_row_predictions_are_saved(self) -> None:
        required = [
            "reports/day7_final_decision_lock.json",
            "reports/day7_final_test_metrics.json",
            "reports/day7_validation_test_comparison.csv",
            "reports/DAY7_FINAL_EVALUATION_REPORT.md",
            "reports/figures/day7_final_test_roc.png",
            "reports/figures/day7_final_test_precision_recall.png",
            "reports/figures/day7_final_test_confusion_matrix.png",
        ]
        self.assertTrue(all(Path(path).is_file() for path in required))
        self.assertFalse(Path("reports/day7_final_test_predictions.csv").exists())
        self.assertFalse(self.metrics["row_level_predictions_saved"])


if __name__ == "__main__":
    unittest.main()
