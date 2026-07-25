import json
import unittest
from collections import Counter
from pathlib import Path

import joblib

from src.data import (
    APPROVED_PRIMARY_FEATURES,
    AUDIT_ONLY_FEATURES,
    TARGET,
    load_split_assignments,
)
from src.evaluate import REQUIRED_VALIDATION_METRICS
from src.preprocessing import (
    MONETARY_FIELDS,
    REPAYMENT_STATUS_FIELDS,
    build_logistic_baseline_pipeline,
    validate_primary_fields,
)
from src.train import DAY3_PARTITIONS, fit_logistic_baseline, load_day3_partitions


class Day3LogisticBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.partitions = load_day3_partitions()
        cls.pipeline, cls.convergence_warnings = fit_logistic_baseline(cls.partitions)

    def test_only_approved_source_fields_enter_preprocessing(self) -> None:
        self.assertEqual(tuple(self.partitions.X_train.columns), APPROVED_PRIMARY_FEATURES)
        configured = self.pipeline.named_steps["preprocessor"].transformers
        configured_fields = tuple(field for _, _, fields in configured for field in fields)
        self.assertEqual(set(configured_fields), set(APPROVED_PRIMARY_FEATURES))
        self.assertEqual(len(configured_fields), 19)

    def test_excluded_fields_cannot_enter_model(self) -> None:
        excluded = {"ID", TARGET, *AUDIT_ONLY_FEATURES}
        self.assertTrue(excluded.isdisjoint(self.partitions.X_train.columns))
        self.assertFalse(any(name.endswith("_CATEGORY") for name in self.partitions.X_train.columns))
        transformed_names = self.pipeline.named_steps["preprocessor"].get_feature_names_out()
        self.assertFalse(any(any(field in name for field in excluded) for name in transformed_names))
        with self.assertRaisesRegex(ValueError, "Missing required primary fields"):
            validate_primary_fields(self.partitions.X_train.drop(columns=["LIMIT_BAL"]).columns)

    def test_preprocessing_statistics_come_from_training_only(self) -> None:
        scaler = self.pipeline.named_steps["preprocessor"].named_transformers_["monetary"]
        training_means = self.partitions.X_train.loc[:, list(MONETARY_FIELDS)].mean()
        combined_means = __import__("pandas").concat(
            [self.partitions.X_train, self.partitions.X_validation]
        ).loc[:, list(MONETARY_FIELDS)].mean()
        for position, field in enumerate(MONETARY_FIELDS):
            self.assertAlmostEqual(scaler.mean_[position], training_means[field], places=10)
        self.assertTrue(any(abs(scaler.mean_[i] - combined_means[field]) > 1e-6 for i, field in enumerate(MONETARY_FIELDS)))

    def test_unseen_repayment_category_is_safe(self) -> None:
        unseen = self.partitions.X_validation.iloc[[0]].copy()
        unseen.loc[:, "PAY_0"] = 999
        probability = self.pipeline.predict_proba(unseen)[0, 1]
        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)

    def test_probabilities_are_valid_and_runs_are_deterministic(self) -> None:
        first = self.pipeline.predict_proba(self.partitions.X_validation)[:, 1]
        second_pipeline = build_logistic_baseline_pipeline()
        second_pipeline.fit(self.partitions.X_train, self.partitions.y_train)
        second = second_pipeline.predict_proba(self.partitions.X_validation)[:, 1]
        self.assertTrue(((first >= 0.0) & (first <= 1.0)).all())
        self.assertTrue((first == second).all())

    def test_day3_uses_no_final_test_rows(self) -> None:
        assignments = load_split_assignments()
        used_ids = set(self.partitions.training_ids) | set(self.partitions.validation_ids)
        expected_ids = {identifier for identifier, split in assignments.items() if split in DAY3_PARTITIONS}
        test_ids = {identifier for identifier, split in assignments.items() if split == "test"}
        self.assertEqual(used_ids, expected_ids)
        self.assertTrue(used_ids.isdisjoint(test_ids))
        used_counts = Counter(assignments[i] for i in used_ids)
        self.assertEqual(used_counts["train"], len(self.partitions.training_ids))
        self.assertEqual(used_counts["validation"], len(self.partitions.validation_ids))
        self.assertGreater(used_counts["train"], 0)
        self.assertGreater(used_counts["validation"], 0)

    def test_saved_metrics_and_pipeline(self) -> None:
        metrics = json.loads(Path("reports/day3_logistic_validation_metrics.json").read_text(encoding="utf-8"))
        self.assertTrue(REQUIRED_VALIDATION_METRICS.issubset(metrics))
        self.assertEqual(metrics["partition"], "validation")
        self.assertFalse(metrics["test_partition_evaluated"])
        self.assertEqual(metrics["source_feature_count"], 19)
        self.assertEqual(metrics["encoded_feature_count"], 76)
        self.assertEqual(metrics["convergence_warnings"], [])
        saved = joblib.load("models/day3_logistic_regression_baseline.joblib")
        self.assertEqual(len(saved.named_steps["preprocessor"].get_feature_names_out()), 76)


if __name__ == "__main__":
    unittest.main()
