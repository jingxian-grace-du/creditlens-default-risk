import csv
import json
import unittest
from pathlib import Path

import joblib
from sklearn.base import clone

from src.data import APPROVED_PRIMARY_FEATURES, AUDIT_ONLY_FEATURES, ORIGINAL_PREDICTORS, TARGET, load_split_assignments
from src.evaluate import REQUIRED_VALIDATION_METRICS
from src.preprocessing import MONETARY_FIELDS, REPAYMENT_STATUS_FIELDS
from src.random_forest import (
    CV_FOLDS,
    DAY4_PARTITIONS,
    SEARCH_ITERATIONS,
    _profile_identifiers,
    audit_grouped_cv,
    build_grouped_cv,
    load_day4_partitions,
)


class Day4RandomForestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.partitions = load_day4_partitions()
        cls.model = joblib.load("models/day4_random_forest_pipeline.joblib")
        cls.metrics = json.loads(
            Path("reports/day4_random_forest_validation_metrics.json").read_text(encoding="utf-8")
        )

    def test_only_approved_fields_enter_random_forest(self) -> None:
        self.assertEqual(tuple(self.partitions.X_train.columns), APPROVED_PRIMARY_FEATURES)
        configured = self.model.named_steps["preprocessor"].transformers
        configured_fields = tuple(field for _, _, fields in configured for field in fields)
        self.assertEqual(set(configured_fields), set(APPROVED_PRIMARY_FEATURES))
        self.assertEqual(len(configured_fields), 19)
        self.assertEqual(set(REPAYMENT_STATUS_FIELDS) | set(MONETARY_FIELDS), set(configured_fields))
        self.assertTrue({"ID", TARGET, *AUDIT_ONLY_FEATURES}.isdisjoint(configured_fields))

    def test_audit_fields_are_grouping_metadata_only(self) -> None:
        self.assertTrue(AUDIT_ONLY_FEATURES.isdisjoint(self.partitions.X_train.columns))
        sample = {field: [0, 0] for field in ORIGINAL_PREDICTORS}
        sample["SEX"] = [1, 2]
        frame = __import__("pandas").DataFrame(sample)
        groups = _profile_identifiers(frame)
        self.assertNotEqual(groups.iloc[0], groups.iloc[1])

    def test_internal_cv_has_zero_group_overlap(self) -> None:
        audit = audit_grouped_cv(
            build_grouped_cv(),
            self.partitions.X_train,
            self.partitions.y_train,
            self.partitions.training_groups,
        )
        self.assertEqual(len(audit), CV_FOLDS)
        self.assertTrue(all(fold["overlapping_groups"] == 0 for fold in audit))

    def test_preprocessing_is_fitted_inside_saved_pipeline(self) -> None:
        encoder = self.model.named_steps["preprocessor"].named_transformers_["repayment_status"]
        self.assertEqual(len(encoder.categories_), len(REPAYMENT_STATUS_FIELDS))
        self.assertEqual(len(self.model.named_steps["preprocessor"].get_feature_names_out()), 76)

    def test_search_is_training_only_and_validation_excluded(self) -> None:
        selected = json.loads(
            Path("reports/day4_random_forest_selected_params.json").read_text(encoding="utf-8")
        )
        self.assertEqual(selected["search_partition"], "training")
        self.assertFalse(selected["validation_used_for_selection"])
        self.assertEqual(selected["search_iterations"], SEARCH_ITERATIONS)
        self.assertEqual(selected["fitted_candidates"], SEARCH_ITERATIONS * CV_FOLDS)
        with Path("reports/day4_random_forest_search_results.csv").open(newline="", encoding="utf-8") as source:
            self.assertEqual(len(list(csv.DictReader(source))), SEARCH_ITERATIONS)

    def test_test_rows_and_outcomes_are_not_materialised(self) -> None:
        assignments = load_split_assignments()
        used_ids = set(self.partitions.training_ids) | set(self.partitions.validation_ids)
        expected = {identifier for identifier, split in assignments.items() if split in DAY4_PARTITIONS}
        test_ids = {identifier for identifier, split in assignments.items() if split == "test"}
        self.assertEqual(used_ids, expected)
        self.assertTrue(used_ids.isdisjoint(test_ids))

    def test_probabilities_are_valid(self) -> None:
        probabilities = self.model.predict_proba(self.partitions.X_validation)[:, 1]
        self.assertTrue(((probabilities >= 0.0) & (probabilities <= 1.0)).all())

    def test_selected_model_is_deterministic(self) -> None:
        repeated = clone(self.model)
        repeated.named_steps["classifier"].set_params(n_jobs=1)
        repeated.fit(self.partitions.X_train, self.partitions.y_train)
        expected = self.model.predict_proba(self.partitions.X_validation.iloc[:250])[:, 1]
        actual = repeated.predict_proba(self.partitions.X_validation.iloc[:250])[:, 1]
        maximum_difference = max(abs(float(left) - float(right)) for left, right in zip(expected, actual))
        self.assertLessEqual(maximum_difference, 1e-12)
        self.assertTrue(((expected >= 0.5) == (actual >= 0.5)).all())

    def test_required_metrics_and_artefacts_exist(self) -> None:
        self.assertTrue(REQUIRED_VALIDATION_METRICS.issubset(self.metrics))
        self.assertEqual(self.metrics["partition"], "validation")
        self.assertFalse(self.metrics["test_partition_evaluated"])
        self.assertFalse(self.metrics["validation_used_for_hyperparameter_selection"])
        required = [
            "models/day4_random_forest_pipeline.joblib",
            "reports/day4_random_forest_search_results.csv",
            "reports/day4_random_forest_selected_params.json",
            "reports/day4_random_forest_validation_metrics.json",
            "reports/day4_random_forest_encoded_importance.csv",
            "reports/day4_random_forest_source_importance.csv",
            "reports/day4_validation_model_comparison.csv",
            "reports/figures/day4_random_forest_validation_roc.png",
            "reports/figures/day4_random_forest_validation_precision_recall.png",
            "reports/DAY4_RANDOM_FOREST_REPORT.md",
        ]
        self.assertTrue(all(Path(path).is_file() for path in required))


if __name__ == "__main__":
    unittest.main()
