import csv
import json
import unittest
from collections import defaultdict
from pathlib import Path

import joblib
import xgboost
from sklearn.base import clone

from src.data import (
    APPROVED_PRIMARY_FEATURES,
    AUDIT_ONLY_FEATURES,
    TARGET,
    load_split_assignments,
)
from src.evaluate import REQUIRED_VALIDATION_METRICS
from src.preprocessing import MONETARY_FIELDS, REPAYMENT_STATUS_FIELDS
from src.random_forest import (
    DAY4_PARTITIONS,
    audit_grouped_cv,
    build_grouped_cv,
    load_day4_partitions,
)
from src.xgboost_model import (
    CV_FOLDS,
    PARAMETER_DISTRIBUTIONS,
    SEARCH_ITERATIONS,
    _improvement_description,
    build_search,
)


class Day5XGBoostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.partitions = load_day4_partitions()
        cls.model = joblib.load("models/day5_xgboost_pipeline.joblib")
        cls.metrics = json.loads(
            Path("reports/day5_xgboost_validation_metrics.json").read_text(
                encoding="utf-8"
            )
        )
        cls.selected = json.loads(
            Path("reports/day5_xgboost_selected_params.json").read_text(
                encoding="utf-8"
            )
        )

    def test_exact_approved_feature_scope(self) -> None:
        self.assertEqual(
            tuple(self.partitions.X_train.columns),
            APPROVED_PRIMARY_FEATURES,
        )
        configured = self.model.named_steps["preprocessor"].transformers
        configured_fields = tuple(
            field
            for _, _, fields in configured
            for field in fields
        )
        self.assertEqual(set(configured_fields), set(APPROVED_PRIMARY_FEATURES))
        self.assertEqual(len(configured_fields), 19)
        excluded = {"ID", TARGET, *AUDIT_ONLY_FEATURES}
        self.assertTrue(excluded.isdisjoint(configured_fields))
        self.assertFalse(any(field.endswith("_CATEGORY") for field in configured_fields))
        self.assertEqual(
            set(configured_fields),
            set(REPAYMENT_STATUS_FIELDS) | set(MONETARY_FIELDS),
        )

    def test_preprocessing_is_fitted_inside_pipeline(self) -> None:
        encoder = self.model.named_steps["preprocessor"].named_transformers_[
            "repayment_status"
        ]
        self.assertEqual(len(encoder.categories_), len(REPAYMENT_STATUS_FIELDS))
        self.assertEqual(
            len(self.model.named_steps["preprocessor"].get_feature_names_out()),
            76,
        )

    def test_grouped_cv_has_zero_overlap(self) -> None:
        audit = audit_grouped_cv(
            build_grouped_cv(),
            self.partitions.X_train,
            self.partitions.y_train,
            self.partitions.training_groups,
        )
        self.assertEqual(len(audit), CV_FOLDS)
        self.assertTrue(all(fold["overlapping_groups"] == 0 for fold in audit))
        self.assertEqual(audit, self.selected["cv_group_audit"])

    def test_bounded_search_configuration(self) -> None:
        search = build_search()
        self.assertEqual(search.n_iter, 12)
        self.assertEqual(search.refit, "average_precision")
        self.assertEqual(search.random_state, 42)
        self.assertFalse(search.return_train_score)
        self.assertEqual(set(search.scoring), {"average_precision", "roc_auc"})
        expected_dimensions = {
            "n_estimators",
            "max_depth",
            "learning_rate",
            "min_child_weight",
            "subsample",
            "colsample_bytree",
            "gamma",
            "reg_lambda",
            "scale_pos_weight",
        }
        self.assertEqual(
            {
                key.removeprefix("classifier__")
                for key in PARAMETER_DISTRIBUTIONS
            },
            expected_dimensions,
        )

    def test_search_is_training_only_and_validation_excluded(self) -> None:
        self.assertEqual(self.selected["search_partition"], "training")
        self.assertFalse(self.selected["validation_used_for_selection"])
        self.assertFalse(
            self.metrics["validation_used_for_hyperparameter_selection"]
        )
        self.assertEqual(self.selected["search_iterations"], SEARCH_ITERATIONS)
        self.assertEqual(
            self.selected["fitted_candidates"],
            SEARCH_ITERATIONS * CV_FOLDS,
        )
        with Path("reports/day5_xgboost_search_results.csv").open(
            newline="",
            encoding="utf-8",
        ) as source:
            self.assertEqual(len(list(csv.DictReader(source))), SEARCH_ITERATIONS)

    def test_final_test_rows_and_outcomes_are_excluded(self) -> None:
        assignments = load_split_assignments()
        used_ids = (
            set(self.partitions.training_ids)
            | set(self.partitions.validation_ids)
        )
        expected_ids = {
            identifier
            for identifier, split in assignments.items()
            if split in DAY4_PARTITIONS
        }
        final_test_ids = {
            identifier
            for identifier, split in assignments.items()
            if split == "test"
        }
        self.assertEqual(used_ids, expected_ids)
        self.assertTrue(used_ids.isdisjoint(final_test_ids))
        self.assertFalse(self.metrics["test_partition_evaluated"])

    def test_probabilities_are_valid_and_deterministic(self) -> None:
        expected = self.model.predict_proba(
            self.partitions.X_validation.iloc[:250]
        )[:, 1]
        self.assertTrue(((expected >= 0.0) & (expected <= 1.0)).all())

        repeated = clone(self.model)
        repeated.named_steps["classifier"].set_params(n_jobs=1)
        repeated.fit(self.partitions.X_train, self.partitions.y_train)
        actual = repeated.predict_proba(
            self.partitions.X_validation.iloc[:250]
        )[:, 1]
        maximum_difference = max(
            abs(float(left) - float(right))
            for left, right in zip(expected, actual)
        )
        self.assertLessEqual(maximum_difference, 1e-7)
        self.assertTrue(((expected >= 0.5) == (actual >= 0.5)).all())

    def test_metrics_and_version(self) -> None:
        self.assertEqual(xgboost.__version__, "2.1.4")
        self.assertEqual(self.metrics["xgboost_version"], "2.1.4")
        self.assertTrue(REQUIRED_VALIDATION_METRICS.issubset(self.metrics))
        self.assertEqual(self.metrics["partition"], "validation")
        self.assertEqual(self.metrics["source_feature_count"], 19)
        self.assertEqual(self.metrics["encoded_feature_count"], 76)
        self.assertFalse(self.metrics["threshold_selected"])

    def test_gain_importance_aggregation_is_correct(self) -> None:
        with Path("reports/day5_xgboost_encoded_importance.csv").open(
            newline="",
            encoding="utf-8",
        ) as source:
            encoded = list(csv.DictReader(source))
        with Path("reports/day5_xgboost_source_importance.csv").open(
            newline="",
            encoding="utf-8",
        ) as source:
            source_rows = list(csv.DictReader(source))
        self.assertEqual(len(encoded), 76)
        self.assertEqual(len(source_rows), 19)

        aggregated = defaultdict(float)
        for row in encoded:
            aggregated[row["source_feature"]] += float(row["gain"])
        reported = {
            row["source_feature"]: float(row["aggregated_gain"])
            for row in source_rows
        }
        self.assertEqual(set(aggregated), set(APPROVED_PRIMARY_FEATURES))
        for field in APPROVED_PRIMARY_FEATURES:
            self.assertAlmostEqual(aggregated[field], reported[field], places=7)
        self.assertAlmostEqual(
            sum(float(row["normalised_gain"]) for row in source_rows),
            1.0,
            places=9,
        )

    def test_required_artefacts_exist(self) -> None:
        required = [
            "models/day5_xgboost_pipeline.joblib",
            "reports/day5_xgboost_selected_params.json",
            "reports/day5_xgboost_search_results.csv",
            "reports/day5_xgboost_validation_metrics.json",
            "reports/day5_validation_model_comparison.csv",
            "reports/day5_xgboost_encoded_importance.csv",
            "reports/day5_xgboost_source_importance.csv",
            "reports/figures/day5_xgboost_validation_roc.png",
            "reports/figures/day5_xgboost_validation_precision_recall.png",
            "reports/DAY5_XGBOOST_REPORT.md",
        ]
        self.assertTrue(all(Path(path).is_file() for path in required))


class ImprovementDescriptionTests(unittest.TestCase):
    def test_small_positive_difference_is_negligible(self) -> None:
        self.assertEqual(_improvement_description(0.504, 0.500), "negligible")

    def test_small_negative_difference_is_negligible(self) -> None:
        self.assertEqual(
            _improvement_description(
                0.5366433171661142,
                0.539979526640148,
            ),
            "negligible",
        )

    def test_modest_positive_difference_preserves_direction(self) -> None:
        self.assertEqual(
            _improvement_description(0.510, 0.500),
            "modest improvement",
        )

    def test_modest_negative_difference_preserves_direction(self) -> None:
        self.assertEqual(
            _improvement_description(0.490, 0.500),
            "modest decline",
        )

    def test_meaningful_positive_difference_preserves_direction(self) -> None:
        self.assertEqual(
            _improvement_description(0.525, 0.500),
            "meaningful improvement",
        )

    def test_meaningful_negative_difference_preserves_direction(self) -> None:
        self.assertEqual(
            _improvement_description(0.475, 0.500),
            "meaningful decline",
        )


if __name__ == "__main__":
    unittest.main()
