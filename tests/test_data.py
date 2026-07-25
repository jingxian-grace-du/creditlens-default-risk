import unittest
from collections import Counter, defaultdict

from src.data import (
    APPROVED_PRIMARY_FEATURES,
    AUDIT_ONLY_FEATURES,
    DEFAULT_PREVALENCE_TOLERANCE,
    EXPECTED_SPLIT_PROPORTIONS,
    ORIGINAL_PREDICTORS,
    SPLIT_PROPORTION_TOLERANCE,
    TARGET,
    load_split_assignments,
    load_validated_rows,
    primary_feature_names,
    verify_raw_checksums,
)


class DataIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = load_validated_rows()
        cls.assignments = load_split_assignments(validated_rows=cls.rows)

    def test_raw_checksums(self) -> None:
        verify_raw_checksums()

    def test_validated_data(self) -> None:
        self.assertEqual(len(self.rows), 30_000)
        identifiers = [int(float(row["ID"])) for row in self.rows]
        self.assertEqual(len(set(identifiers)), 30_000)
        self.assertEqual({int(float(row[TARGET])) for row in self.rows}, {0, 1})

    def test_assignment_ids_exactly_match_validated_ids(self) -> None:
        validated_ids = {int(float(row["ID"])) for row in self.rows}
        self.assertEqual(set(self.assignments), validated_ids)

    def test_split_proportions_are_within_tolerance(self) -> None:
        counts = Counter(self.assignments.values())
        self.assertEqual(set(counts), set(EXPECTED_SPLIT_PROPORTIONS))
        for label, expected in EXPECTED_SPLIT_PROPORTIONS.items():
            actual = counts[label] / len(self.assignments)
            self.assertLessEqual(abs(actual - expected), SPLIT_PROPORTION_TOLERANCE)

    def test_no_predictor_profile_crosses_partitions(self) -> None:
        profile_partitions = defaultdict(set)
        for row in self.rows:
            profile = tuple(row[field] for field in ORIGINAL_PREDICTORS)
            profile_partitions[profile].add(self.assignments[int(float(row["ID"]))])
        crossing = [profile for profile, partitions in profile_partitions.items() if len(partitions) > 1]
        self.assertEqual(crossing, [])

    def test_partition_default_prevalence_is_close_to_full_data(self) -> None:
        full_prevalence = sum(int(float(row[TARGET])) for row in self.rows) / len(self.rows)
        by_partition = defaultdict(list)
        for row in self.rows:
            partition = self.assignments[int(float(row["ID"]))]
            by_partition[partition].append(int(float(row[TARGET])))
        for values in by_partition.values():
            prevalence = sum(values) / len(values)
            self.assertLessEqual(abs(prevalence - full_prevalence), DEFAULT_PREVALENCE_TOLERANCE)

    def test_primary_features_match_approved_fields(self) -> None:
        features = primary_feature_names(self.rows[0].keys())
        self.assertTrue(AUDIT_ONLY_FEATURES.isdisjoint(features))
        self.assertNotIn("ID", features)
        self.assertNotIn(TARGET, features)
        self.assertFalse(any(field.endswith("_CATEGORY") for field in features))
        self.assertEqual(len(features), 19)
        self.assertEqual(tuple(features), APPROVED_PRIMARY_FEATURES)


if __name__ == "__main__":
    unittest.main()
