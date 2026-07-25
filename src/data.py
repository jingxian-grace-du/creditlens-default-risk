"""Reusable loaders and integrity checks for CreditLens derived data.

This module deliberately uses only Python's standard library. The official XLS
workbook is converted by ``scripts/day2_prepare.R`` using the existing R
``readxl`` package; raw files are never changed in place.
"""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from pathlib import Path
from typing import Iterable

EXPECTED_RAW_SHA256 = "30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933"
EXPECTED_ARCHIVE_SHA256 = "56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602"
EXPECTED_ROWS = 30_000
SPLIT_SEED = 42
EXPECTED_SPLIT_PROPORTIONS = {"train": 0.60, "validation": 0.20, "test": 0.20}
SPLIT_PROPORTION_TOLERANCE = 0.01
DEFAULT_PREVALENCE_TOLERANCE = 0.005
TARGET = "DEFAULT_NEXT_MONTH"
AUDIT_ONLY_FEATURES = frozenset({"SEX", "AGE", "MARRIAGE", "EDUCATION"})
NON_MODEL_FIELDS = frozenset({"ID", TARGET}) | AUDIT_ONLY_FEATURES
ORIGINAL_PREDICTORS = (
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
)
APPROVED_PRIMARY_FEATURES = (
    "LIMIT_BAL", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
)


def sha256sum(path: str | Path) -> str:
    """Return the SHA-256 digest of *path* without modifying it."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_raw_checksums(raw_directory: str | Path = "data/raw") -> None:
    """Raise ``ValueError`` if either official raw artefact has changed."""

    raw_directory = Path(raw_directory)
    expected = {
        raw_directory / "default of credit card clients.xls": EXPECTED_RAW_SHA256,
        raw_directory / "default-of-credit-card-clients.zip": EXPECTED_ARCHIVE_SHA256,
    }
    for path, expected_digest in expected.items():
        actual = sha256sum(path)
        if actual != expected_digest:
            raise ValueError(f"Checksum mismatch for {path}: {actual}")


def load_validated_rows(
    path: str | Path = "data/processed/credit_default_validated.csv",
) -> list[dict[str, str]]:
    """Load the validated, derived CSV and apply core schema assertions."""

    with Path(path).open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} rows, found {len(rows)}")
    identifiers = [int(float(row["ID"])) for row in rows]
    if len(set(identifiers)) != EXPECTED_ROWS:
        raise ValueError("ID values are not unique")
    if {int(float(row[TARGET])) for row in rows} != {0, 1}:
        raise ValueError("Target is not binary")
    return rows


def load_split_assignments(
    path: str | Path = "data/processed/split_assignments.csv",
    validated_rows: Iterable[dict[str, str]] | None = None,
) -> dict[int, str]:
    """Load and validate persisted row-to-split assignments.

    Partition proportions may differ from their 60/20/20 targets because an
    identical 23-predictor profile is indivisible. Each observed proportion
    must be within one absolute percentage point of its target.
    """

    with Path(path).open(newline="", encoding="utf-8") as source:
        records = list(csv.DictReader(source))
    if len(records) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} assignment records, found {len(records)}")
    identifiers = [int(float(row["ID"])) for row in records]
    if len(set(identifiers)) != EXPECTED_ROWS:
        raise ValueError("Split assignment IDs must be unique")
    assignments = {identifier: row["split"] for identifier, row in zip(identifiers, records)}
    counts = Counter(assignments.values())
    if set(counts) != set(EXPECTED_SPLIT_PROPORTIONS):
        raise ValueError(f"Unexpected split labels: {sorted(counts)}")
    if any(counts[label] == 0 for label in EXPECTED_SPLIT_PROPORTIONS):
        raise ValueError("Every partition must be non-empty")
    for label, expected in EXPECTED_SPLIT_PROPORTIONS.items():
        actual = counts[label] / EXPECTED_ROWS
        if abs(actual - expected) > SPLIT_PROPORTION_TOLERANCE:
            raise ValueError(
                f"{label} proportion {actual:.4f} is outside the "
                f"±{SPLIT_PROPORTION_TOLERANCE:.2f} tolerance around {expected:.2f}"
            )
    if validated_rows is not None:
        validated_ids = {int(float(row["ID"])) for row in validated_rows}
        if validated_ids != set(assignments):
            raise ValueError("Validated-data and assignment ID sets do not match exactly")
    return assignments


def primary_feature_names(fieldnames: Iterable[str]) -> list[str]:
    """Return the approved version-1 feature fields in their source order."""

    return [
        name
        for name in fieldnames
        if name not in NON_MODEL_FIELDS and not name.endswith("_CATEGORY")
    ]
