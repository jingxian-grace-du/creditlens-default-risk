"""Leakage-safe preprocessing and Logistic Regression baseline construction."""

from __future__ import annotations

from collections.abc import Iterable

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import APPROVED_PRIMARY_FEATURES

REPAYMENT_STATUS_FIELDS = (
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
)
MONETARY_FIELDS = (
    "LIMIT_BAL",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
)


def validate_primary_fields(fieldnames: Iterable[str]) -> None:
    """Fail clearly when any approved source field is absent."""

    available = set(fieldnames)
    missing = [field for field in APPROVED_PRIMARY_FEATURES if field not in available]
    if missing:
        raise ValueError(f"Missing required primary fields: {', '.join(missing)}")


def build_logistic_baseline_pipeline() -> Pipeline:
    """Return the fixed, untuned Day 3 baseline pipeline.

    Repayment codes remain distinct categorical values. Monetary variables are
    standardised without clipping, winsorisation, imputation or outlier removal.
    All learned preprocessing is fitted when ``Pipeline.fit`` receives training
    data; the validation partition is never supplied to ``fit``.
    """

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "repayment_status",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                list(REPAYMENT_STATUS_FIELDS),
            ),
            ("monetary", StandardScaler(), list(MONETARY_FIELDS)),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    classifier = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="liblinear",
        max_iter=1_000,
        random_state=42,
        class_weight=None,
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def build_random_forest_pipeline() -> Pipeline:
    """Return the untuned Random Forest pipeline used inside Day 4 search.

    Repayment codes are one-hot encoded inside the pipeline. Monetary fields
    pass through unchanged because tree splits do not require scaling. No
    imputation, capping, winsorisation or outlier removal is performed.
    """

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "repayment_status",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(REPAYMENT_STATUS_FIELDS),
            ),
            ("monetary", "passthrough", list(MONETARY_FIELDS)),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    classifier = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        bootstrap=True,
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])
