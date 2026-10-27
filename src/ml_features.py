"""Approved model inputs for the next-event dataset.

configs/ml_feature_classification.json classifies every dataset column as
CAUSAL_FEATURE / TARGET / OFFLINE_DIAGNOSTIC / METADATA / EXCLUDED and marks
the approved training features explicitly. Training code must build its
feature matrix with `select_feature_matrix` (an allowlist), never by
dropping known-bad columns: a new column is unusable until it has been
classified and approved.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLASSIFICATION_PATH = PROJECT_ROOT / "configs" / "ml_feature_classification.json"
CATEGORIES = frozenset({"CAUSAL_FEATURE", "TARGET", "OFFLINE_DIAGNOSTIC", "METADATA", "EXCLUDED"})


class FeatureSafetyError(ValueError):
    """A column outside the approved causal allowlist reached a feature matrix,
    or the dataset schema and the classification disagree."""


def load_feature_classification(path: Path = DEFAULT_CLASSIFICATION_PATH) -> dict[str, Any]:
    classification = json.loads(Path(path).read_text(encoding="utf-8"))
    for column, entry in classification["columns"].items():
        if entry["category"] not in CATEGORIES:
            raise FeatureSafetyError(f"{column}: unknown category {entry['category']!r}")
        if entry["approved_for_training"] and entry["category"] != "CAUSAL_FEATURE":
            raise FeatureSafetyError(f"{column}: only CAUSAL_FEATURE columns can be approved")
    return classification


def approved_feature_columns(classification: dict[str, Any] | None = None) -> list[str]:
    """The approved causal features, in dataset column order."""
    classification = classification or load_feature_classification()
    return [column for column, entry in classification["columns"].items() if entry["approved_for_training"]]


def target_column(classification: dict[str, Any] | None = None) -> str:
    return (classification or load_feature_classification())["target"]


def validate_schema(columns: list[str], classification: dict[str, Any] | None = None) -> None:
    """Every dataset column must be classified, and every classified column present."""
    classified = set((classification or load_feature_classification())["columns"])
    unclassified = [c for c in columns if c not in classified]
    missing = sorted(classified - set(columns))
    if unclassified or missing:
        raise FeatureSafetyError(f"schema mismatch: unclassified {unclassified}, missing {missing}")


def validate_feature_columns(columns: list[str], classification: dict[str, Any] | None = None) -> None:
    """Raise unless every column is an approved causal feature."""
    approved = set(approved_feature_columns(classification))
    unsafe = [c for c in columns if c not in approved]
    if unsafe:
        raise FeatureSafetyError(f"not approved as model inputs: {unsafe}")


def select_feature_matrix(frame, classification: dict[str, Any] | None = None):
    """The approved feature columns of a dataset DataFrame (validated)."""
    classification = classification or load_feature_classification()
    validate_schema(list(frame.columns), classification)
    columns = approved_feature_columns(classification)
    validate_feature_columns(columns, classification)
    return frame[columns]
