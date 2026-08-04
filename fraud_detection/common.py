"""Shared validation and normalization helpers."""

from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np
import pandas as pd


def to_snake_case(value: object) -> str:
    """Convert a label to lowercase snake_case."""
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(value).strip())
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with unique snake_case column names."""
    names = [to_snake_case(column) for column in frame.columns]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate normalized columns: {duplicates}")
    result = frame.copy()
    result.columns = names
    return result


def require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    """Raise ``ValueError`` if any required column is missing."""
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def to_binary(value: object) -> float:
    """Convert common boolean-like values to 1.0, 0.0, or NaN."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return 1.0
    if text in {"0", "false", "f", "no", "n"}:
        return 0.0
    try:
        return float(float(text) != 0)
    except (TypeError, ValueError):
        return np.nan
