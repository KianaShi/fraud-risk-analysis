"""Shared validation and normalization helpers."""

from __future__ import annotations

import re
from collections.abc import Iterable
from numbers import Real

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
    """Convert an explicit boolean-like value to 1.0, 0.0, or NaN."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    if isinstance(value, Real):
        numeric = float(value)
        if numeric in (0.0, 1.0):
            return numeric
        raise ValueError(f"Invalid binary value: {value!r}")
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return 1.0
    if text in {"0", "false", "f", "no", "n"}:
        return 0.0
    raise ValueError(f"Invalid binary value: {value!r}")


def coerce_binary_series(series: pd.Series, column: str) -> pd.Series:
    """Coerce explicit binary tokens while reporting invalid observed values."""
    converted: list[float] = []
    invalid: list[object] = []
    for value in series:
        try:
            converted.append(to_binary(value))
        except ValueError:
            converted.append(np.nan)
            invalid.append(value)
    if invalid:
        examples = [str(value) for value in invalid[:5]]
        raise ValueError(
            f"Column {column!r} contains invalid binary values: count={len(invalid)}, "
            f"examples={examples}"
        )
    return pd.Series(converted, index=series.index, dtype=float, name=series.name)
