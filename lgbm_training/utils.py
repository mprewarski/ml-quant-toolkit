"""Shared helpers for the LightGBM training toolkit."""

from __future__ import annotations

import json
from typing import Any, Iterable

import numpy as np
import pandas as pd


def ensure_datetime(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series, errors="coerce")


def to_native(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Period):
        return str(value)
    return value


def list_to_native(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    native_rows: list[dict[str, Any]] = []
    for row in rows:
        native_rows.append({key: to_native(value) for key, value in row.items()})
    return native_rows


def dumps_json(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True, default=to_native)
