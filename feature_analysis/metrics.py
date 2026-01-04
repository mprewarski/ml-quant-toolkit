"""Metrics for feature quality, redundancy, and stability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .utils import ensure_datetime, infer_target_kind, is_numeric_series, list_to_native


@dataclass
class CorrelationResult:
    partners: list[dict[str, Any]]
    top_pairs: list[dict[str, Any]]


def basic_profile(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for col in df.columns:
        series = df[col]
        missing_pct = series.isna().mean() * 100.0
        nunique = series.nunique(dropna=True)
        entry: dict[str, Any] = {
            "feature": col,
            "dtype": str(series.dtype),
            "missing_pct": float(missing_pct),
            "nunique": int(nunique),
        }
        if is_numeric_series(series):
            entry.update(
                {
                    "mean": float(series.mean(skipna=True)),
                    "std": float(series.std(skipna=True)),
                    "min": float(series.min(skipna=True)),
                    "max": float(series.max(skipna=True)),
                }
            )
        rows.append(entry)
    return list_to_native(rows)


def correlation_analysis(df: pd.DataFrame) -> CorrelationResult:
    if df.shape[1] < 2:
        return CorrelationResult(partners=[], top_pairs=[])

    corr = df.corr(method="pearson")
    partners: list[dict[str, Any]] = []
    top_pairs: list[dict[str, Any]] = []

    abs_corr = corr.abs()
    for feature in corr.columns:
        series = abs_corr[feature].drop(index=feature, errors="ignore").dropna()
        if series.empty:
            continue
        partner = series.idxmax()
        partners.append(
            {
                "feature": feature,
                "partner": partner,
                "corr": float(corr.loc[feature, partner]),
                "abs_corr": float(abs_corr.loc[feature, partner]),
            }
        )

    visited: set[tuple[str, str]] = set()
    for i, f1 in enumerate(corr.columns):
        for f2 in corr.columns[i + 1 :]:
            pair = tuple(sorted((f1, f2)))
            if pair in visited:
                continue
            visited.add(pair)
            value = corr.loc[f1, f2]
            if pd.isna(value):
                continue
            top_pairs.append({"feature_a": f1, "feature_b": f2, "corr": float(value)})

    top_pairs = sorted(top_pairs, key=lambda row: abs(row["corr"]), reverse=True)[:50]
    return CorrelationResult(partners=list_to_native(partners), top_pairs=list_to_native(top_pairs))


def ic_analysis(df: pd.DataFrame, target: pd.Series) -> list[dict[str, Any]]:
    if df.empty:
        return []
    rows: list[dict[str, Any]] = []
    target = pd.to_numeric(target, errors="coerce")
    for col in df.columns:
        series = pd.to_numeric(df[col], errors="coerce")
        ic_value = series.corr(target, method="spearman")
        if pd.isna(ic_value):
            continue
        rows.append({"feature": col, "ic": float(ic_value)})
    rows.sort(key=lambda row: abs(row["ic"]), reverse=True)
    return list_to_native(rows)


def mutual_information(df: pd.DataFrame, target: pd.Series) -> list[dict[str, Any]]:
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

    if df.empty:
        return []

    target_kind = infer_target_kind(target)
    x = df.fillna(0.0)
    if target_kind == "continuous":
        scores = mutual_info_regression(x, target, random_state=42)
    else:
        scores = mutual_info_classif(x, target, random_state=42)
    rows = [
        {"feature": feature, "mi": float(score)}
        for feature, score in zip(df.columns, scores, strict=False)
    ]
    rows.sort(key=lambda row: row["mi"], reverse=True)
    return list_to_native(rows)


def tree_importance(df: pd.DataFrame, target: pd.Series) -> list[dict[str, Any]]:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    if df.empty:
        return []

    target_kind = infer_target_kind(target)
    x = df.fillna(df.median(numeric_only=True)).fillna(0.0)
    if target_kind == "continuous":
        model = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        )
    else:
        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        )
    model.fit(x, target)
    rows = [
        {"feature": feature, "importance": float(score)}
        for feature, score in zip(df.columns, model.feature_importances_, strict=False)
    ]
    rows.sort(key=lambda row: row["importance"], reverse=True)
    return list_to_native(rows)


def vif_analysis(df: pd.DataFrame, max_features: int = 50) -> list[dict[str, Any]]:
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
    except ImportError:
        return []

    if df.shape[1] == 0:
        return []

    if df.shape[1] > max_features:
        df = df.loc[:, df.var().sort_values(ascending=False).head(max_features).index]

    x = df.fillna(df.median(numeric_only=True)).fillna(0.0)
    x = x.assign(constant=1.0)
    values = x.values
    rows: list[dict[str, Any]] = []
    for idx, feature in enumerate(x.columns):
        if feature == "constant":
            continue
        vif = variance_inflation_factor(values, idx)
        rows.append({"feature": feature, "vif": float(vif)})
    rows.sort(key=lambda row: row["vif"], reverse=True)
    return list_to_native(rows)


def drift_mean_cv(
    df: pd.DataFrame,
    time_column: pd.Series,
    bins: int = 5,
) -> list[dict[str, Any]]:
    time_series = ensure_datetime(time_column)
    if time_series.isna().all():
        return []

    frame = df.copy()
    frame["_time"] = time_series
    frame = frame.dropna(subset=["_time"]).sort_values("_time")
    if frame.empty:
        return []

    labels = pd.qcut(frame["_time"].rank(method="first"), q=bins, labels=False, duplicates="drop")
    frame["_bin"] = labels

    rows: list[dict[str, Any]] = []
    for col in df.columns:
        series = frame[[col, "_bin"]].dropna()
        if series.empty:
            continue
        mean_by_bin = series.groupby("_bin")[col].mean()
        if mean_by_bin.empty:
            continue
        overall_mean = mean_by_bin.mean()
        if overall_mean == 0:
            mean_cv = np.nan
        else:
            mean_cv = float(mean_by_bin.std() / abs(overall_mean))
        rows.append({"feature": col, "mean_cv": mean_cv})

    rows.sort(key=lambda row: (np.nan_to_num(row["mean_cv"], nan=-np.inf)), reverse=True)
    return list_to_native(rows)


def leakage_checks(df: pd.DataFrame, target: pd.Series) -> list[dict[str, Any]]:
    if df.empty:
        return []
    rows: list[dict[str, Any]] = []
    target_numeric = pd.to_numeric(target, errors="coerce")
    for col in df.columns:
        series = pd.to_numeric(df[col], errors="coerce")
        corr = series.corr(target_numeric)
        equal_frac = float((series == target_numeric).mean())
        rows.append(
            {
                "feature": col,
                "corr_to_target": None if pd.isna(corr) else float(corr),
                "equal_frac": equal_frac,
            }
        )
    rows.sort(key=lambda row: abs(row["corr_to_target"] or 0.0), reverse=True)
    return list_to_native(rows)


def pca_analysis(df: pd.DataFrame, n_components: int = 6) -> dict[str, Any]:
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return {"components": []}

    if df.empty:
        return {"components": []}

    limit = min(df.shape[0], df.shape[1])
    if limit < 1:
        return {"components": []}

    components = min(n_components, limit)
    x = df.fillna(df.median(numeric_only=True)).fillna(0.0)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    pca = PCA(n_components=components, random_state=42)
    pca.fit(x_scaled)

    rows: list[dict[str, Any]] = []
    cumulative = 0.0
    for idx, variance in enumerate(pca.explained_variance_ratio_):
        cumulative += float(variance)
        rows.append(
            {
                "component": idx + 1,
                "explained_variance": float(variance),
                "cumulative_variance": float(cumulative),
            }
        )

    return {"components": list_to_native(rows)}
