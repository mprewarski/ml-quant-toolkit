"""Orchestrates EDA metrics and assembles overview tables."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .metrics import (
    basic_profile,
    correlation_analysis,
    drift_mean_cv,
    ic_analysis,
    leakage_checks,
    mutual_information,
    pca_analysis,
    tree_importance,
    vif_analysis,
)
from .utils import list_to_native, to_native


def analyze_dataset(
    df: pd.DataFrame,
    target_column: str | None = None,
    time_column: str | None = None,
    max_vif_features: int = 50,
    heatmap_features: int = 40,
    analyses: list[str] | None = None,
    pca_components: int = 6,
    exclude_columns: list[str] | None = None,
) -> dict[str, Any]:
    if exclude_columns:
        df = df.drop(columns=exclude_columns, errors="ignore")
    numeric_df = df.select_dtypes(include=["number"])
    enabled = normalize_analyses(analyses)

    summary = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "numeric_columns": int(numeric_df.shape[1]),
        "non_numeric_columns": int(df.shape[1] - numeric_df.shape[1]),
        "missing_total_pct": float(df.isna().mean().mean() * 100.0),
    }

    results: dict[str, Any] = {"summary": summary, "warnings": []}
    results["feature_profile"] = basic_profile(df) if "profile" in enabled else []

    if "correlation" in enabled:
        corr_result = correlation_analysis(numeric_df)
        results["correlation_partners"] = corr_result.partners
        results["correlation_top_pairs"] = corr_result.top_pairs
    else:
        results["correlation_partners"] = []
        results["correlation_top_pairs"] = []

    if "heatmap" in enabled:
        results["corr_heatmap"] = correlation_heatmap(numeric_df, top_n=heatmap_features)
    else:
        results["corr_heatmap"] = {"features": [], "matrix": []}

    target_series = None
    if target_column:
        if target_column not in df.columns:
            results["warnings"].append(f"Target column '{target_column}' not found.")
        else:
            target_series = df[target_column]

    if target_series is not None:
        numeric_features = numeric_df.drop(columns=[target_column], errors="ignore")
        valid_mask = target_series.notna()
        numeric_features = numeric_features.loc[valid_mask]
        target_series = target_series.loc[valid_mask]
        target_numeric = pd.to_numeric(target_series, errors="coerce")
        results["ic"] = ic_analysis(numeric_features, target_numeric) if "ic" in enabled else []
        results["mutual_info"] = (
            mutual_information(numeric_features, target_series) if "mutual_info" in enabled else []
        )
        results["tree_importance"] = (
            tree_importance(numeric_features, target_series) if "tree_importance" in enabled else []
        )
        results["leakage"] = leakage_checks(numeric_features, target_series) if "leakage" in enabled else []
    else:
        results["ic"] = []
        results["mutual_info"] = []
        results["tree_importance"] = []
        results["leakage"] = []

    results["vif"] = vif_analysis(numeric_df, max_features=max_vif_features) if "vif" in enabled else []

    if "drift" in enabled:
        if time_column and time_column in df.columns:
            results["drift"] = drift_mean_cv(numeric_df, df[time_column])
        else:
            if time_column:
                results["warnings"].append(f"Time column '{time_column}' not found.")
            results["drift"] = []
    else:
        results["drift"] = []

    results["pca"] = (
        pca_analysis(numeric_df, n_components=pca_components)
        if "pca" in enabled
        else {"components": []}
    )

    results["overview"] = build_overview_table(results, numeric_df.columns)
    return results


def build_overview_table(results: dict[str, Any], features: pd.Index) -> list[dict[str, Any]]:
    profile_lookup = {row["feature"]: row for row in results.get("feature_profile", [])}
    partner_lookup = {row["feature"]: row for row in results.get("correlation_partners", [])}
    ic_lookup = {row["feature"]: row for row in results.get("ic", [])}
    mi_lookup = {row["feature"]: row for row in results.get("mutual_info", [])}
    tree_lookup = {row["feature"]: row for row in results.get("tree_importance", [])}
    vif_lookup = {row["feature"]: row for row in results.get("vif", [])}
    drift_lookup = {row["feature"]: row for row in results.get("drift", [])}
    leakage_lookup = {row["feature"]: row for row in results.get("leakage", [])}

    rows: list[dict[str, Any]] = []
    for feature in features:
        profile = profile_lookup.get(feature, {})
        partner = partner_lookup.get(feature, {})
        leakage = leakage_lookup.get(feature, {})
        row = {
            "feature": feature,
            "missing_pct": profile.get("missing_pct"),
            "nunique": profile.get("nunique"),
            "corr_partner": partner.get("partner"),
            "corr_partner_abs": partner.get("abs_corr"),
            "ic": ic_lookup.get(feature, {}).get("ic"),
            "mutual_info": mi_lookup.get(feature, {}).get("mi"),
            "tree_importance": tree_lookup.get(feature, {}).get("importance"),
            "vif": vif_lookup.get(feature, {}).get("vif"),
            "mean_cv": drift_lookup.get(feature, {}).get("mean_cv"),
            "leakage_corr": leakage.get("corr_to_target"),
            "leakage_equal_frac": leakage.get("equal_frac"),
        }
        row["flag_poison"] = poison_flag(row)
        rows.append(row)

    return list_to_native(rows)


def poison_flag(row: dict[str, Any]) -> str:
    if row.get("leakage_equal_frac", 0.0) and row["leakage_equal_frac"] >= 0.98:
        return "high"
    if row.get("leakage_corr") is not None and abs(row["leakage_corr"]) >= 0.98:
        return "high"
    if row.get("vif") is not None and row["vif"] >= 10.0:
        return "medium"
    if row.get("corr_partner_abs") is not None and row["corr_partner_abs"] >= 0.9:
        return "medium"
    if row.get("mean_cv") is not None and row["mean_cv"] >= 1.0:
        return "medium"
    return "low"


def correlation_heatmap(df: pd.DataFrame, top_n: int = 40) -> dict[str, Any]:
    if df.shape[1] < 2:
        return {"features": [], "matrix": []}

    corr = df.corr(method="pearson")
    if corr.empty:
        return {"features": [], "matrix": []}

    max_abs = {}
    for feature in corr.columns:
        series = corr[feature].drop(index=feature, errors="ignore").abs()
        if series.empty:
            max_abs[feature] = 0.0
            continue
        value = series.max(skipna=True)
        max_abs[feature] = 0.0 if pd.isna(value) else float(value)

    ranked = sorted(max_abs.items(), key=lambda item: item[1], reverse=True)
    features = [feature for feature, _ in ranked[:top_n] if feature in corr.columns]
    if len(features) < 2:
        return {"features": [], "matrix": []}

    sub = corr.loc[features, features]
    matrix = []
    for row in sub.values.tolist():
        converted = []
        for value in row:
            converted.append(None if pd.isna(value) else to_native(value))
        matrix.append(converted)
    return {"features": features, "matrix": matrix}


def normalize_analyses(analyses: list[str] | None) -> set[str]:
    if not analyses:
        return {
            "profile",
            "correlation",
            "heatmap",
            "ic",
            "mutual_info",
            "tree_importance",
            "vif",
            "drift",
            "leakage",
            "pca",
        }
    cleaned = {item.strip().lower() for item in analyses if item.strip()}
    return cleaned
