"""Model training and validation orchestration for LightGBM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import base64
import io

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor, plot_tree
from sklearn.preprocessing import LabelEncoder

from .metrics import (
    calibration_data,
    classification_metrics,
    lift_curve_data,
    quantile_performance,
    regression_metrics,
)
from .utils import ensure_datetime, list_to_native


@dataclass
class FoldSplit:
    train_idx: np.ndarray
    holdout_idx: np.ndarray
    meta: dict[str, Any]


def train_lightgbm(
    df: pd.DataFrame,
    target_column: str,
    time_column: str | None,
    task: str | None,
    params: dict[str, Any] | None,
    gap: int,
    holdout_length: int,
    training_length: int,
    n_folds: int,
    threshold: float,
    exclude_columns: list[str] | None = None,
) -> dict[str, Any]:
    warnings: list[str] = []
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    if exclude_columns:
        df = df.drop(columns=exclude_columns, errors="ignore")

    if time_column and time_column in df.columns:
        df = df.copy()
        df[time_column] = ensure_datetime(df[time_column])
        missing_time = df[time_column].isna().sum()
        if missing_time:
            warnings.append(f"Dropped {missing_time} rows with invalid time values.")
            df = df.loc[df[time_column].notna()]
        df = df.sort_values(time_column)
    else:
        if time_column:
            warnings.append(f"Time column '{time_column}' not found. Using row order.")
        time_column = None

    df = df.dropna(subset=[target_column])
    if df.empty:
        raise ValueError("No rows remain after dropping missing target values.")

    feature_columns = [
        col
        for col in df.columns
        if col not in {target_column, time_column}
    ]
    if not feature_columns:
        raise ValueError("No feature columns remain after exclusions.")

    X = df[feature_columns].copy()
    X, name_notes, name_map = sanitize_feature_names(X)
    warnings.extend(name_notes)
    model_feature_columns = list(X.columns)
    y = df[target_column]

    inferred_task, label_encoder = infer_task(y, task)
    task = inferred_task

    if task == "classification":
        y_encoded = pd.Series(label_encoder.transform(y.astype(str)), index=y.index)
    else:
        y_encoded = pd.to_numeric(y, errors="coerce")
        if y_encoded.isna().any():
            missing = int(y_encoded.isna().sum())
            warnings.append(f"Dropped {missing} rows with non-numeric target values.")
            df = df.loc[y_encoded.notna()]
            X = df[feature_columns]
            y_encoded = y_encoded.loc[df.index]

    X, encoding_notes = ensure_numeric_features(X)
    warnings.extend(encoding_notes)

    splits = build_splits(
        df,
        time_column,
        gap=gap,
        training_length=training_length,
        holdout_length=holdout_length,
        n_folds=n_folds,
    )

    if not splits:
        raise ValueError("Unable to build any train/holdout splits with the provided settings.")

    if len(splits) < n_folds:
        warnings.append(
            f"Requested {n_folds} folds, but only {len(splits)} fit the data length."
        )

    num_classes = len(label_encoder.classes_) if task == "classification" else None
    params = build_params(task, params, num_classes=num_classes)

    fold_metrics: list[dict[str, Any]] = []
    split_meta: list[dict[str, Any]] = []
    last_predictions: dict[str, Any] | None = None
    last_importance: list[dict[str, Any]] = []
    tree_plot: str | None = None

    for fold_idx, split in enumerate(splits, start=1):
        X_train = X.iloc[split.train_idx]
        y_train = y_encoded.iloc[split.train_idx]
        X_holdout = X.iloc[split.holdout_idx]
        y_holdout = y_encoded.iloc[split.holdout_idx]

        model = build_model(task, params)
        model.fit(X_train, y_train)

        if task == "classification":
            y_prob = model.predict_proba(X_holdout)
            if y_prob.shape[1] == 2:
                y_prob_pos = y_prob[:, 1]
                y_pred = (y_prob_pos >= threshold).astype(int)
                metrics = classification_metrics(
                    y_holdout.to_numpy(),
                    y_pred,
                    y_prob_pos,
                    average="binary",
                )
                holdout_pred = {
                    "y_true": y_holdout.to_numpy().tolist(),
                    "y_pred": y_pred.tolist(),
                    "y_prob": y_prob_pos.tolist(),
                }
            else:
                y_pred = np.argmax(y_prob, axis=1)
                metrics = classification_metrics(
                    y_holdout.to_numpy(),
                    y_pred,
                    y_prob,
                    average="macro",
                    labels=list(range(y_prob.shape[1])),
                )
                holdout_pred = {
                    "y_true": y_holdout.to_numpy().tolist(),
                    "y_pred": y_pred.tolist(),
                    "y_prob": y_prob.tolist(),
                }
        else:
            y_pred = model.predict(X_holdout)
            metrics = regression_metrics(y_holdout.to_numpy(), y_pred)
            holdout_pred = {
                "y_true": y_holdout.to_numpy().tolist(),
                "y_pred": y_pred.tolist(),
                "y_prob": None,
            }

        metrics["fold"] = fold_idx
        metrics["train_rows"] = int(len(split.train_idx))
        metrics["holdout_rows"] = int(len(split.holdout_idx))
        fold_metrics.append(metrics)
        split_meta.append(split.meta)

        if fold_idx == len(splits):
            last_predictions = holdout_pred
            last_importance = feature_importances(model, model_feature_columns)
            last_importance = remap_feature_importance(last_importance, name_map)
            tree_plot = render_tree_plot(model)

    summary_metrics = summarize_metrics(fold_metrics)

    quantiles = []
    lift_data = None
    calibration = None
    if last_predictions is not None:
        if task == "classification":
            y_score = np.array(last_predictions["y_prob"]) if last_predictions["y_prob"] else None
            if y_score is not None and y_score.ndim == 1:
                quantiles = quantile_performance(
                    np.array(last_predictions["y_true"]),
                    y_score,
                    task=task,
                )
                lift_data = lift_curve_data(np.array(last_predictions["y_true"]), y_score)
                calibration = calibration_data(np.array(last_predictions["y_true"]), y_score)
        else:
            y_score = np.array(last_predictions["y_pred"])
            quantiles = quantile_performance(
                np.array(last_predictions["y_true"]),
                y_score,
                task=task,
            )

    stability = feature_stability(df, time_column, feature_columns)
    leakage = leakage_checks(df, time_column, target_column, feature_columns)

    return {
        "summary": {
            "rows": int(df.shape[0]),
            "features": int(len(feature_columns)),
            "task": task,
            "folds": len(splits),
        },
        "split_config": {
            "gap": gap,
            "holdout_length": holdout_length,
            "training_length": training_length,
            "n_folds": n_folds,
        },
        "splits": list_to_native(split_meta),
        "fold_metrics": list_to_native(fold_metrics),
        "metrics_summary": summary_metrics,
        "predictions": last_predictions,
        "feature_importance": list_to_native(last_importance),
        "feature_name_map": name_map,
        "quantile_performance": quantiles,
        "lift_curve": lift_data,
        "calibration": calibration,
        "tree_plot": tree_plot,
        "feature_stability": stability,
        "leakage_checks": leakage,
        "warnings": warnings,
        "class_labels": label_encoder.classes_.tolist() if task == "classification" else None,
    }


def infer_task(y: pd.Series, task: str | None) -> tuple[str, LabelEncoder]:
    encoder = LabelEncoder()
    if task:
        task_clean = task.strip().lower()
    else:
        task_clean = ""

    if task_clean in {"classification", "regression"}:
        if task_clean == "classification":
            encoder.fit(y.astype(str))
        return task_clean, encoder

    if pd.api.types.is_numeric_dtype(y):
        nunique = y.nunique(dropna=True)
        if nunique <= 10:
            encoder.fit(y.astype(str))
            return "classification", encoder
        return "regression", encoder

    encoder.fit(y.astype(str))
    return "classification", encoder


def build_params(
    task: str,
    params: dict[str, Any] | None,
    num_classes: int | None = None,
) -> dict[str, Any]:
    base = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "force_col_wise": True,
    }
    if task == "classification":
        base["objective"] = "binary"
        if num_classes and num_classes > 2:
            base["objective"] = "multiclass"
            base["num_class"] = num_classes
    else:
        base["objective"] = "regression"
    if params:
        base.update(params)
    return base


def build_model(task: str, params: dict[str, Any]) -> Any:
    if task == "classification":
        return LGBMClassifier(**params)
    return LGBMRegressor(**params)


def build_splits(
    df: pd.DataFrame,
    time_column: str | None,
    gap: int,
    training_length: int,
    holdout_length: int,
    n_folds: int,
) -> list[FoldSplit]:
    if training_length <= 0 or holdout_length <= 0 or n_folds <= 0:
        raise ValueError("Training length, holdout length, and folds must be positive.")
    if gap < 0:
        raise ValueError("Gap must be zero or positive.")

    indices = np.arange(len(df))

    if time_column and time_column in df.columns:
        periods = ensure_datetime(df[time_column]).dt.to_period("W")
        period_list = []
        seen = set()
        for period in periods:
            if pd.isna(period):
                continue
            if period not in seen:
                period_list.append(period)
                seen.add(period)
        return build_period_splits(indices, periods, period_list, gap, training_length, holdout_length, n_folds)

    return build_index_splits(indices, gap, training_length, holdout_length, n_folds)


def build_period_splits(
    indices: np.ndarray,
    periods: pd.Series,
    period_list: list[pd.Period],
    gap: int,
    training_length: int,
    holdout_length: int,
    n_folds: int,
) -> list[FoldSplit]:
    splits: list[FoldSplit] = []
    total_periods = len(period_list)
    for fold in range(n_folds):
        holdout_end = total_periods - fold * holdout_length
        holdout_start = holdout_end - holdout_length
        gap_start = holdout_start - gap
        train_end = gap_start
        train_start = train_end - training_length
        if train_start < 0:
            break
        train_periods = set(period_list[train_start:train_end])
        holdout_periods = set(period_list[holdout_start:holdout_end])
        train_idx = indices[periods.isin(train_periods)]
        holdout_idx = indices[periods.isin(holdout_periods)]
        if len(train_idx) == 0 or len(holdout_idx) == 0:
            continue
        meta = {
            "fold": fold + 1,
            "train_period_start": str(period_list[train_start]),
            "train_period_end": str(period_list[train_end - 1]),
            "holdout_period_start": str(period_list[holdout_start]),
            "holdout_period_end": str(period_list[holdout_end - 1]),
            "train_rows": int(len(train_idx)),
            "holdout_rows": int(len(holdout_idx)),
        }
        splits.append(FoldSplit(train_idx=train_idx, holdout_idx=holdout_idx, meta=meta))
    return splits


def build_index_splits(
    indices: np.ndarray,
    gap: int,
    training_length: int,
    holdout_length: int,
    n_folds: int,
) -> list[FoldSplit]:
    splits: list[FoldSplit] = []
    total = len(indices)
    for fold in range(n_folds):
        holdout_end = total - fold * holdout_length
        holdout_start = holdout_end - holdout_length
        gap_start = holdout_start - gap
        train_end = gap_start
        train_start = train_end - training_length
        if train_start < 0:
            break
        train_idx = indices[train_start:train_end]
        holdout_idx = indices[holdout_start:holdout_end]
        if len(train_idx) == 0 or len(holdout_idx) == 0:
            continue
        meta = {
            "fold": fold + 1,
            "train_index_start": int(train_start),
            "train_index_end": int(train_end - 1),
            "holdout_index_start": int(holdout_start),
            "holdout_index_end": int(holdout_end - 1),
            "train_rows": int(len(train_idx)),
            "holdout_rows": int(len(holdout_idx)),
        }
        splits.append(FoldSplit(train_idx=train_idx, holdout_idx=holdout_idx, meta=meta))
    return splits


def summarize_metrics(fold_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    if not fold_metrics:
        return {}
    keys = [key for key in fold_metrics[0].keys() if key not in {"fold", "train_rows", "holdout_rows"}]
    summary: dict[str, Any] = {}
    for key in keys:
        values = [metric[key] for metric in fold_metrics if metric.get(key) is not None]
        if not values:
            continue
        arr = np.array(values, dtype=float)
        summary[key] = {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}
    return summary


def feature_importances(model: Any, feature_columns: list[str], top_n: int = 30) -> list[dict[str, Any]]:
    if not hasattr(model, "feature_importances_"):
        return []
    importances = model.feature_importances_
    rows = [
        {"feature": feature, "importance": float(score)}
        for feature, score in zip(feature_columns, importances)
    ]
    rows.sort(key=lambda item: item["importance"], reverse=True)
    return rows[:top_n]


def remap_feature_importance(
    rows: list[dict[str, Any]],
    name_map: dict[str, str],
) -> list[dict[str, Any]]:
    if not rows or not name_map:
        return rows
    inverse = {sanitized: original for original, sanitized in name_map.items()}
    remapped = []
    for row in rows:
        feature = row.get("feature")
        if feature in inverse:
            remapped.append({**row, "feature": inverse[feature]})
        else:
            remapped.append(row)
    return remapped


def sanitize_feature_names(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    mapping: dict[str, str] = {}
    notes: list[str] = []
    new_columns: list[str] = []
    seen: dict[str, int] = {}
    for col in df.columns:
        safe = "".join(ch if ch.isalnum() or ch in {"_", "."} else "_" for ch in str(col))
        if not safe:
            safe = "feature"
        if safe in seen:
            seen[safe] += 1
            safe = f"{safe}_{seen[safe]}"
        else:
            seen[safe] = 0
        mapping[str(col)] = safe
        new_columns.append(safe)
    if any(orig != mapping[orig] for orig in mapping):
        notes.append("Sanitized feature names to avoid unsupported JSON characters.")
    df = df.copy()
    df.columns = new_columns
    return df, notes, mapping


def ensure_numeric_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    notes: list[str] = []
    converted = df.copy()
    non_numeric = [
        col
        for col in converted.columns
        if not pd.api.types.is_numeric_dtype(converted[col])
        and not pd.api.types.is_bool_dtype(converted[col])
    ]
    if non_numeric:
        for col in non_numeric:
            converted[col] = converted[col].astype("category").cat.codes
        notes.append(f"Encoded {len(non_numeric)} non-numeric feature(s) as categorical codes.")
    return converted, notes


def feature_stability(
    df: pd.DataFrame,
    time_column: str | None,
    feature_columns: list[str],
) -> list[dict[str, Any]]:
    if not time_column or time_column not in df.columns:
        return []

    periods = ensure_datetime(df[time_column]).dt.to_period("W")
    numeric_features = df[feature_columns].select_dtypes(include=["number"])
    if numeric_features.empty:
        return []

    grouped = numeric_features.groupby(periods)
    means = grouped.mean()
    stds = grouped.std()

    rows: list[dict[str, Any]] = []
    for feature in numeric_features.columns:
        mean_series = means[feature].dropna()
        std_series = stds[feature].dropna()
        if mean_series.empty:
            continue
        mean_cv = float(mean_series.std(ddof=0) / (mean_series.mean() or 1.0))
        std_cv = float(std_series.std(ddof=0) / (std_series.mean() or 1.0)) if not std_series.empty else None
        rows.append({"feature": feature, "mean_cv": mean_cv, "std_cv": std_cv})

    rows.sort(key=lambda item: abs(item.get("mean_cv", 0.0)), reverse=True)
    return list_to_native(rows[:30])


def leakage_checks(
    df: pd.DataFrame,
    time_column: str | None,
    target_column: str,
    feature_columns: list[str],
) -> list[dict[str, Any]]:
    if not time_column or time_column not in df.columns:
        return []

    df = df.sort_values(time_column)
    target = pd.to_numeric(df[target_column], errors="coerce")
    target_future = target.shift(-1)
    numeric_features = df[feature_columns].select_dtypes(include=["number"])

    rows: list[dict[str, Any]] = []
    if target_future.nunique(dropna=True) <= 1:
        return []
    for feature in numeric_features.columns:
        feature_series = pd.to_numeric(numeric_features[feature], errors="coerce")
        if feature_series.nunique(dropna=True) <= 1:
            continue
        corr = feature_series.corr(target_future)
        if pd.isna(corr):
            continue
        rows.append({"feature": feature, "future_target_corr": float(corr)})

    rows.sort(key=lambda item: abs(item["future_target_corr"]), reverse=True)
    return list_to_native(rows[:30])


def render_tree_plot(model: Any) -> str | None:
    if not hasattr(model, "booster_"):
        return None
    try:
        fig, ax = plt.subplots(figsize=(12, 8))
        plot_tree(model, ax=ax, tree_index=0)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None
