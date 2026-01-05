"""Metrics and diagnostics for LightGBM training runs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from .utils import list_to_native, to_native


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    average: str = "binary",
    labels: list[int] | None = None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
    }

    if y_prob is not None:
        if average == "binary":
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
            metrics["pr_auc"] = average_precision_score(y_true, y_prob)
            metrics["log_loss"] = log_loss(y_true, y_prob)
            metrics["brier"] = brier_score_loss(y_true, y_prob)
        else:
            metrics["roc_auc"] = roc_auc_score(
                y_true,
                y_prob,
                multi_class="ovr",
                average=average,
                labels=labels,
            )
            metrics["log_loss"] = log_loss(y_true, y_prob, labels=labels)
    return {key: to_native(value) for key, value in metrics.items()}


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    spearman = spearmanr(y_true, y_pred, nan_policy="omit").correlation
    return {
        "rmse": to_native(rmse),
        "mae": to_native(mae),
        "r2": to_native(r2),
        "spearman": to_native(spearman),
    }


def quantile_performance(
    y_true: np.ndarray,
    y_score: np.ndarray,
    task: str,
    bins: int = 10,
) -> list[dict[str, Any]]:
    if len(y_true) == 0:
        return []
    quantiles = pd.qcut(y_score, q=bins, labels=False, duplicates="drop")
    df = pd.DataFrame({"y_true": y_true, "y_score": y_score, "quantile": quantiles})
    rows: list[dict[str, Any]] = []
    for quantile, group in df.groupby("quantile"):
        row = {
            "quantile": int(quantile),
            "count": int(group.shape[0]),
            "mean_score": float(group["y_score"].mean()),
            "mean_target": float(group["y_true"].mean()),
        }
        if task == "classification":
            row["event_rate"] = float(group["y_true"].mean())
        rows.append(row)
    return list_to_native(rows)


def lift_curve_data(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    if len(y_true) == 0:
        return {"fraction": [], "capture": []}
    order = np.argsort(y_prob)[::-1]
    y_sorted = y_true[order]
    total_events = max(1, int(y_sorted.sum()))
    cumulative_events = np.cumsum(y_sorted)
    fractions = np.arange(1, len(y_sorted) + 1) / len(y_sorted)
    capture = cumulative_events / total_events
    return {"fraction": fractions.tolist(), "capture": capture.tolist()}


def calibration_data(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> dict[str, Any]:
    if len(y_true) == 0:
        return {"mean_pred": [], "fraction_pos": []}
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=bins, strategy="quantile")
    return {"mean_pred": mean_pred.tolist(), "fraction_pos": frac_pos.tolist()}
