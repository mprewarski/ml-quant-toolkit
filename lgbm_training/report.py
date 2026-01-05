"""HTML report generator for LightGBM training results."""

from __future__ import annotations

import base64
import html
import io
import warnings
from datetime import datetime
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def render_report(results: dict[str, Any], title: str, dataset_name: str) -> str:
    summary = results.get("summary", {})
    split_config = results.get("split_config", {})
    warnings = results.get("warnings", [])
    metrics_summary = results.get("metrics_summary", {})
    fold_metrics = results.get("fold_metrics", [])
    feature_importance = results.get("feature_importance", [])
    quantile_rows = results.get("quantile_performance", [])
    feature_stability = results.get("feature_stability", [])
    leakage_checks = results.get("leakage_checks", [])
    tree_plot = results.get("tree_plot")

    plots = build_plots(results)

    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #0c1110;
      --panel: #171f1c;
      --panel-2: #111815;
      --accent: #f4b63d;
      --accent-2: #5dd2c1;
      --text: #eef4f0;
      --muted: #aab8b1;
      --danger: #f26d6d;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Source Serif Pro", "Georgia", serif;
      background: radial-gradient(circle at top, #22302a, #0c1110 55%);
      color: var(--text);
    }}
    header {{
      padding: 28px 24px 18px;
      background: linear-gradient(120deg, rgba(93,210,193,0.18), transparent 60%);
      border-bottom: 1px solid rgba(255,255,255,0.08);
    }}
    header h1 {{ margin: 0 0 6px; font-size: 26px; }}
    header p {{ margin: 0; color: var(--muted); }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
      margin-top: 16px;
    }}
    .summary-card {{
      padding: 10px 12px;
      background: var(--panel);
      border-radius: 12px;
      border: 1px solid rgba(255,255,255,0.08);
    }}
    nav {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      padding: 10px 24px;
      background: var(--panel-2);
      border-bottom: 1px solid rgba(255,255,255,0.06);
    }}
    nav a {{
      color: var(--text);
      text-decoration: none;
      font-size: 12px;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(255,255,255,0.08);
    }}
    main {{ padding: 24px; display: grid; gap: 24px; }}
    section {{
      background: var(--panel);
      border-radius: 16px;
      padding: 18px;
      border: 1px solid rgba(255,255,255,0.08);
    }}
    section h2 {{ margin: 0 0 12px; font-size: 20px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 8px 6px; border-bottom: 1px solid rgba(255,255,255,0.08); text-align: left; }}
    th {{ color: var(--accent-2); }}
    .muted {{ color: var(--muted); }}
    .warning {{
      background: rgba(242,109,109,0.12);
      border: 1px solid rgba(242,109,109,0.3);
      color: #ffdede;
      padding: 10px 12px;
      border-radius: 12px;
      margin-top: 12px;
    }}
    .plot-grid {{ display: grid; gap: 16px; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); }}
    .plot-card {{ background: rgba(255,255,255,0.04); padding: 12px; border-radius: 12px; }}
    .plot-card img {{ width: 100%; border-radius: 10px; }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <p>Dataset: {html.escape(dataset_name)} | Generated {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}</p>
    <div class=\"summary\">
      {summary_cards(summary, split_config)}
    </div>
    {render_warnings(warnings)}
  </header>
  <nav>
    <a href=\"#metrics\">Metrics</a>
    <a href=\"#folds\">Folds</a>
    <a href=\"#plots\">Plots</a>
    <a href=\"#quantiles\">Quantiles</a>
    <a href=\"#features\">Feature Diagnostics</a>
  </nav>
  <main>
    <section id=\"metrics\">
      <h2>Metrics Summary</h2>
      {render_metrics_summary(metrics_summary)}
    </section>
    <section id=\"folds\">
      <h2>Fold Metrics</h2>
      {render_fold_metrics(fold_metrics)}
    </section>
    <section id=\"plots\">
      <h2>Validation Plots</h2>
      {render_plots(plots, tree_plot)}
    </section>
    <section id=\"quantiles\">
      <h2>Quantile Performance</h2>
      {render_quantiles(quantile_rows)}
    </section>
    <section id=\"features\">
      <h2>Feature Diagnostics</h2>
      {render_feature_tables(feature_importance, feature_stability, leakage_checks)}
    </section>
  </main>
</body>
</html>"""


def summary_cards(summary: dict[str, Any], split_config: dict[str, Any]) -> str:
    cards = {
        "Rows": summary.get("rows"),
        "Features": summary.get("features"),
        "Task": summary.get("task"),
        "Folds": summary.get("folds"),
        "Train Length": split_config.get("training_length"),
        "Holdout Length": split_config.get("holdout_length"),
        "Gap": split_config.get("gap"),
    }
    blocks = []
    for label, value in cards.items():
        if value is None:
            continue
        blocks.append(
            f"<div class=\"summary-card\"><div class=\"muted\">{html.escape(str(label))}</div>"
            f"<div>{html.escape(str(value))}</div></div>"
        )
    return "".join(blocks)


def render_warnings(warnings: list[str]) -> str:
    if not warnings:
        return ""
    content = "".join(f"<div>{html.escape(msg)}</div>" for msg in warnings)
    return f"<div class=\"warning\">{content}</div>"


def render_metrics_summary(metrics_summary: dict[str, Any]) -> str:
    if not metrics_summary:
        return "<p class=\"muted\">No metrics summary available.</p>"
    rows = "".join(
        f"<tr><td>{html.escape(metric)}</td><td>{fmt(stats.get('mean'))}</td><td>{fmt(stats.get('std'))}</td></tr>"
        for metric, stats in metrics_summary.items()
    )
    return (
        "<table><thead><tr><th>Metric</th><th>Mean</th><th>Std</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def render_fold_metrics(fold_metrics: list[dict[str, Any]]) -> str:
    if not fold_metrics:
        return "<p class=\"muted\">No fold metrics available.</p>"
    columns = list(fold_metrics[0].keys())
    header = "".join(f"<th>{html.escape(col)}</th>" for col in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{fmt(row.get(col))}</td>" for col in columns) + "</tr>"
        for row in fold_metrics
    )
    return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"


def render_plots(plots: dict[str, str], tree_plot: str | None) -> str:
    items = []
    for label, data_uri in plots.items():
        items.append(
            "<div class=\"plot-card\">"
            f"<div class=\"muted\">{html.escape(label)}</div>"
            f"<img src=\"data:image/png;base64,{data_uri}\" alt=\"{html.escape(label)}\" />"
            "</div>"
        )
    if tree_plot:
        items.append(
            "<div class=\"plot-card\">"
            "<div class=\"muted\">Tree Diagram</div>"
            f"<img src=\"data:image/png;base64,{tree_plot}\" alt=\"Tree Diagram\" />"
            "</div>"
        )
    if not items:
        return "<p class=\"muted\">No plots available.</p>"
    return f"<div class=\"plot-grid\">{''.join(items)}</div>"


def render_quantiles(quantile_rows: list[dict[str, Any]]) -> str:
    if not quantile_rows:
        return "<p class=\"muted\">No quantile analysis available.</p>"
    columns = list(quantile_rows[0].keys())
    header = "".join(f"<th>{html.escape(col)}</th>" for col in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{fmt(row.get(col))}</td>" for col in columns) + "</tr>"
        for row in quantile_rows
    )
    return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"


def render_feature_tables(
    feature_importance: list[dict[str, Any]],
    feature_stability: list[dict[str, Any]],
    leakage_checks: list[dict[str, Any]],
) -> str:
    sections = []
    sections.append("<h3>Feature Importance</h3>")
    sections.append(render_simple_table(feature_importance))
    sections.append("<h3>Stability</h3>")
    sections.append(render_simple_table(feature_stability))
    sections.append("<h3>Leakage Checks</h3>")
    sections.append(render_simple_table(leakage_checks))
    return "".join(sections)


def render_simple_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p class=\"muted\">No data available.</p>"
    columns = list(rows[0].keys())
    header = "".join(f"<th>{html.escape(col)}</th>" for col in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{fmt(row.get(col))}</td>" for col in columns) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"


def build_plots(results: dict[str, Any]) -> dict[str, str]:
    plots: dict[str, str] = {}
    task = results.get("summary", {}).get("task")
    predictions = results.get("predictions") or {}
    y_true = np.array(predictions.get("y_true") or [])
    y_pred = np.array(predictions.get("y_pred") or [])
    y_prob = predictions.get("y_prob")

    if task == "classification" and isinstance(y_prob, list):
        y_prob_arr = np.array(y_prob)
        if y_prob_arr.ndim == 1:
            plots["ROC Curve"] = plot_roc_curve(y_true, y_prob_arr)
            plots["Precision Recall"] = plot_pr_curve(y_true, y_prob_arr)
            plots["Calibration"] = plot_calibration(results)
            plots["Lift Chart"] = plot_lift(results)
        plots["Quantile Lift"] = plot_quantile_bars(results)
    if task == "regression":
        plots["Prediction vs Truth"] = plot_pred_scatter(y_true, y_pred)
        plots["Quantile Lift"] = plot_quantile_bars(results)

    if results.get("feature_importance"):
        plots["Feature Importance"] = plot_feature_importance(results["feature_importance"])

    return plots


def plot_feature_importance(rows: list[dict[str, Any]]) -> str:
    features = [row["feature"] for row in rows][::-1]
    values = [row["importance"] for row in rows][::-1]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(features, values, color="#5dd2c1")
    ax.set_xlabel("Importance")
    ax.set_title("Top Features")
    fig.tight_layout()
    return fig_to_base64(fig)


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray) -> str:
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fpr, tpr, color="#f4b63d", label="ROC")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#aab8b1")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    fig.tight_layout()
    return fig_to_base64(fig)


def plot_pr_curve(y_true: np.ndarray, y_prob: np.ndarray) -> str:
    from sklearn.metrics import precision_recall_curve

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(recall, precision, color="#5dd2c1")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    fig.tight_layout()
    return fig_to_base64(fig)


def plot_calibration(results: dict[str, Any]) -> str:
    calib = results.get("calibration") or {}
    mean_pred = np.array(calib.get("mean_pred") or [])
    frac_pos = np.array(calib.get("fraction_pos") or [])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(mean_pred, frac_pos, marker="o", color="#f4b63d")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#aab8b1")
    ax.set_xlabel("Mean Predicted")
    ax.set_ylabel("Fraction Positive")
    fig.tight_layout()
    return fig_to_base64(fig)


def plot_lift(results: dict[str, Any]) -> str:
    lift = results.get("lift_curve") or {}
    fraction = np.array(lift.get("fraction") or [])
    capture = np.array(lift.get("capture") or [])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fraction, capture, color="#5dd2c1")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#aab8b1")
    ax.set_xlabel("Fraction of Sample")
    ax.set_ylabel("Fraction of Events")
    fig.tight_layout()
    return fig_to_base64(fig)


def plot_quantile_bars(results: dict[str, Any]) -> str:
    rows = results.get("quantile_performance") or []
    if not rows:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, "No quantile data", ha="center", va="center")
        ax.axis("off")
        return fig_to_base64(fig)
    quantiles = [row["quantile"] for row in rows]
    mean_target = [row.get("mean_target") for row in rows]
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(quantiles, mean_target, color="#f4b63d")
    ax.set_xlabel("Quantile")
    ax.set_ylabel("Mean Target")
    fig.tight_layout()
    return fig_to_base64(fig)


def plot_pred_scatter(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(y_true, y_pred, alpha=0.4, color="#5dd2c1", s=18)
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    fig.tight_layout()
    return fig_to_base64(fig)


def fig_to_base64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return html.escape(str(value))
