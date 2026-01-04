"""HTML report generator for EDA results."""

from __future__ import annotations

import html
import json
from datetime import datetime
from typing import Any


def render_report(results: dict[str, Any], title: str, dataset_name: str) -> str:
    summary = results.get("summary", {})
    warnings = results.get("warnings", [])
    overview_rows = results.get("overview", [])

    sections = [
        section_overview(overview_rows),
        section_heatmap(results.get("corr_heatmap", {})),
        section_top_pairs(results.get("correlation_top_pairs", [])),
        section_ic(results.get("ic", [])),
        section_mutual_info(results.get("mutual_info", [])),
        section_tree_importance(results.get("tree_importance", [])),
        section_vif(results.get("vif", [])),
        section_drift(results.get("drift", [])),
        section_leakage(results.get("leakage", [])),
        section_pca(results.get("pca", {})),
    ]

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #0f1418;
      --panel: #1b2228;
      --panel-2: #141a1f;
      --accent: #f4b63d;
      --accent-2: #7ac7ff;
      --text: #e6edf3;
      --muted: #9fb0bd;
      --danger: #ff6b6b;
      --warn: #f4b63d;
      --ok: #60d394;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino", "Palatino Linotype", serif;
      background: radial-gradient(circle at top, #24303a, #0f1418 40%);
      color: var(--text);
    }}
    header {{
      padding: 32px 24px 12px;
      border-bottom: 1px solid rgba(255,255,255,0.08);
      background: linear-gradient(120deg, rgba(244,182,61,0.12), transparent 60%);
    }}
    header h1 {{
      margin: 0 0 6px;
      font-size: 28px;
      letter-spacing: 0.4px;
    }}
    header p {{
      margin: 0;
      color: var(--muted);
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
      margin: 18px 0 0;
    }}
    .summary-card {{
      padding: 12px 14px;
      background: var(--panel);
      border-radius: 12px;
      border: 1px solid rgba(255,255,255,0.06);
    }}
    nav {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      padding: 12px 24px;
      background: var(--panel-2);
      border-bottom: 1px solid rgba(255,255,255,0.06);
    }}
    nav a {{
      color: var(--text);
      text-decoration: none;
      background: rgba(255,255,255,0.06);
      padding: 6px 10px;
      border-radius: 999px;
      font-size: 13px;
    }}
    main {{
      padding: 24px;
      display: grid;
      gap: 24px;
    }}
    section {{
      background: var(--panel);
      border-radius: 16px;
      padding: 18px;
      border: 1px solid rgba(255,255,255,0.06);
    }}
    section h2 {{
      margin: 0 0 12px;
      font-size: 20px;
    }}
    .table-controls {{
      display: flex;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 10px;
    }}
    .table-controls input {{
      background: #0b1116;
      border: 1px solid rgba(255,255,255,0.12);
      color: var(--text);
      padding: 6px 10px;
      border-radius: 8px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      padding: 8px 6px;
      border-bottom: 1px solid rgba(255,255,255,0.06);
      text-align: left;
    }}
    th {{
      cursor: pointer;
      color: var(--accent-2);
      position: sticky;
      top: 0;
      background: var(--panel);
      z-index: 1;
    }}
    tr:hover td {{
      background: rgba(255,255,255,0.04);
    }}
    .indicator {{
      display: inline-block;
      width: 10px;
      height: 10px;
      margin-right: 6px;
      border-radius: 4px;
    }}
    .flag-low {{ background: var(--ok); border-radius: 999px; }}
    .flag-medium {{ background: var(--warn); }}
    .flag-high {{ background: var(--danger); transform: rotate(45deg); }}
    .muted {{ color: var(--muted); }}
    .warning {{
      background: rgba(255,107,107,0.12);
      color: #ffd3d3;
      padding: 10px 12px;
      border-radius: 10px;
      border: 1px solid rgba(255,107,107,0.3);
      margin-top: 12px;
    }}
    .heatmap-wrapper {{
      overflow: auto;
      border-radius: 12px;
      border: 1px solid rgba(255,255,255,0.08);
    }}
    .heatmap-grid {{
      display: grid;
      gap: 2px;
      background: rgba(255,255,255,0.06);
      padding: 8px;
      min-width: max-content;
    }}
    .heatmap-cell {{
      width: 18px;
      height: 18px;
      border-radius: 4px;
      position: relative;
    }}
    .heatmap-tooltip {{
      position: absolute;
      bottom: 120%;
      left: 50%;
      transform: translateX(-50%);
      background: #0b1116;
      color: var(--text);
      padding: 4px 6px;
      border-radius: 6px;
      font-size: 11px;
      white-space: nowrap;
      border: 1px solid rgba(255,255,255,0.2);
      display: none;
      z-index: 2;
    }}
    .heatmap-cell:hover .heatmap-tooltip {{
      display: block;
    }}
    .heatmap-labels {{
      display: grid;
      gap: 2px;
      font-size: 11px;
      color: var(--muted);
      margin-bottom: 8px;
    }}
    .heatmap-labels div {{
      writing-mode: vertical-rl;
      transform: rotate(180deg);
      text-align: left;
      height: 120px;
    }}
    .heatmap-legend {{
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 12px;
      color: var(--muted);
      margin-top: 8px;
    }}
    .legend-bar {{
      height: 10px;
      width: 120px;
      border-radius: 999px;
      background: linear-gradient(90deg, #2f74ff, #0b1116, #ff6b6b);
      border: 1px solid rgba(255,255,255,0.2);
    }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <p>Dataset: {html.escape(dataset_name)}</p>
    <div class="summary">
      {summary_card("Rows", summary.get("rows"))}
      {summary_card("Columns", summary.get("columns"))}
      {summary_card("Numeric", summary.get("numeric_columns"))}
      {summary_card("Missing %", fmt_float(summary.get("missing_total_pct")))}
    </div>
    {warnings_block(warnings)}
  </header>
  <nav>
    <a href="#overview">Overview</a>
    <a href="#heatmap">Correlation Heatmap</a>
    <a href="#correlation">Top Correlations</a>
    <a href="#ic">Information Coefficient</a>
    <a href="#mi">Mutual Information</a>
    <a href="#importance">Tree Importance</a>
    <a href="#vif">VIF</a>
    <a href="#drift">Drift</a>
    <a href="#leakage">Leakage</a>
    <a href="#pca">PCA</a>
  </nav>
  <main>
    {''.join(sections)}
  </main>
  <script>
    const heatmapData = {json.dumps(results.get("corr_heatmap", {}))};

    function colorForCorrelation(value) {{
      if (value === null || value === undefined || Number.isNaN(value)) {{
        return "#11171c";
      }}
      const capped = Math.max(-1, Math.min(1, value));
      const intensity = Math.abs(capped);
      if (capped >= 0) {{
        return "rgba(255, 107, 107, " + (0.15 + intensity * 0.85) + ")";
      }}
      return "rgba(47, 116, 255, " + (0.15 + intensity * 0.85) + ")";
    }}

    function renderHeatmap() {{
      const container = document.getElementById("heatmap-grid");
      if (!container || !heatmapData.features || heatmapData.features.length === 0) {{
        return;
      }}
      const size = heatmapData.features.length;
      container.style.gridTemplateColumns = "repeat(" + size + ", 18px)";
      container.innerHTML = "";
      for (let row = 0; row < size; row++) {{
        for (let col = 0; col < size; col++) {{
          const value = heatmapData.matrix[row][col];
          const cell = document.createElement("div");
          cell.className = "heatmap-cell";
          cell.style.background = colorForCorrelation(value);
          const tooltip = document.createElement("div");
          tooltip.className = "heatmap-tooltip";
          const labelA = heatmapData.features[row];
          const labelB = heatmapData.features[col];
          const display = value === null ? "nan" : value.toFixed(4);
          tooltip.textContent = labelA + " vs " + labelB + ": " + display;
          cell.appendChild(tooltip);
          container.appendChild(cell);
        }}
      }}
    }}

    function sortTable(table, columnIndex) {{
      const tbody = table.tBodies[0];
      const rows = Array.from(tbody.rows);
      const asc = table.getAttribute("data-sort") !== "asc";
      rows.sort((a, b) => {{
        const av = a.cells[columnIndex].dataset.sort || a.cells[columnIndex].innerText;
        const bv = b.cells[columnIndex].dataset.sort || b.cells[columnIndex].innerText;
        const an = parseFloat(av);
        const bn = parseFloat(bv);
        if (!Number.isNaN(an) && !Number.isNaN(bn)) {{
          return asc ? an - bn : bn - an;
        }}
        return asc ? av.localeCompare(bv) : bv.localeCompare(av);
      }});
      rows.forEach(row => tbody.appendChild(row));
      table.setAttribute("data-sort", asc ? "asc" : "desc");
    }}

    document.querySelectorAll("table.sortable th").forEach((th, idx) => {{
      th.addEventListener("click", () => sortTable(th.closest("table"), idx));
    }});

    document.querySelectorAll("[data-filter]").forEach(input => {{
      input.addEventListener("input", () => {{
        const term = input.value.toLowerCase();
        const table = document.getElementById(input.dataset.filter);
        if (!table) return;
        Array.from(table.tBodies[0].rows).forEach(row => {{
          const text = row.innerText.toLowerCase();
          row.style.display = text.includes(term) ? "" : "none";
        }});
      }});
    }});

    renderHeatmap();
  </script>
</body>
</html>
"""


def summary_card(label: str, value: Any) -> str:
    return f"""
    <div class="summary-card">
      <div class="muted">{html.escape(label)}</div>
      <div>{html.escape(str(value))}</div>
    </div>
    """


def warnings_block(warnings: list[str]) -> str:
    if not warnings:
        return ""
    content = "<br />".join(html.escape(warning) for warning in warnings)
    return f'<div class="warning">{content}</div>'


def fmt_float(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


class SafeHtml(str):
    """Marks strings safe for direct HTML insertion."""


def section_overview(rows: list[dict[str, Any]]) -> str:
    headers = [
        "Flag",
        "Feature",
        "Missing %",
        "Nunique",
        "Corr Partner",
        "Abs Corr",
        "IC",
        "Mutual Info",
        "Tree Importance",
        "VIF",
        "Mean CV",
        "Leak Corr",
        "Leak Equal",
    ]
    body_rows = []
    for row in rows:
        flag = row.get("flag_poison", "low")
        indicator = SafeHtml(f'<span class="indicator flag-{flag}"></span>{html.escape(flag)}')
        body_rows.append(
            [
                indicator,
                row.get("feature"),
                fmt_float(row.get("missing_pct")),
                row.get("nunique"),
                row.get("corr_partner"),
                fmt_float(row.get("corr_partner_abs")),
                fmt_float(row.get("ic")),
                fmt_float(row.get("mutual_info")),
                fmt_float(row.get("tree_importance")),
                fmt_float(row.get("vif")),
                fmt_float(row.get("mean_cv")),
                fmt_float(row.get("leakage_corr")),
                fmt_float(row.get("leakage_equal_frac")),
            ]
        )

    return render_section(
        section_id="overview",
        title="Overview Table",
        description=(
            "Consolidated feature health table. Scan missing %, correlation partners, drift, "
            "and leakage indicators together to identify high-risk or redundant features. "
            "Flags: low=ok, medium=review, high=likely poison."
        ),
        headers=headers,
        rows=body_rows,
        filter_id="overview-table",
    )


def section_heatmap(data: dict[str, Any]) -> str:
    features = data.get("features", [])
    if not features:
        return """
        <section id="heatmap">
          <h2>Correlation Heatmap</h2>
          <p class="muted">Not enough numeric data to build a heatmap.</p>
        </section>
        """

    labels = "".join(f"<div>{html.escape(label)}</div>" for label in features)
    return f"""
    <section id="heatmap">
      <h2>Correlation Heatmap</h2>
      <p class="muted">
        Top {len(features)} features by absolute correlation strength. Each cell shows the "
        pairwise Pearson correlation: red = positive, blue = negative, stronger color = "
        stronger relationship. Use this to spot clusters of redundant features.
      </p>
      <div class="heatmap-labels" style="grid-template-columns: repeat({len(features)}, 1fr);">
        {labels}
      </div>
      <div class="heatmap-wrapper">
        <div id="heatmap-grid" class="heatmap-grid"></div>
      </div>
      <div class="heatmap-legend">
        <span>-1</span>
        <div class="legend-bar"></div>
        <span>+1</span>
        <span class="muted">Blue = negative, red = positive</span>
      </div>
    </section>
    """


def section_top_pairs(rows: list[dict[str, Any]]) -> str:
    headers = ["Feature A", "Feature B", "Correlation"]
    body_rows = [[row.get("feature_a"), row.get("feature_b"), fmt_float(row.get("corr"))] for row in rows]
    return render_section(
        section_id="correlation",
        title="Top Correlations",
        description=(
            "Strongest absolute Pearson correlations between numeric features. "
            "Large magnitudes imply redundancy; consider dropping or combining one "
            "feature from each highly correlated pair."
        ),
        headers=headers,
        rows=body_rows,
        filter_id="correlation-table",
    )


def section_ic(rows: list[dict[str, Any]]) -> str:
    headers = ["Feature", "IC (Spearman)"]
    body_rows = [[row.get("feature"), fmt_float(row.get("ic"))] for row in rows]
    return render_section(
        section_id="ic",
        title="Information Coefficient",
        description=(
            "Spearman rank correlation to the target. Values near 0 suggest weak monotonic "
            "signal; consistent positive/negative values indicate predictive direction."
        ),
        headers=headers,
        rows=body_rows,
        filter_id="ic-table",
    )


def section_mutual_info(rows: list[dict[str, Any]]) -> str:
    headers = ["Feature", "Mutual Information"]
    body_rows = [[row.get("feature"), fmt_float(row.get("mi"))] for row in rows]
    return render_section(
        section_id="mi",
        title="Mutual Information",
        description=(
            "Mutual information between each feature and the target. "
            "Higher values indicate stronger non-linear dependency; "
            "compare relative ranks rather than absolute scale."
        ),
        headers=headers,
        rows=body_rows,
        filter_id="mi-table",
    )


def section_tree_importance(rows: list[dict[str, Any]]) -> str:
    headers = ["Feature", "Importance"]
    body_rows = [[row.get("feature"), fmt_float(row.get("importance"))] for row in rows]
    return render_section(
        section_id="importance",
        title="Tree Importance",
        description=(
            "Random forest impurity-based importances. Useful for quick ranking, "
            "but bias toward high-cardinality or noisy features; validate with IC/MI."
        ),
        headers=headers,
        rows=body_rows,
        filter_id="importance-table",
    )


def section_vif(rows: list[dict[str, Any]]) -> str:
    headers = ["Feature", "VIF"]
    body_rows = [[row.get("feature"), fmt_float(row.get("vif"))] for row in rows]
    return render_section(
        section_id="vif",
        title="Variance Inflation Factor",
        description=(
            "Variance Inflation Factor estimates multicollinearity. "
            "Values > 10 often indicate unstable coefficients and redundant features."
        ),
        headers=headers,
        rows=body_rows,
        filter_id="vif-table",
    )


def section_drift(rows: list[dict[str, Any]]) -> str:
    headers = ["Feature", "Mean CV"]
    body_rows = [[row.get("feature"), fmt_float(row.get("mean_cv"))] for row in rows]
    return render_section(
        section_id="drift",
        title="Drift (Mean CV)",
        description=(
            "Mean coefficient of variation across time bins. "
            "Higher values imply unstable feature distributions over time."
        ),
        headers=headers,
        rows=body_rows,
        filter_id="drift-table",
    )


def section_leakage(rows: list[dict[str, Any]]) -> str:
    headers = ["Feature", "Corr to Target", "Equal Fraction"]
    body_rows = [
        [row.get("feature"), fmt_float(row.get("corr_to_target")), fmt_float(row.get("equal_frac"))]
        for row in rows
    ]
    return render_section(
        section_id="leakage",
        title="Leakage Checks",
        description=(
            "Leakage screening: high correlation to target or near-identical values "
            "suggest future information leaking into features."
        ),
        headers=headers,
        rows=body_rows,
        filter_id="leakage-table",
    )


def section_pca(data: dict[str, Any]) -> str:
    components = data.get("components", [])
    if not components:
        return """
        <section id="pca">
          <h2>PCA Variance</h2>
          <p class="muted">No PCA components available.</p>
        </section>
        """

    headers = ["Component", "Explained Var", "Cumulative Var"]
    body_rows = []
    for row in components:
        body_rows.append(
            [
                row.get("component"),
                fmt_float(row.get("explained_variance")),
                fmt_float(row.get("cumulative_variance")),
            ]
        )
    return render_section(
        section_id="pca",
        title="PCA Variance Analysis",
        description=(
            "Explained variance per component and cumulative variance. "
            "Use this to gauge intrinsic dimensionality and how quickly variance accumulates."
        ),
        headers=headers,
        rows=body_rows,
        filter_id="pca-table",
    )


def render_section(
    section_id: str,
    title: str,
    description: str,
    headers: list[str],
    rows: list[list[Any]],
    filter_id: str,
) -> str:
    table_rows = []
    for row in rows:
        cells = []
        for cell in row:
            if cell is None:
                cells.append("<td>-</td>")
                continue
            if isinstance(cell, SafeHtml):
                cells.append(f"<td>{cell}</td>")
                continue
            cell_text = html.escape(str(cell))
            cells.append(f"<td>{cell_text}</td>")
        table_rows.append("<tr>" + "".join(cells) + "</tr>")

    header_row = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    return f"""
    <section id="{html.escape(section_id)}">
      <h2>{html.escape(title)}</h2>
      <p class="muted">{html.escape(description)} (Generated {now})</p>
      <div class="table-controls">
        <span class="muted">Rows: {len(rows)}</span>
        <input type="text" placeholder="Filter" data-filter="{html.escape(filter_id)}" />
      </div>
      <div style="overflow:auto; max-height: 480px;">
        <table id="{html.escape(filter_id)}" class="sortable">
          <thead><tr>{header_row}</tr></thead>
          <tbody>
            {''.join(table_rows)}
          </tbody>
        </table>
      </div>
    </section>
    """
