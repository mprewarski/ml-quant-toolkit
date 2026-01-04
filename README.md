# Quant Toolkit Feature Analysis Scripts

This toolkit provides Python scripts for exploratory data analysis on tabular datasets
used in tree-based ML workflows. It focuses on redundancy, predictive power, stability,
and leakage checks, with a unified HTML report and project save/load support.

## Features

- Feature profiling: missing %, unique counts, numeric summary stats
- Correlation analysis: top pairs, correlation partners, interactive heatmap
- Predictive power: IC (Spearman), mutual information, tree importances
- Stability: time-binned drift using Mean CV
- Leakage checks: correlation to target and identical-value detection
- PCA variance analysis: explained and cumulative variance per component
- Reporting: interactive HTML with filters, sorting, and navigation
- Save/load projects: cached results with `meta.json`, `results.json`, `report.html`

## Quick Start

```bash
python -m feature_analysis.cli analyze \
  --data sp500-y-15-69.csv \
  --project-dir projects/sp500_eda
```

Regenerate the HTML report without re-running analysis:

```bash
python -m feature_analysis.cli report --project-dir projects/sp500_eda
```

## Outputs

- `projects/<name>_eda/meta.json` - analysis configuration.
- `projects/<name>_eda/results.json` - computed metrics.
- `projects/<name>_eda/report.html` - interactive report.

## Configuration

Global defaults live in `config.json`. Per-project overrides are stored in
`projects/<name>_eda/meta.json` under the `config` key. CLI flags override both.

Precedence order:
1) CLI args
2) Project config (`meta.json`)
3) Global defaults (`config.json`)

Example `config.json`:

```json
{
  "defaults": {
    "target": "TARGET",
    "time_column": "Date",
    "max_vif_features": 50,
    "analyses": [
      "profile",
      "correlation",
      "heatmap",
      "ic",
      "mutual_info",
      "tree_importance",
      "vif",
      "drift",
      "leakage",
      "pca"
    ],
    "exclude_columns": ["P123 ID", "Ticker"]
  }
}
```

To disable columns for a specific project, set `exclude_columns` in that project's
`meta.json` config.

Example `meta.json` project override:

```json
{
  "config": {
    "data": "sp500-y-15-69.csv",
    "target": "TARGET",
    "time_column": "Date",
    "analyses": ["profile", "correlation", "heatmap", "pca"],
    "exclude_columns": ["P123 ID", "Ticker"]
  }
}
```

## CLI Options

Analyze:

```bash
python -m feature_analysis.cli analyze \
  --data <csv_path> \
  --project-dir <output_dir> \
  --target <column_name> \
  --time-column <column_name> \
  --analyses profile,correlation,heatmap,pca \
  --exclude-columns col_a,col_b \
  --sample-rows 50000 \
  --max-vif-features 50 \
  --output-html report.html
```

Report:

```bash
python -m feature_analysis.cli report --project-dir <output_dir> --output-html report.html
```

## Analyses List

- `profile`: dataset column profile and summary stats
- `correlation`: correlation partners and top pairs
- `heatmap`: interactive correlation heatmap (top 40 by abs corr)
- `ic`: information coefficient (Spearman) to target
- `mutual_info`: mutual information to target
- `tree_importance`: random forest importances
- `vif`: variance inflation factor
- `drift`: time-binned mean coefficient of variation
- `leakage`: correlation/equality checks to target
- `pca`: PCA variance analysis
