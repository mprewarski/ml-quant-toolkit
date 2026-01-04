"""Command-line interface for the feature analysis toolkit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .analysis import analyze_dataset
from .project import load_project, save_project
from .report import render_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Feature analysis toolkit for tabular datasets.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Run EDA analysis and save a project.")
    analyze.add_argument("--data", required=True, help="Path to CSV file.")
    analyze.add_argument("--target", default=None, help="Target column name.")
    analyze.add_argument("--time-column", default=None, help="Time column name for drift analysis.")
    analyze.add_argument("--project-dir", default=None, help="Output project directory.")
    analyze.add_argument("--output-html", default=None, help="Optional report HTML path.")
    analyze.add_argument("--sample-rows", type=int, default=None, help="Optional row cap for faster runs.")
    analyze.add_argument(
        "--max-vif-features",
        type=int,
        default=None,
        help="Maximum numeric features for VIF computation.",
    )
    analyze.add_argument(
        "--analyses",
        default=None,
        help="Comma-separated list of analyses to run (e.g. correlation,heatmap,pca).",
    )
    analyze.add_argument(
        "--exclude-columns",
        default=None,
        help="Comma-separated list of columns to exclude from analysis.",
    )

    report = subparsers.add_parser("report", help="Regenerate report from a saved project.")
    report.add_argument("--project-dir", required=True, help="Project directory.")
    report.add_argument("--output-html", default=None, help="Optional report HTML path.")

    return parser


def load_data(path: str, sample_rows: int | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if sample_rows is not None and sample_rows > 0:
        df = df.head(sample_rows)
    return df


def analyze_command(args: argparse.Namespace) -> int:
    data_path = Path(args.data)
    df = load_data(str(data_path), args.sample_rows)

    project_dir = args.project_dir
    if project_dir is None:
        project_dir = f"projects/{data_path.stem}_eda"

    defaults = load_main_config()
    project_config = load_project_config(project_dir)

    target = resolve_config_value(args.target, project_config.get("target"), defaults.get("target", "TARGET"))
    time_column = resolve_config_value(
        args.time_column, project_config.get("time_column"), defaults.get("time_column", "Date")
    )
    max_vif_features = resolve_config_value(
        args.max_vif_features,
        project_config.get("max_vif_features"),
        defaults.get("max_vif_features", 50),
    )
    analyses = parse_analyses(args.analyses)
    if analyses is None:
        analyses = project_config.get("analyses") or defaults.get("analyses")
    exclude_columns = parse_csv_list(args.exclude_columns)
    if exclude_columns is None:
        exclude_columns = project_config.get("exclude_columns") or defaults.get("exclude_columns", [])

    results = analyze_dataset(
        df,
        target_column=target,
        time_column=time_column,
        max_vif_features=max_vif_features,
        analyses=analyses,
        exclude_columns=exclude_columns,
    )

    title = f"EDA Report - {data_path.stem}"
    report_html = render_report(results, title=title, dataset_name=data_path.name)

    config: dict[str, Any] = {
        "data": str(data_path),
        "target": target,
        "time_column": time_column,
        "sample_rows": args.sample_rows,
        "max_vif_features": max_vif_features,
        "analyses": analyses,
        "exclude_columns": exclude_columns,
    }

    save_project(project_dir, config, results, report_html)

    if args.output_html:
        Path(args.output_html).write_text(report_html, encoding="utf-8")

    print(f"Saved project to {project_dir}")
    return 0


def report_command(args: argparse.Namespace) -> int:
    meta, results = load_project(args.project_dir)
    config = meta.get("config", {})
    dataset = Path(config.get("data", "dataset")).name
    title = f"EDA Report - {Path(dataset).stem}"

    report_html = render_report(results, title=title, dataset_name=dataset)

    output_path = args.output_html or str(Path(args.project_dir) / "report.html")
    Path(output_path).write_text(report_html, encoding="utf-8")
    print(f"Wrote report to {output_path}")
    return 0


def load_project_config(project_dir: str | None) -> dict[str, Any]:
    if not project_dir:
        return {}
    meta_path = Path(project_dir) / "meta.json"
    if not meta_path.exists():
        return {}
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return meta.get("config", {})


def parse_analyses(value: str | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return value
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_csv_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return value
    return [item.strip() for item in value.split(",") if item.strip()]


def load_main_config() -> dict[str, Any]:
    config_path = Path("config.json")
    if not config_path.exists():
        return {}
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    defaults = data.get("defaults", {})
    return defaults if isinstance(defaults, dict) else {}


def resolve_config_value(cli_value: Any, project_value: Any, default_value: Any) -> Any:
    if cli_value is not None:
        return cli_value
    if project_value is not None:
        return project_value
    return default_value


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "analyze":
        return analyze_command(args)
    if args.command == "report":
        return report_command(args)
    raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
