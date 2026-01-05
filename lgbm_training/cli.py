"""Command-line interface for the LightGBM training toolkit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .project import load_project, save_project
from .report import render_report
from .training import train_lightgbm


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LightGBM training toolkit for tabular datasets.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train LightGBM model and save a project.")
    train.add_argument("--data", required=True, help="Path to CSV file.")
    train.add_argument("--target", default=None, help="Target column name.")
    train.add_argument("--time-column", default=None, help="Time column name for time splits.")
    train.add_argument("--task", default=None, help="Task type: classification or regression.")
    train.add_argument("--project-dir", default=None, help="Output project directory.")
    train.add_argument("--output-html", default=None, help="Optional report HTML path.")
    train.add_argument("--sample-rows", type=int, default=None, help="Optional row cap for faster runs.")
    train.add_argument("--exclude-columns", default=None, help="Comma-separated list of columns to exclude.")
    train.add_argument("--gap", type=int, default=None, help="Gap length in weeks/rows.")
    train.add_argument("--holdout-length", type=int, default=None, help="Holdout length in weeks/rows.")
    train.add_argument("--training-length", type=int, default=None, help="Training length in weeks/rows.")
    train.add_argument("--n-folds", type=int, default=None, help="Number of rolling folds.")
    train.add_argument("--threshold", type=float, default=None, help="Classification threshold.")
    train.add_argument("--params", default=None, help="JSON string or path with LightGBM params.")

    report = subparsers.add_parser("report", help="Regenerate report from a saved project.")
    report.add_argument("--project-dir", required=True, help="Project directory.")
    report.add_argument("--output-html", default=None, help="Optional report HTML path.")

    return parser


def load_data(path: str, sample_rows: int | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if sample_rows is not None and sample_rows > 0:
        df = df.head(sample_rows)
    return df


def parse_csv_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_params(value: str | None) -> dict[str, Any] | None:
    if not value:
        return None
    path = Path(value)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(value)


def load_main_config() -> dict[str, Any]:
    config_path = Path("config.json")
    if not config_path.exists():
        return {}
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    defaults = data.get("lgbm_defaults", {})
    return defaults if isinstance(defaults, dict) else {}


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


def resolve_config_value(cli_value: Any, project_value: Any, default_value: Any) -> Any:
    if cli_value is not None:
        return cli_value
    if project_value is not None:
        return project_value
    return default_value


def train_command(args: argparse.Namespace) -> int:
    data_path = Path(args.data)
    df = load_data(str(data_path), args.sample_rows)

    project_dir = args.project_dir
    if project_dir is None:
        project_dir = f"projects/{data_path.stem}_lgbm"

    defaults = load_main_config()
    project_config = load_project_config(project_dir)

    target = resolve_config_value(args.target, project_config.get("target"), defaults.get("target", "TARGET"))
    time_column = resolve_config_value(
        args.time_column, project_config.get("time_column"), defaults.get("time_column", "Date")
    )
    task = resolve_config_value(args.task, project_config.get("task"), defaults.get("task"))
    gap = resolve_config_value(args.gap, project_config.get("gap"), defaults.get("gap", 0))
    holdout_length = resolve_config_value(
        args.holdout_length,
        project_config.get("holdout_length"),
        defaults.get("holdout_length", 26),
    )
    training_length = resolve_config_value(
        args.training_length,
        project_config.get("training_length"),
        defaults.get("training_length", 156),
    )
    n_folds = resolve_config_value(args.n_folds, project_config.get("n_folds"), defaults.get("n_folds", 1))
    threshold = resolve_config_value(
        args.threshold, project_config.get("threshold"), defaults.get("threshold", 0.5)
    )

    exclude_columns = parse_csv_list(args.exclude_columns)
    if exclude_columns is None:
        exclude_columns = project_config.get("exclude_columns") or defaults.get("exclude_columns", [])

    params = None
    if args.params:
        params = parse_params(args.params)
    elif project_config.get("params"):
        params = project_config.get("params")
    elif defaults.get("params"):
        params = defaults.get("params")

    results = train_lightgbm(
        df,
        target_column=target,
        time_column=time_column,
        task=task,
        params=params,
        gap=int(gap),
        holdout_length=int(holdout_length),
        training_length=int(training_length),
        n_folds=int(n_folds),
        threshold=float(threshold),
        exclude_columns=exclude_columns,
    )

    title = f"LightGBM Training Report - {data_path.stem}"
    report_html = render_report(results, title=title, dataset_name=data_path.name)

    config: dict[str, Any] = {
        "data": str(data_path),
        "target": target,
        "time_column": time_column,
        "task": task,
        "gap": gap,
        "holdout_length": holdout_length,
        "training_length": training_length,
        "n_folds": n_folds,
        "threshold": threshold,
        "sample_rows": args.sample_rows,
        "exclude_columns": exclude_columns,
        "params": params,
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
    title = f"LightGBM Training Report - {Path(dataset).stem}"

    report_html = render_report(results, title=title, dataset_name=dataset)

    output_path = args.output_html or str(Path(args.project_dir) / "report.html")
    Path(output_path).write_text(report_html, encoding="utf-8")
    print(f"Wrote report to {output_path}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "train":
        return train_command(args)
    if args.command == "report":
        return report_command(args)
    raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
