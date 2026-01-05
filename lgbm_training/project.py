"""Save/load LightGBM training projects."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .utils import dumps_json


def save_project(
    project_dir: str | Path,
    config: dict[str, Any],
    results: dict[str, Any],
    report_html: str,
) -> None:
    project_path = Path(project_dir)
    project_path.mkdir(parents=True, exist_ok=True)

    meta = {"config": config}
    (project_path / "meta.json").write_text(dumps_json(meta), encoding="utf-8")
    (project_path / "results.json").write_text(dumps_json(results), encoding="utf-8")
    (project_path / "report.html").write_text(report_html, encoding="utf-8")


def load_project(project_dir: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    project_path = Path(project_dir)
    meta_path = project_path / "meta.json"
    results_path = project_path / "results.json"
    if not meta_path.exists() or not results_path.exists():
        raise FileNotFoundError("Project metadata or results not found.")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    results = json.loads(results_path.read_text(encoding="utf-8"))
    return meta, results
