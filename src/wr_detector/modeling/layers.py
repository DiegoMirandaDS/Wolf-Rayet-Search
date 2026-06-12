"""Discovery and loading of validation-layer runs for the Model Explorer.

A validation layer is any post-first-stage re-ranking/compatibility stage
(the current second layer, a future third layer, ...). Each layer is
described by a :class:`ValidationLayer` spec. Runs are read from the
canonical training-history DuckDB first (``<layer>_runs`` /
``<layer>_results`` tables) and fall back to per-run CSV discovery for
runs that were never synchronized.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

from wr_detector.config import load_yaml, resolve_path
from wr_detector.modeling.history import training_history_db_path


@dataclass(frozen=True)
class ValidationLayer:
    key: str
    title: str
    config_path: str
    results_filename: str
    train_command: str

    @property
    def runs_table(self) -> str:
        return f"{self.key}_runs"

    @property
    def results_table(self) -> str:
        return f"{self.key}_results"


VALIDATION_LAYERS: list[ValidationLayer] = [
    ValidationLayer(
        key="second_layer",
        title="Second layer · one-class validators",
        config_path="configs/second_layer.yaml",
        results_filename="second_layer_validation_results.csv",
        train_command="wr-detector train-second-layer --config configs/second_layer.yaml",
    ),
]


def layer_runs_root(layer: ValidationLayer) -> Path:
    """Root directory that contains one subdirectory per layer run."""
    config = load_yaml(layer.config_path)
    template = str(
        (config.get("outputs") or {}).get(
            "run_dir_template", f"reports/modeling/{layer.key}/runs/{{run_id}}"
        )
    )
    return resolve_path(template.split("{run_id}")[0])


def layer_history_db_path(layer: ValidationLayer) -> Path:
    """Canonical history DB for the layer (shared with first-layer history)."""
    config = load_yaml(layer.config_path)
    models_config = load_yaml(config.get("models_config", "configs/models.yaml"))
    return training_history_db_path(models_config)


def list_layer_runs(layer: ValidationLayer) -> pd.DataFrame:
    """All known runs for a layer (history DB first, CSV-only runs appended)."""
    db_runs = _list_db_runs(layer)
    csv_runs = _list_csv_runs(layer)
    known = set(db_runs["run_id"])
    extra = csv_runs[~csv_runs["run_id"].isin(known)]
    runs = pd.concat([db_runs, extra], ignore_index=True)
    return runs.sort_values("modified_at", ascending=False).reset_index(drop=True)


def load_layer_results(layer: ValidationLayer, run_id: str) -> pd.DataFrame:
    """Result rows for one layer run; empty frame when the run is unknown."""
    db_path = layer_history_db_path(layer)
    if db_path.exists():
        with duckdb.connect(str(db_path), read_only=True) as con:
            if _table_exists(con, layer.results_table):
                results = con.execute(
                    f"SELECT * FROM {layer.results_table} WHERE run_id = ?", [run_id]
                ).fetchdf()
                if not results.empty:
                    return results
    csv_runs = _list_csv_runs(layer)
    match = csv_runs[csv_runs["run_id"].eq(run_id)]
    if match.empty:
        return pd.DataFrame()
    return pd.read_csv(match.iloc[0]["results_path"])


def _list_db_runs(layer: ValidationLayer) -> pd.DataFrame:
    columns = ["run_id", "modified_at", "source", "results_path"]
    db_path = layer_history_db_path(layer)
    if not db_path.exists():
        return pd.DataFrame(columns=columns)
    with duckdb.connect(str(db_path), read_only=True) as con:
        if not _table_exists(con, layer.runs_table):
            return pd.DataFrame(columns=columns)
        runs = con.execute(
            f"SELECT run_id, imported_at, source_csv FROM {layer.runs_table}"
        ).fetchdf()
    runs["modified_at"] = pd.to_datetime(runs["imported_at"], utc=True, errors="coerce")
    runs["source"] = "history_db"
    runs = runs.rename(columns={"source_csv": "results_path"})
    return runs[columns]


def _list_csv_runs(layer: ValidationLayer) -> pd.DataFrame:
    root = layer_runs_root(layer)
    rows = []
    if root.exists():
        for run_dir in sorted(root.iterdir()):
            results_path = run_dir / layer.results_filename
            if not run_dir.is_dir() or not results_path.exists():
                continue
            modified = datetime.fromtimestamp(results_path.stat().st_mtime, tz=timezone.utc)
            rows.append(
                {
                    "run_id": run_dir.name,
                    "modified_at": modified,
                    "source": "csv",
                    "results_path": str(results_path),
                }
            )
    return pd.DataFrame(rows, columns=["run_id", "modified_at", "source", "results_path"])


def _table_exists(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    return bool(
        con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
            [table],
        ).fetchone()[0]
    )
