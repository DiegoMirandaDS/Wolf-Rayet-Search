"""Synchronization and cleanup of training artifacts in the canonical DuckDB history."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd

from wr_detector.config import load_yaml, resolve_path
from wr_detector.modeling.data import ensure_parent_dir


ARTIFACT_COLUMNS = {
    "model": "model_path",
    "metadata_json": "metadata_json_path",
    "confusion_matrix": "holdout_confusion_matrix_path",
    "roc_curve": "holdout_roc_curve_path",
    "precision_recall_curve": "holdout_pr_curve_path",
    "predictions": "predictions_path",
    "feature_importance": "feature_importance_path",
    "feature_importance_figure": "feature_importance_figure_path",
}

PATH_COLUMNS = [
    "model_path",
    "metadata_json_path",
    "holdout_confusion_matrix_path",
    "holdout_roc_curve_path",
    "holdout_pr_curve_path",
    "predictions_path",
    "feature_importance_path",
    "feature_importance_figure_path",
]


def sync_training_history(
    config_path: str | Path = "configs/models.yaml",
    *,
    run_id: str | None = None,
    run_name: str | None = None,
    notes: str | None = None,
    source_csv: str | Path | None = None,
    replace_run: bool = False,
) -> dict[str, object]:
    config = load_yaml(config_path)
    csv_path = resolve_path(source_csv or config["outputs"]["training_results"])
    results = pd.read_csv(csv_path)
    if results.empty:
        raise ValueError(f"No training results found in {csv_path}")

    return sync_training_history_frame(
        config_path,
        results,
        run_id=run_id,
        run_name=run_name,
        notes=notes,
        source_csv=csv_path,
        replace_run=replace_run,
        csv_sha256=_file_sha256(csv_path),
    )


def sync_training_history_frame(
    config_path: str | Path,
    results: pd.DataFrame,
    *,
    run_id: str | None = None,
    run_name: str | None = None,
    notes: str | None = None,
    source_csv: str | Path | None = None,
    replace_run: bool = False,
    csv_sha256: str | None = None,
) -> dict[str, object]:
    config = load_yaml(config_path)
    db_path = training_history_db_path(config)
    if results.empty:
        raise ValueError("No training results provided.")
    imported_at = datetime.now(UTC).isoformat()
    run_id = run_id or make_run_id_from_frame(results)
    run_name = run_name or "train_models"
    results = _prepare_results(results, run_id=run_id, imported_at=imported_at)
    artifacts = _collect_artifacts(results, run_id=run_id)
    feature_importance = _collect_feature_importance(results, run_id=run_id)
    predictions = _collect_predictions(results, run_id=run_id)
    metadata = _collect_model_metadata(results, run_id=run_id)

    db_path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(db_path)) as con:
        _ensure_history_schema(con)
        if _run_exists(con, run_id) and not replace_run:
            raise ValueError(f"Training run already exists: {run_id}. Use --replace-run to overwrite it.")
        if replace_run:
            _delete_run(con, run_id)
        con.execute(
            """
            INSERT OR REPLACE INTO training_runs
            (run_id, run_name, imported_at, source_csv, config_path, notes, row_count, csv_sha256)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                run_id,
                run_name,
                imported_at,
                str(source_csv) if source_csv is not None else None,
                str(Path(config_path)),
                notes,
                int(len(results)),
                csv_sha256 or _dataframe_sha256(results),
            ],
        )
        _append_dataframe(con, "model_results", results)
        _append_dataframe(con, "model_artifacts", artifacts)
        if not feature_importance.empty:
            _append_dataframe(con, "feature_importance", feature_importance)
        if not predictions.empty:
            _append_dataframe(con, "model_predictions", predictions)
        if not metadata.empty:
            _append_dataframe(con, "model_metadata", metadata)

    return {
        "db_path": str(db_path),
        "run_id": run_id,
        "rows": int(len(results)),
        "artifacts": int(len(artifacts)),
        "feature_importance_rows": int(len(feature_importance)),
        "prediction_rows": int(len(predictions)),
        "metadata_rows": int(len(metadata)),
    }


def sync_second_layer_history(
    models_config: dict,
    results: pd.DataFrame,
    *,
    run_id: str,
    source_csv: str | Path | None = None,
    config_path: str | Path | None = None,
    replace_run: bool = False,
) -> dict[str, object]:
    """Sync second-layer validation results into the canonical history DB.

    Results land in ``second_layer_results`` with a run registry in
    ``second_layer_runs``, mirroring the first-layer tables so the
    DuckDB history stays the single experiment store across layers.
    """
    if results.empty:
        raise ValueError("No second-layer results provided.")
    db_path = training_history_db_path(models_config)
    imported_at = datetime.now(UTC).isoformat()
    prepared = results.copy()
    if "run_id" not in prepared.columns:
        prepared.insert(0, "run_id", run_id)
    prepared["imported_at"] = imported_at
    prepared = _normalize_dataframe_paths(prepared)

    db_path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(db_path)) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS second_layer_runs (
                run_id VARCHAR PRIMARY KEY,
                imported_at VARCHAR,
                source_csv VARCHAR,
                config_path VARCHAR,
                row_count INTEGER,
                csv_sha256 VARCHAR
            )
            """
        )
        exists = bool(
            con.execute("SELECT COUNT(*) FROM second_layer_runs WHERE run_id = ?", [run_id]).fetchone()[0]
        )
        if exists and not replace_run:
            raise ValueError(f"Second-layer run already exists: {run_id}. Use replace_run to overwrite it.")
        if exists:
            if _table_exists(con, "second_layer_results"):
                con.execute("DELETE FROM second_layer_results WHERE run_id = ?", [run_id])
            con.execute("DELETE FROM second_layer_runs WHERE run_id = ?", [run_id])
        _ensure_second_layer_result_types(con)
        con.execute(
            """
            INSERT INTO second_layer_runs
            (run_id, imported_at, source_csv, config_path, row_count, csv_sha256)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                run_id,
                imported_at,
                make_project_relative(source_csv) if source_csv is not None else None,
                str(config_path) if config_path is not None else None,
                int(len(prepared)),
                _dataframe_sha256(prepared),
            ],
        )
        _append_dataframe(con, "second_layer_results", prepared)

    return {"db_path": str(db_path), "run_id": run_id, "rows": int(len(prepared))}


def _ensure_second_layer_result_types(con: duckdb.DuckDBPyConnection) -> None:
    """Repair early history schemas that inferred fractional lineage as INTEGER."""
    if not _table_exists(con, "second_layer_results"):
        return
    schema = con.execute("DESCRIBE second_layer_results").fetchdf()
    types = dict(zip(schema["column_name"], schema["column_type"], strict=False))
    holdout_type = str(types.get("holdout_fraction", "")).upper()
    if holdout_type and not any(
        token in holdout_type for token in ["DOUBLE", "FLOAT", "DECIMAL", "REAL"]
    ):
        con.execute(
            "ALTER TABLE second_layer_results "
            "ALTER COLUMN holdout_fraction SET DATA TYPE DOUBLE"
        )


def list_training_runs(config_path: str | Path = "configs/models.yaml") -> pd.DataFrame:
    config = load_yaml(config_path)
    db_path = training_history_db_path(config)
    if not db_path.exists():
        return pd.DataFrame(
            columns=[
                "run_id",
                "run_name",
                "imported_at",
                "notes",
                "row_count",
                "best_holdout_wr_at_100",
                "best_holdout_average_precision",
                "best_holdout_recall_at_fpr_0p005",
            ]
        )
    with duckdb.connect(str(db_path), read_only=True) as con:
        if not _table_exists(con, "training_runs"):
            return pd.DataFrame()
        if not _table_exists(con, "model_results"):
            return con.execute("SELECT * FROM training_runs ORDER BY imported_at DESC").fetchdf()
        model_columns = set(con.execute("DESCRIBE model_results").fetchdf()["column_name"])
        wr_at_100 = "max(m.holdout_wr_at_100)" if "holdout_wr_at_100" in model_columns else "NULL"
        average_precision = "max(m.holdout_average_precision)" if "holdout_average_precision" in model_columns else "NULL"
        recall_fpr = "max(m.holdout_recall_at_fpr_0p005)" if "holdout_recall_at_fpr_0p005" in model_columns else "NULL"
        return con.execute(
            f"""
            SELECT
                r.run_id,
                r.run_name,
                r.imported_at,
                r.notes,
                r.row_count,
                {wr_at_100} AS best_holdout_wr_at_100,
                {average_precision} AS best_holdout_average_precision,
                {recall_fpr} AS best_holdout_recall_at_fpr_0p005
            FROM training_runs r
            LEFT JOIN model_results m USING (run_id)
            GROUP BY r.run_id, r.run_name, r.imported_at, r.notes, r.row_count
            ORDER BY r.imported_at DESC
            """
        ).fetchdf()


def load_training_run_results(config_path: str | Path = "configs/models.yaml", *, run_id: str) -> pd.DataFrame:
    config = load_yaml(config_path)
    db_path = training_history_db_path(config)
    if not db_path.exists():
        raise FileNotFoundError(f"Training history DB not found: {db_path}")
    with duckdb.connect(str(db_path), read_only=True) as con:
        if not _table_exists(con, "model_results"):
            raise ValueError(f"No model_results table found in {db_path}")
        return con.execute("SELECT * FROM model_results WHERE run_id = ? ORDER BY holdout_wr_at_100 DESC NULLS LAST", [run_id]).fetchdf()


def cleanup_unreferenced_model_artifacts(
    config_path: str | Path = "configs/models.yaml",
    *,
    apply: bool = False,
    remove_db_backed_sidecars: bool = False,
) -> pd.DataFrame:
    config = load_yaml(config_path)
    results_path = resolve_path(config["outputs"]["training_results"])
    results = _prepare_results(pd.read_csv(results_path), run_id="current", imported_at="")
    referenced = {
        _normalize_path(path)
        for path in _iter_referenced_artifact_paths(results)
        if path is not None
    }
    candidates = list(_iter_artifact_candidates(config))
    rows = []
    deleted_paths = []
    for path in candidates:
        normalized = _normalize_path(path)
        keep = normalized in referenced
        reason = "referenced_current_results" if keep else "unreferenced"
        if keep and remove_db_backed_sidecars and _is_db_backed_sidecar(path):
            keep = False
            reason = "db_backed_sidecar"
        rows.append(
            {
                "path": str(path),
                "keep": keep,
                "reason": reason,
                "size_bytes": int(path.stat().st_size) if path.exists() else 0,
                "last_modified": datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat() if path.exists() else None,
            }
        )
        if apply and not keep and path.exists():
            path.unlink()
            deleted_paths.append(make_project_relative(path))
    if apply and deleted_paths:
        _mark_artifacts_deleted(training_history_db_path(config), deleted_paths)
    return pd.DataFrame(rows)


def normalize_training_history_paths(config_path: str | Path = "configs/models.yaml") -> dict[str, object]:
    config = load_yaml(config_path)
    db_path = training_history_db_path(config)
    if not db_path.exists():
        raise FileNotFoundError(f"Training history DB not found: {db_path}")
    updated: dict[str, int] = {}
    with duckdb.connect(str(db_path)) as con:
        for table in ["model_results", "model_metadata", "training_runs"]:
            if not _table_exists(con, table):
                continue
            df = con.execute(f"SELECT * FROM {table}").fetchdf()
            if df.empty:
                updated[table] = 0
                continue
            df = _normalize_dataframe_paths(df)
            if table == "model_metadata" and "metadata_json" in df.columns:
                df["metadata_json"] = df["metadata_json"].map(_normalize_metadata_json)
            con.execute(f"DROP TABLE {table}")
            _append_dataframe(con, table, df)
            updated[table] = int(len(df))
        if _table_exists(con, "model_artifacts"):
            artifacts = con.execute("SELECT * FROM model_artifacts").fetchdf()
            if not artifacts.empty:
                artifacts["path"] = artifacts["path"].map(make_project_relative)
                artifacts["exists"] = artifacts["path"].map(lambda value: _resolve_project_path(value).exists())
                artifacts["size_bytes"] = artifacts["path"].map(lambda value: _resolve_project_path(value).stat().st_size if _resolve_project_path(value).exists() else None)
                artifacts = artifacts[artifacts["exists"]].reset_index(drop=True)
                con.execute("DROP TABLE model_artifacts")
                _append_dataframe(con, "model_artifacts", artifacts)
            updated["model_artifacts"] = int(len(artifacts))
    return {"db_path": str(db_path), "updated": updated}


def training_history_db_path(config: dict) -> Path:
    return resolve_path(config["outputs"].get("training_history_db", "data/databases/training_history.duckdb"))


def make_run_id(source_csv: str | Path) -> str:
    path = Path(source_csv)
    digest = _file_sha256(path)[:12] if path.exists() else "missing"
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"run_{stamp}_{digest}"


def make_training_run_id(config_path: str | Path = "configs/models.yaml") -> str:
    path = Path(config_path)
    digest = _file_sha256(path)[:12] if path.exists() else "missing"
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"run_{stamp}_{digest}"


def make_run_id_from_frame(results: pd.DataFrame) -> str:
    digest = _dataframe_sha256(results)[:12]
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"run_{stamp}_{digest}"


def _prepare_results(results: pd.DataFrame, *, run_id: str, imported_at: str) -> pd.DataFrame:
    prepared = results.copy()
    for column in ["run_id", "result_id", "imported_at"]:
        if column in prepared.columns:
            prepared = prepared.drop(columns=[column])
    prepared.insert(0, "run_id", run_id)
    prepared.insert(1, "result_id", [_result_id(row) for _, row in prepared.iterrows()])
    prepared.insert(2, "imported_at", imported_at)
    if "model_path" in prepared.columns:
        prepared["metadata_json_path"] = prepared["model_path"].map(_metadata_path_from_model_path)
    return _normalize_dataframe_paths(prepared)


def _result_id(row: pd.Series) -> str:
    parts = [
        str(row.get("dataset_variant", "")),
        str(row.get("feature_set", "")),
        str(row.get("model", "")),
        str(row.get("sampler", "")),
        str(row.get("train_positive_cohort", "all")),
        str(row.get("evaluation_positive_cohort", "all")),
        str(row.get("model_path", "")),
    ]
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()


def _metadata_path_from_model_path(value: object) -> str | None:
    if pd.isna(value) or not str(value).strip():
        return None
    return str(Path(str(value)).with_suffix(".json"))


def _collect_artifacts(results: pd.DataFrame, *, run_id: str) -> pd.DataFrame:
    rows = []
    for _, row in results.iterrows():
        result_id = str(row["result_id"])
        for artifact_type, column in ARTIFACT_COLUMNS.items():
            if column not in row.index:
                continue
            path = _path_or_none(row[column])
            if path is None:
                continue
            stat = path.stat() if path.exists() else None
            rows.append(
                {
                    "run_id": run_id,
                    "result_id": result_id,
                    "artifact_type": artifact_type,
                    "path": make_project_relative(path),
                    "exists": bool(path.exists()),
                    "size_bytes": int(stat.st_size) if stat else None,
                    "last_modified": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat() if stat else None,
                }
            )
    return pd.DataFrame(rows)


def _collect_feature_importance(results: pd.DataFrame, *, run_id: str) -> pd.DataFrame:
    rows = []
    for _, row in results.iterrows():
        path = _path_or_none(row.get("feature_importance_path"))
        if path is None or not path.exists():
            continue
        table = pd.read_csv(path)
        table.insert(0, "run_id", run_id)
        table.insert(1, "result_id", row["result_id"])
        table.insert(2, "dataset_variant", row.get("dataset_variant"))
        table.insert(3, "feature_set", row.get("feature_set"))
        table.insert(4, "model", row.get("model"))
        table.insert(5, "sampler", row.get("sampler"))
        rows.append(table)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def _collect_predictions(results: pd.DataFrame, *, run_id: str) -> pd.DataFrame:
    rows = []
    for _, row in results.iterrows():
        path = _path_or_none(row.get("predictions_path"))
        if path is None or not path.exists():
            continue
        table = pd.read_csv(path)
        table.insert(0, "run_id", run_id)
        table.insert(1, "result_id", row["result_id"])
        table.insert(2, "dataset_variant", row.get("dataset_variant"))
        table.insert(3, "feature_set", row.get("feature_set"))
        table.insert(4, "model", row.get("model"))
        table.insert(5, "sampler", row.get("sampler"))
        rows.append(table)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def _collect_model_metadata(results: pd.DataFrame, *, run_id: str) -> pd.DataFrame:
    rows = []
    for _, row in results.iterrows():
        path = _path_or_none(row.get("metadata_json_path"))
        if path is None or not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        metadata = _normalize_metadata_paths(metadata)
        rows.append(
            {
                "run_id": run_id,
                "result_id": row["result_id"],
                "dataset_variant": row.get("dataset_variant"),
                "feature_set": row.get("feature_set"),
                "model": row.get("model"),
                "sampler": row.get("sampler"),
                "metadata_json": json.dumps(metadata, sort_keys=True),
                "metadata_path": make_project_relative(path),
            }
        )
    return pd.DataFrame(rows)


def _ensure_history_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS training_runs (
            run_id VARCHAR PRIMARY KEY,
            run_name VARCHAR,
            imported_at VARCHAR,
            source_csv VARCHAR,
            config_path VARCHAR,
            notes VARCHAR,
            row_count INTEGER,
            csv_sha256 VARCHAR
        )
        """
    )


def _delete_run(con: duckdb.DuckDBPyConnection, run_id: str) -> None:
    for table in ["model_results", "model_artifacts", "feature_importance", "model_predictions", "model_metadata"]:
        if _table_exists(con, table):
            con.execute(f"DELETE FROM {table} WHERE run_id = ?", [run_id])
    con.execute("DELETE FROM training_runs WHERE run_id = ?", [run_id])


def _run_exists(con: duckdb.DuckDBPyConnection, run_id: str) -> bool:
    return bool(con.execute("SELECT COUNT(*) FROM training_runs WHERE run_id = ?", [run_id]).fetchone()[0])


def _mark_artifacts_deleted(db_path: Path, paths: list[str]) -> None:
    if not db_path.exists():
        return
    with duckdb.connect(str(db_path)) as con:
        if not _table_exists(con, "model_artifacts"):
            return
        con.execute("DELETE FROM model_artifacts WHERE path IN (SELECT * FROM UNNEST(?))", [paths])


def _append_dataframe(con: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame) -> None:
    if df.empty:
        return
    prepared = _normalize_history_string_columns(df)
    con.register("_incoming_df", prepared)
    if not _table_exists(con, table):
        con.execute(f"CREATE TABLE {table} AS SELECT * FROM _incoming_df")
    else:
        _add_missing_columns(con, table, "_incoming_df")
        _repair_empty_column_types(con, table, "_incoming_df")
        con.execute(f"INSERT INTO {table} BY NAME SELECT * FROM _incoming_df")
    con.unregister("_incoming_df")


def _normalize_history_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep optional text lineage fields from becoming INTEGER when all-null."""
    prepared = df.copy()
    for column in prepared.columns:
        non_null = prepared[column].dropna()
        is_text = (
            column.endswith(("_sha256", "_json", "_path"))
            or column in {
                "code_git_commit",
                "locus_run_id",
                "train_positive_cohort",
                "evaluation_positive_cohort",
            }
            or (
                not non_null.empty
                and non_null.map(lambda value: isinstance(value, str)).all()
            )
        )
        if is_text:
            prepared[column] = prepared[column].astype("string")
    return prepared


def _repair_empty_column_types(
    con: duckdb.DuckDBPyConnection,
    table: str,
    registered_df: str,
) -> None:
    """Repair legacy columns whose type was inferred from only NULL values."""
    existing = {
        row.column_name: row.column_type
        for row in con.execute(f"DESCRIBE {table}").fetchdf().itertuples(
            index=False
        )
    }
    incoming = {
        row.column_name: row.column_type
        for row in con.execute(
            f"DESCRIBE {registered_df}"
        ).fetchdf().itertuples(index=False)
    }
    for column, incoming_type in incoming.items():
        existing_type = existing.get(column)
        if existing_type is None or existing_type == incoming_type:
            continue
        quoted = _quote_identifier(column)
        populated = con.execute(
            f"SELECT COUNT({quoted}) FROM {table}"
        ).fetchone()[0]
        if int(populated) == 0:
            con.execute(
                f"ALTER TABLE {table} ALTER COLUMN {quoted} "
                f"SET DATA TYPE {incoming_type}"
            )


def _add_missing_columns(con: duckdb.DuckDBPyConnection, table: str, registered_df: str) -> None:
    existing = set(con.execute(f"DESCRIBE {table}").fetchdf()["column_name"])
    incoming = con.execute(f"DESCRIBE {registered_df}").fetchdf()
    for row in incoming.itertuples(index=False):
        if row.column_name not in existing:
            con.execute(f"ALTER TABLE {table} ADD COLUMN {_quote_identifier(row.column_name)} {row.column_type}")


def _table_exists(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    return bool(
        con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
            [table],
        ).fetchone()[0]
    )


def _iter_referenced_artifact_paths(results: pd.DataFrame) -> Iterable[Path | None]:
    for _, row in results.iterrows():
        for column in ARTIFACT_COLUMNS.values():
            if column in row.index:
                yield _path_or_none(row[column])


def _iter_artifact_candidates(config: dict) -> Iterable[Path]:
    reports_dir = resolve_path(config["outputs"]["reports_dir"])
    models_dir = resolve_path(config["outputs"]["models_dir"])
    figures_dir = resolve_path(config["outputs"]["figures_dir"])
    for directory, patterns in [
        (models_dir, ["*.joblib", "*.json"]),
        (figures_dir, ["*.png"]),
        (reports_dir, ["*_predictions.csv", "*_feature_importance.csv"]),
    ]:
        if not directory.exists():
            continue
        for pattern in patterns:
            yield from directory.glob(pattern)
    runs_dir = reports_dir / "runs"
    if runs_dir.exists():
        for pattern in [
            "*/models/*.joblib",
            "*/models/*.json",
            "*/figures/*.png",
            "*/reports/*_predictions.csv",
            "*/reports/*_feature_importance.csv",
        ]:
            yield from runs_dir.glob(pattern)


def _is_db_backed_sidecar(path: Path) -> bool:
    if path.name.endswith("_predictions.csv") or path.name.endswith("_feature_importance.csv"):
        return True
    return path.suffix.lower() == ".json" and path.parent.name == "models"


def _path_or_none(value: object) -> Path | None:
    if value is None or pd.isna(value) or not str(value).strip():
        return None
    return _resolve_project_path(str(value))


def _normalize_path(path: Path) -> str:
    return str(path.resolve()).casefold()


def make_project_relative(value: object) -> str | None:
    if value is None or pd.isna(value) or not str(value).strip():
        return None
    path = Path(str(value))
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(path)


def _resolve_project_path(value: object) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return Path.cwd().resolve() / path


def _normalize_dataframe_paths(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    for column in PATH_COLUMNS + ["source_csv", "config_path", "metadata_path", "path"]:
        if column in normalized.columns:
            normalized[column] = normalized[column].map(make_project_relative)
    return normalized


def _normalize_metadata_json(value: object) -> object:
    if value is None or pd.isna(value):
        return value
    try:
        metadata = json.loads(str(value))
    except json.JSONDecodeError:
        return value
    return json.dumps(_normalize_metadata_paths(metadata), sort_keys=True)


def _normalize_metadata_paths(value):
    if isinstance(value, dict):
        return {key: _normalize_metadata_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_metadata_paths(item) for item in value]
    if isinstance(value, str):
        path = Path(value)
        if path.is_absolute():
            return make_project_relative(path)
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataframe_sha256(df: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    digest.update("|".join(map(str, df.columns)).encode("utf-8"))
    return digest.hexdigest()


def _quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'
