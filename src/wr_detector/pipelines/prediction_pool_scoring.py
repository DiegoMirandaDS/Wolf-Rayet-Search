"""Resumable, lineage-checked scoring for exact-union prediction pools."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path
import re
from time import monotonic
from typing import Any, Iterable, Mapping

import duckdb
from joblib import load
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from wr_detector.config import load_yaml, resolve_path
from wr_detector.modeling.history import training_history_db_path
from wr_detector.pipelines.exact_variant_union import (
    VariantBit,
    VariantMaskSchema,
)
from wr_detector.pipelines.prediction_pool import (
    load_prediction_pool_config,
    make_sky_tiles,
)
from wr_detector.pipelines.prediction_pool_exact_union import (
    ensure_exact_union_builds_status,
    runtime_lineage,
)


SCORE_OUTPUT_COLUMNS = (
    "source_id",
    "source_hash_v1",
    "score",
    "predicted",
    "threshold",
    "tile_id",
    "scoring_run_id",
    "model_run_id",
    "result_id",
    "dataset_variant",
    "model_sha256",
    "pool_build_id",
    "bitmask_schema_sha256",
)


def load_prediction_scoring_config(
    config_path: str | Path,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    required = {
        "pool_config",
        "models_config",
        "output_db",
        "output_dir",
        "manifest_dir",
    }
    missing = sorted(required - set(config))
    if missing:
        raise KeyError(
            f"Prediction scoring config is missing keys: {missing}"
        )
    return config


def list_scoreable_models(
    config_path: str | Path,
    *,
    model_run_id: str,
) -> pd.DataFrame:
    """List model results that can be explicitly selected for pool scoring."""
    config = load_prediction_scoring_config(config_path)
    models_config = load_yaml(config["models_config"])
    history_path = training_history_db_path(models_config)
    if not history_path.exists():
        raise FileNotFoundError(history_path)
    with duckdb.connect(str(history_path), read_only=True) as con:
        columns = {
            row[0]
            for row in con.execute("DESCRIBE model_results").fetchall()
        }
        preferred = [
            "result_id",
            "dataset_variant",
            "feature_set",
            "model",
            "sampler",
            "holdout_average_precision",
            "holdout_wr_at_50",
            "holdout_wr_at_100",
            "holdout_recall_at_fpr_0p005",
            "selection_status",
            "model_path",
            "model_sha256",
        ]
        selected = [column for column in preferred if column in columns]
        frame = con.execute(
            f"""
            SELECT {", ".join(_quote(column) for column in selected)}
            FROM model_results
            WHERE run_id = ?
            ORDER BY holdout_wr_at_100 DESC NULLS LAST,
                     holdout_average_precision DESC NULLS LAST
            """,
            [model_run_id],
        ).fetchdf()
    if frame.empty:
        raise ValueError(
            f"No synchronized model results found for run {model_run_id!r}."
        )
    return frame


def score_prediction_pool(
    config_path: str | Path,
    *,
    scoring_run_id: str,
    model_run_id: str,
    result_ids: Iterable[str],
    max_tiles: int | None = None,
    dry_run: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """Score completed exact-union tiles with explicitly selected models."""
    _validate_path_component(scoring_run_id, label="scoring_run_id")
    _validate_path_component(model_run_id, label="model_run_id")
    result_ids = tuple(dict.fromkeys(str(value) for value in result_ids))
    if not result_ids:
        raise ValueError(
            "At least one explicit model result_id is required. "
            "Automatic best-model selection is intentionally unsupported."
        )
    config_path = resolve_path(config_path)
    config = load_prediction_scoring_config(config_path)
    pool_config_path = resolve_path(config["pool_config"])
    pool_config = load_prediction_pool_config(pool_config_path)
    pool_db = resolve_path(pool_config["output_db"])
    if not pool_db.exists():
        raise FileNotFoundError(pool_db)
    ensure_exact_union_builds_status(
        pool_db,
        expected_tile_ids={
            tile.tile_id for tile in make_sky_tiles(pool_config)
        },
    )
    pool = _load_pool_contract(pool_db, pool_config)
    models = _load_and_validate_models(
        config,
        model_run_id=model_run_id,
        result_ids=result_ids,
        pool=pool,
    )
    tiles = _load_completed_tiles(
        pool_db,
        pool_build_id=pool["pool_build_id"],
    )
    expected_tile_ids = {
        tile.tile_id for tile in make_sky_tiles(pool_config)
    }
    if max_tiles is None and bool(
        config.get("safety", {}).get("require_complete_pool", True)
    ):
        completed_ids = set(tiles["tile_id"].astype(str))
        missing = sorted(expected_tile_ids - completed_ids)
        extra = sorted(completed_ids - expected_tile_ids)
        if missing or extra:
            raise RuntimeError(
                "Full scoring requires a complete pool: "
                f"expected={len(expected_tile_ids)}, "
                f"completed={len(completed_ids)}, "
                f"missing={len(missing)}, extra={len(extra)}. "
                "Use --max-tiles only for a bounded validation."
            )
    if max_tiles is not None:
        tiles = tiles.head(int(max_tiles)).copy()
    if tiles.empty:
        raise ValueError("No completed exact-union tiles are available.")

    plan = {
        "scoring_run_id": scoring_run_id,
        "model_run_id": model_run_id,
        "pool_build_id": pool["pool_build_id"],
        "bitmask_schema_version": pool["schema"].version,
        "bitmask_schema_sha256": pool["schema"].sha256,
        "models": [
            {
                "result_id": model["result_id"],
                "variant": model["variant"],
                "bit": model["bit"],
                "features": model["features"],
                "model_sha256": model["model_sha256"],
            }
            for model in models
        ],
        "tiles": int(len(tiles)),
        "work_units": int(len(tiles) * len(models)),
        "complete_pool_expected_tiles": int(len(expected_tile_ids)),
        "dry_run": bool(dry_run),
    }
    if dry_run:
        return plan

    output_db = resolve_path(config["output_db"])
    output_dir = resolve_path(config["output_dir"])
    manifest_dir = resolve_path(config["manifest_dir"])
    for directory in [output_db.parent, output_dir, manifest_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    selection_sha256 = _canonical_sha256(
        {
            "model_run_id": model_run_id,
            "result_ids": sorted(result_ids),
        }
    )
    _initialize_scoring_database(
        output_db,
        scoring_run_id=scoring_run_id,
        model_run_id=model_run_id,
        selection_sha256=selection_sha256,
        pool=pool,
        config_path=config_path,
        models=models,
    )

    completed = 0
    skipped = 0
    input_hash_cache: dict[Path, str] = {}
    started = monotonic()
    total = len(tiles) * len(models)
    current = 0
    for model in models:
        _validate_path_component(model["result_id"], label="result_id")
        estimator = load(model["model_path"])
        if not hasattr(estimator, "predict_proba"):
            raise TypeError(
                f"Model {model['result_id']} does not implement predict_proba."
            )
        for tile in tiles.to_dict("records"):
            current += 1
            _validate_path_component(
                str(tile["tile_id"]), label="tile_id"
            )
            output_path = (
                output_dir
                / scoring_run_id
                / model["result_id"]
                / f"{tile['tile_id']}.parquet"
            )
            tile_manifest_path = (
                manifest_dir
                / scoring_run_id
                / model["result_id"]
                / f"{tile['tile_id']}.json"
            )
            if _completed_score_is_valid(
                output_db,
                scoring_run_id=scoring_run_id,
                result_id=model["result_id"],
                tile=tile,
                output_path=output_path,
                manifest_path=tile_manifest_path,
                model=model,
                pool=pool,
                input_hash_cache=input_hash_cache,
            ):
                skipped += 1
                if verbose:
                    print(
                        f"[{current}/{total}] skip {model['variant']} / "
                        f"{tile['tile_id']}",
                        flush=True,
                    )
                continue
            if _completed_score_record_exists(
                output_db,
                scoring_run_id=scoring_run_id,
                result_id=model["result_id"],
                tile_id=str(tile["tile_id"]),
            ):
                raise RuntimeError(
                    "A completed scoring record failed integrity validation: "
                    f"{model['result_id']} / {tile['tile_id']}. "
                    "Do not overwrite it; use a new scoring_run_id after "
                    "diagnosing the mismatch."
                )
            _mark_score_running(
                output_db,
                scoring_run_id=scoring_run_id,
                model=model,
                tile=tile,
                output_path=output_path,
                manifest_path=tile_manifest_path,
            )
            try:
                input_path = Path(str(tile["parquet_path"]))
                observed_input_sha256 = _cached_file_sha256(
                    input_path, input_hash_cache
                )
                if observed_input_sha256 != tile["acquisition_sha256"]:
                    raise ValueError(
                        f"Acquisition hash mismatch for {tile['tile_id']}."
                    )
                result = _score_one_tile(
                    estimator,
                    model=model,
                    tile=tile,
                    pool=pool,
                    output_path=output_path,
                    scoring_run_id=scoring_run_id,
                    model_run_id=model_run_id,
                    batch_rows=int(config.get("batch_rows", 250_000)),
                    compression=str(config.get("compression", "zstd")),
                )
                manifest = {
                    **result,
                    "scoring_run_id": scoring_run_id,
                    "model_run_id": model_run_id,
                    "result_id": model["result_id"],
                    "dataset_variant": model["variant"],
                    "variant_bit": model["bit"],
                    "pool_build_id": pool["pool_build_id"],
                    "acquisition_envelope_sha256": pool[
                        "envelope_sha256"
                    ],
                    "bitmask_schema_version": pool["schema"].version,
                    "bitmask_schema_sha256": pool["schema"].sha256,
                    "locus_run_id": model["locus_run_id"],
                    "locus_sha256": model["locus_sha256"],
                    "model_path": str(model["model_path"]),
                    "model_sha256": model["model_sha256"],
                    "selected_threshold": model["threshold"],
                    "features": model["features"],
                    "feature_columns_sha256": model[
                        "feature_columns_sha256"
                    ],
                    "input_acquisition_path": str(input_path),
                    "input_acquisition_sha256": observed_input_sha256,
                    "input_tile_manifest_sha256": tile[
                        "manifest_sha256"
                    ],
                    "runtime_lineage": runtime_lineage(),
                    "written_at": datetime.now(UTC).isoformat(),
                }
                _atomic_write_json(tile_manifest_path, manifest)
                manifest_sha256 = file_sha256(tile_manifest_path)
                _register_completed_score(
                    output_db,
                    scoring_run_id=scoring_run_id,
                    model=model,
                    tile=tile,
                    result=result,
                    manifest_path=tile_manifest_path,
                    manifest_sha256=manifest_sha256,
                )
                completed += 1
            except Exception as exc:
                _mark_score_failed(
                    output_db,
                    scoring_run_id=scoring_run_id,
                    result_id=model["result_id"],
                    tile_id=str(tile["tile_id"]),
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise
            if verbose:
                elapsed = max(monotonic() - started, 1e-9)
                rate = current / elapsed
                eta = (total - current) / rate if rate else 0
                print(
                    f"[{current}/{total}] scored {model['variant']} / "
                    f"{tile['tile_id']} rows={result['rows_scored']} "
                    f"missing={result['rows_missing_features']} "
                    f"eta_minutes={eta / 60:.1f}",
                    flush=True,
                )
    _refresh_scoring_views(output_db, output_dir)
    _update_scoring_run_status(
        output_db,
        scoring_run_id=scoring_run_id,
        expected_work_units=total,
    )
    return {
        **plan,
        "dry_run": False,
        "completed_this_run": completed,
        "skipped_valid": skipped,
        "output_db": str(output_db),
        "output_dir": str(output_dir / scoring_run_id),
    }


def audit_prediction_pool_scoring(
    config_path: str | Path,
    *,
    scoring_run_id: str,
) -> dict[str, Any]:
    """Audit physical score artifacts and their exact-union eligibility."""
    config = load_prediction_scoring_config(config_path)
    pool_config = load_prediction_pool_config(config["pool_config"])
    pool_db = resolve_path(pool_config["output_db"])
    output_db = resolve_path(config["output_db"])
    if not output_db.exists():
        raise FileNotFoundError(output_db)
    if pool_db.exists():
        ensure_exact_union_builds_status(
            pool_db,
            expected_tile_ids={
                tile.tile_id for tile in make_sky_tiles(pool_config)
            },
        )
    pool = _load_pool_contract(pool_db, pool_config)
    with duckdb.connect(str(output_db), read_only=True) as con:
        records = con.execute(
            """
            SELECT tiles.*, models.model_path, models.model_sha256
            FROM prediction_scoring_tiles tiles
            INNER JOIN prediction_scoring_models models
              USING (scoring_run_id, result_id)
            WHERE tiles.scoring_run_id=?
            ORDER BY tiles.result_id, tiles.tile_id
            """,
            [scoring_run_id],
        ).fetchdf()
    errors: list[str] = (
        ["no_registered_work_units"] if records.empty else []
    )
    audited: list[dict[str, Any]] = []
    model_hash_cache: dict[Path, str] = {}
    input_hash_cache: dict[Path, str] = {}
    for row in records.to_dict("records"):
        row_errors: list[str] = []
        output_path = Path(str(row["output_path"]))
        manifest_path = Path(str(row["manifest_path"]))
        input_path = Path(str(row["input_path"]))
        if row["status"] != "completed":
            row_errors.append(f"status={row['status']}")
        if not output_path.exists():
            row_errors.append("output_missing")
        elif file_sha256(output_path) != row["output_sha256"]:
            row_errors.append("output_sha256_mismatch")
        if not manifest_path.exists():
            row_errors.append("manifest_missing")
        elif file_sha256(manifest_path) != row["manifest_sha256"]:
            row_errors.append("manifest_sha256_mismatch")
        if not input_path.exists():
            row_errors.append("input_missing")
        elif (
            _cached_file_sha256(input_path, input_hash_cache)
            != row["input_sha256"]
        ):
            row_errors.append("input_sha256_mismatch")
        model_path = Path(str(row["model_path"]))
        if not model_path.exists():
            row_errors.append("model_missing")
        elif (
            _cached_file_sha256(model_path, model_hash_cache)
            != row["model_sha256"]
        ):
            row_errors.append("model_sha256_mismatch")
        physical_rows = None
        invalid_rows = None
        duplicate_rows = None
        if output_path.exists() and input_path.exists():
            bit_value = 1 << int(row["variant_bit"])
            with duckdb.connect() as con:
                physical_rows, duplicate_rows = con.execute(
                    """
                    SELECT COUNT(*), COUNT(*) - COUNT(DISTINCT source_id)
                    FROM read_parquet(?)
                    """,
                    [str(output_path)],
                ).fetchone()
                invalid_rows = con.execute(
                    """
                    SELECT COUNT(*)
                    FROM read_parquet(?) scored
                    LEFT JOIN read_parquet(?) source
                      ON scored.source_id = source.source_id
                    WHERE source.source_id IS NULL
                       OR scored.source_hash_v1 <>
                          source.source_hash_v1
                       OR source.known_source_exclusion IS NOT NULL
                       OR NOT source.passes_any_exact_variant
                       OR (
                            CAST(source.compatible_variant_mask AS UBIGINT)
                            & ?::UBIGINT
                          ) = 0
                    """,
                    [str(output_path), str(input_path), bit_value],
                ).fetchone()[0]
            if int(physical_rows) != int(row["rows_scored"]):
                row_errors.append("row_count_mismatch")
            if int(duplicate_rows):
                row_errors.append("duplicate_source_ids")
            if int(invalid_rows):
                row_errors.append("variant_ineligible_rows")
        errors.extend(
            f"{row['result_id']} / {row['tile_id']}: {error}"
            for error in row_errors
        )
        audited.append(
            {
                "result_id": row["result_id"],
                "tile_id": row["tile_id"],
                "rows_scored": physical_rows,
                "duplicates": duplicate_rows,
                "invalid_rows": invalid_rows,
                "errors": row_errors,
            }
        )
    return {
        "scoring_run_id": scoring_run_id,
        "pool_build_id": pool["pool_build_id"],
        "registered_work_units": int(len(records)),
        "completed_work_units": int(
            records["status"].eq("completed").sum()
        )
        if not records.empty
        else 0,
        "errors": errors,
        "ok": not errors and not records.empty,
        "tiles": audited,
    }


def _load_pool_contract(
    pool_db: Path,
    pool_config: Mapping[str, Any],
) -> dict[str, Any]:
    pool_build_id = str(pool_config["build"]["pool_build_id"])
    with duckdb.connect(str(pool_db), read_only=True) as con:
        build = con.execute(
            """
            SELECT status, envelope_sha256, bitmask_schema_version,
                   bitmask_schema_sha256
            FROM exact_union_builds
            WHERE pool_build_id=?
            """,
            [pool_build_id],
        ).fetchone()
        contract = con.execute(
            """
            SELECT bitmask_schema, loci
            FROM exact_union_contract
            """
        ).fetchone()
    if build is None or contract is None:
        raise ValueError(
            f"Incomplete exact-union contract for {pool_build_id!r}."
        )
    if str(build[0]) != "completed":
        raise ValueError(
            f"Exact-union pool {pool_build_id!r} is not completed: {build[0]}."
        )
    schema_payload = _as_json(contract[0])
    loci = _as_json(contract[1])
    schema = VariantMaskSchema(
        family=str(schema_payload["family"]),
        major_version=int(schema_payload["major_version"]),
        entries=tuple(
            VariantBit(
                variant=str(entry["variant"]),
                bit=int(entry["bit"]),
            )
            for entry in sorted(
                schema_payload["entries"],
                key=lambda value: int(value["bit"]),
            )
        ),
    )
    if schema.version != str(build[2]) or schema.sha256 != str(build[3]):
        raise ValueError(
            "The persisted bitmask schema does not match exact_union_builds."
        )
    return {
        "pool_build_id": pool_build_id,
        "envelope_sha256": str(build[1]),
        "schema": schema,
        "loci": loci,
    }


def _load_completed_tiles(
    pool_db: Path,
    *,
    pool_build_id: str,
) -> pd.DataFrame:
    with duckdb.connect(str(pool_db), read_only=True) as con:
        return con.execute(
            """
            SELECT tile_id, parquet_path, acquisition_sha256,
                   manifest_path, manifest_sha256, written
            FROM exact_union_tiles
            WHERE pool_build_id=? AND status='completed'
            ORDER BY tile_id
            """,
            [pool_build_id],
        ).fetchdf()


def _load_and_validate_models(
    config: Mapping[str, Any],
    *,
    model_run_id: str,
    result_ids: tuple[str, ...],
    pool: Mapping[str, Any],
) -> list[dict[str, Any]]:
    models_config = load_yaml(config["models_config"])
    history_path = training_history_db_path(models_config)
    if not history_path.exists():
        raise FileNotFoundError(history_path)
    with duckdb.connect(str(history_path), read_only=True) as con:
        placeholders = ", ".join("?" for _ in result_ids)
        rows = con.execute(
            f"""
            SELECT * FROM model_results
            WHERE run_id=? AND result_id IN ({placeholders})
            """,
            [model_run_id, *result_ids],
        ).fetchdf()
    found = set(rows["result_id"].astype(str)) if not rows.empty else set()
    missing = sorted(set(result_ids) - found)
    if missing:
        raise ValueError(
            f"Model result_ids are absent from run {model_run_id!r}: {missing}"
        )
    models: list[dict[str, Any]] = []
    for row in rows.to_dict("records"):
        variant = str(row["dataset_variant"])
        bit = pool["schema"].bit_for(variant)
        locus = pool["loci"].get(variant)
        if not locus:
            raise ValueError(
                f"Pool contract has no locus for model variant {variant!r}."
            )
        locus_run_id = str(row.get("locus_run_id") or "")
        locus_sha256 = str(row.get("reference_dataset_sha256") or "")
        if locus_run_id != str(locus["locus_run_id"]):
            raise ValueError(
                f"Locus run mismatch for {row['result_id']}: "
                f"model={locus_run_id}, pool={locus['locus_run_id']}."
            )
        if locus_sha256 != str(locus["source_sha256"]):
            raise ValueError(
                f"Locus artifact mismatch for {row['result_id']}."
            )
        model_path = resolve_path(str(row["model_path"]))
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        observed_model_sha256 = file_sha256(model_path)
        expected_model_sha256 = str(row["model_sha256"])
        if observed_model_sha256 != expected_model_sha256:
            raise ValueError(
                f"Model hash mismatch for {row['result_id']}."
            )
        features = _as_json(row["feature_columns_json"])
        if not isinstance(features, list) or not features:
            raise ValueError(
                f"Invalid feature list for {row['result_id']}."
            )
        threshold = float(row["selected_threshold"])
        if not np.isfinite(threshold):
            raise ValueError(
                f"Invalid threshold for {row['result_id']}."
            )
        models.append(
            {
                "result_id": str(row["result_id"]),
                "variant": variant,
                "bit": bit,
                "features": [str(value) for value in features],
                "feature_columns_sha256": _canonical_sha256(features),
                "model_path": model_path,
                "model_sha256": expected_model_sha256,
                "threshold": threshold,
                "locus_run_id": locus_run_id,
                "locus_sha256": locus_sha256,
            }
        )
    return sorted(
        models,
        key=lambda model: result_ids.index(model["result_id"]),
    )


def _initialize_scoring_database(
    db_path: Path,
    *,
    scoring_run_id: str,
    model_run_id: str,
    selection_sha256: str,
    pool: Mapping[str, Any],
    config_path: Path,
    models: list[Mapping[str, Any]],
) -> None:
    with duckdb.connect(str(db_path)) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_scoring_runs (
                scoring_run_id VARCHAR PRIMARY KEY,
                model_run_id VARCHAR,
                model_selection_sha256 VARCHAR,
                pool_build_id VARCHAR,
                envelope_sha256 VARCHAR,
                bitmask_schema_version VARCHAR,
                bitmask_schema_sha256 VARCHAR,
                config_path VARCHAR,
                config_sha256 VARCHAR,
                status VARCHAR,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS prediction_scoring_models (
                scoring_run_id VARCHAR,
                result_id VARCHAR,
                dataset_variant VARCHAR,
                variant_bit INTEGER,
                locus_run_id VARCHAR,
                locus_sha256 VARCHAR,
                model_path VARCHAR,
                model_sha256 VARCHAR,
                selected_threshold DOUBLE,
                feature_columns_json JSON,
                feature_columns_sha256 VARCHAR,
                PRIMARY KEY (scoring_run_id, result_id)
            );
            CREATE TABLE IF NOT EXISTS prediction_scoring_tiles (
                scoring_run_id VARCHAR,
                result_id VARCHAR,
                tile_id VARCHAR,
                dataset_variant VARCHAR,
                variant_bit INTEGER,
                status VARCHAR,
                input_path VARCHAR,
                input_sha256 VARCHAR,
                output_path VARCHAR,
                output_sha256 VARCHAR,
                output_bytes BIGINT,
                manifest_path VARCHAR,
                manifest_sha256 VARCHAR,
                rows_variant_compatible BIGINT,
                rows_missing_features BIGINT,
                rows_scored BIGINT,
                rows_predicted_positive BIGINT,
                error_message VARCHAR,
                updated_at TIMESTAMP,
                PRIMARY KEY (scoring_run_id, result_id, tile_id)
            );
            """
        )
        existing = con.execute(
            """
            SELECT model_run_id, model_selection_sha256, pool_build_id,
                   envelope_sha256, bitmask_schema_sha256
            FROM prediction_scoring_runs
            WHERE scoring_run_id=?
            """,
            [scoring_run_id],
        ).fetchone()
        expected = (
            model_run_id,
            selection_sha256,
            pool["pool_build_id"],
            pool["envelope_sha256"],
            pool["schema"].sha256,
        )
        if existing is not None and tuple(existing) != expected:
            raise ValueError(
                "scoring_run_id already exists with a different model or "
                "pool contract. Use a new scoring_run_id."
            )
        now = datetime.now(UTC)
        con.execute(
            """
            INSERT INTO prediction_scoring_runs VALUES
            (?, ?, ?, ?, ?, ?, ?, ?, ?, 'running', ?, ?)
            ON CONFLICT (scoring_run_id) DO UPDATE SET
                status='running', updated_at=excluded.updated_at
            """,
            [
                scoring_run_id,
                model_run_id,
                selection_sha256,
                pool["pool_build_id"],
                pool["envelope_sha256"],
                pool["schema"].version,
                pool["schema"].sha256,
                str(config_path),
                file_sha256(config_path),
                now,
                now,
            ],
        )
        for model in models:
            con.execute(
                """
                INSERT INTO prediction_scoring_models VALUES
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?::JSON, ?)
                ON CONFLICT (scoring_run_id, result_id) DO NOTHING
                """,
                [
                    scoring_run_id,
                    model["result_id"],
                    model["variant"],
                    model["bit"],
                    model["locus_run_id"],
                    model["locus_sha256"],
                    str(model["model_path"]),
                    model["model_sha256"],
                    model["threshold"],
                    json.dumps(model["features"]),
                    model["feature_columns_sha256"],
                ],
            )


def _score_one_tile(
    estimator: Any,
    *,
    model: Mapping[str, Any],
    tile: Mapping[str, Any],
    pool: Mapping[str, Any],
    output_path: Path,
    scoring_run_id: str,
    model_run_id: str,
    batch_rows: int,
    compression: str,
) -> dict[str, Any]:
    input_path = Path(str(tile["parquet_path"]))
    features = list(model["features"])
    bit_value = 1 << int(model["bit"])
    selected = [
        _quote("source_id"),
        _quote("source_hash_v1"),
        *[_quote(feature) for feature in features],
    ]
    predicate = """
        passes_any_exact_variant
        AND known_source_exclusion IS NULL
        AND (
            CAST(compatible_variant_mask AS UBIGINT) & ?::UBIGINT
        ) <> 0
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".parquet.tmp")
    temporary_path.unlink(missing_ok=True)
    writer: pq.ParquetWriter | None = None
    rows_compatible = 0
    rows_missing = 0
    rows_scored = 0
    rows_positive = 0
    with duckdb.connect() as con:
        compatible, unique_sources = con.execute(
            f"""
            SELECT COUNT(*), COUNT(DISTINCT source_id)
            FROM read_parquet(?)
            WHERE {predicate}
            """,
            [str(input_path), bit_value],
        ).fetchone()
        if int(compatible) != int(unique_sources):
            raise ValueError(
                f"Duplicate compatible source_id values in {tile['tile_id']}."
            )
        reader = con.execute(
            f"""
            SELECT {", ".join(selected)}
            FROM read_parquet(?)
            WHERE {predicate}
            ORDER BY source_id
            """,
            [str(input_path), bit_value],
        ).to_arrow_reader(batch_size=max(1, int(batch_rows)))
        for batch in reader:
            frame = batch.to_pandas()
            rows_compatible += len(frame)
            complete = frame[features].notna().all(axis=1)
            rows_missing += int((~complete).sum())
            scoreable = frame.loc[complete].copy()
            if scoreable.empty:
                continue
            x = scoreable[features].astype("float64")
            score = np.asarray(
                estimator.predict_proba(x)[:, 1],
                dtype="float64",
            )
            if not np.isfinite(score).all():
                raise ValueError(
                    f"Non-finite scores in {tile['tile_id']}."
                )
            output = pd.DataFrame(
                {
                    "source_id": scoreable["source_id"].astype("int64"),
                    "source_hash_v1": scoreable[
                        "source_hash_v1"
                    ].astype("string"),
                    "score": score.astype("float32"),
                    "predicted": score >= float(model["threshold"]),
                    "threshold": np.float32(model["threshold"]),
                    "tile_id": str(tile["tile_id"]),
                    "scoring_run_id": scoring_run_id,
                    "model_run_id": model_run_id,
                    "result_id": model["result_id"],
                    "dataset_variant": model["variant"],
                    "model_sha256": model["model_sha256"],
                    "pool_build_id": pool["pool_build_id"],
                    "bitmask_schema_sha256": pool["schema"].sha256,
                }
            )
            table = pa.Table.from_pandas(
                output[list(SCORE_OUTPUT_COLUMNS)],
                preserve_index=False,
            )
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary_path,
                    table.schema,
                    compression=compression,
                    use_dictionary=True,
                )
            writer.write_table(table)
            rows_scored += len(output)
            rows_positive += int(output["predicted"].sum())
    if writer is not None:
        writer.close()
    else:
        empty = pd.DataFrame(
            {
                "source_id": pd.Series(dtype="int64"),
                "source_hash_v1": pd.Series(dtype="string"),
                "score": pd.Series(dtype="float32"),
                "predicted": pd.Series(dtype="bool"),
                "threshold": pd.Series(dtype="float32"),
                "tile_id": pd.Series(dtype="string"),
                "scoring_run_id": pd.Series(dtype="string"),
                "model_run_id": pd.Series(dtype="string"),
                "result_id": pd.Series(dtype="string"),
                "dataset_variant": pd.Series(dtype="string"),
                "model_sha256": pd.Series(dtype="string"),
                "pool_build_id": pd.Series(dtype="string"),
                "bitmask_schema_sha256": pd.Series(dtype="string"),
            }
        )
        empty.to_parquet(
            temporary_path,
            index=False,
            compression=compression,
        )
    if rows_compatible != int(compatible):
        temporary_path.unlink(missing_ok=True)
        raise AssertionError("Scoring reader did not consume every compatible row.")
    if rows_scored + rows_missing != rows_compatible:
        temporary_path.unlink(missing_ok=True)
        raise AssertionError("Compatible, missing and scored counts do not reconcile.")
    temporary_path.replace(output_path)
    return {
        "tile_id": str(tile["tile_id"]),
        "rows_variant_compatible": rows_compatible,
        "rows_missing_features": rows_missing,
        "rows_scored": rows_scored,
        "rows_predicted_positive": rows_positive,
        "output_path": str(output_path),
        "output_sha256": file_sha256(output_path),
        "output_bytes": output_path.stat().st_size,
    }


def _mark_score_running(
    db_path: Path,
    *,
    scoring_run_id: str,
    model: Mapping[str, Any],
    tile: Mapping[str, Any],
    output_path: Path,
    manifest_path: Path,
) -> None:
    with duckdb.connect(str(db_path)) as con:
        con.execute(
            """
            INSERT INTO prediction_scoring_tiles (
                scoring_run_id, result_id, tile_id, dataset_variant,
                variant_bit, status, input_path, input_sha256,
                output_path, manifest_path, updated_at
            ) VALUES (?, ?, ?, ?, ?, 'running', ?, ?, ?, ?, ?)
            ON CONFLICT (scoring_run_id, result_id, tile_id) DO UPDATE SET
                status='running', error_message=NULL,
                updated_at=excluded.updated_at
            """,
            [
                scoring_run_id,
                model["result_id"],
                tile["tile_id"],
                model["variant"],
                model["bit"],
                tile["parquet_path"],
                tile["acquisition_sha256"],
                str(output_path),
                str(manifest_path),
                datetime.now(UTC),
            ],
        )


def _register_completed_score(
    db_path: Path,
    *,
    scoring_run_id: str,
    model: Mapping[str, Any],
    tile: Mapping[str, Any],
    result: Mapping[str, Any],
    manifest_path: Path,
    manifest_sha256: str,
) -> None:
    with duckdb.connect(str(db_path)) as con:
        con.execute(
            """
            UPDATE prediction_scoring_tiles SET
                status='completed',
                output_sha256=?,
                output_bytes=?,
                manifest_path=?,
                manifest_sha256=?,
                rows_variant_compatible=?,
                rows_missing_features=?,
                rows_scored=?,
                rows_predicted_positive=?,
                error_message=NULL,
                updated_at=?
            WHERE scoring_run_id=? AND result_id=? AND tile_id=?
            """,
            [
                result["output_sha256"],
                result["output_bytes"],
                str(manifest_path),
                manifest_sha256,
                result["rows_variant_compatible"],
                result["rows_missing_features"],
                result["rows_scored"],
                result["rows_predicted_positive"],
                datetime.now(UTC),
                scoring_run_id,
                model["result_id"],
                tile["tile_id"],
            ],
        )


def _mark_score_failed(
    db_path: Path,
    *,
    scoring_run_id: str,
    result_id: str,
    tile_id: str,
    error: str,
) -> None:
    with duckdb.connect(str(db_path)) as con:
        con.execute(
            """
            UPDATE prediction_scoring_tiles
            SET status='failed', error_message=?, updated_at=?
            WHERE scoring_run_id=? AND result_id=? AND tile_id=?
            """,
            [
                error,
                datetime.now(UTC),
                scoring_run_id,
                result_id,
                tile_id,
            ],
        )


def _completed_score_record_exists(
    db_path: Path,
    *,
    scoring_run_id: str,
    result_id: str,
    tile_id: str,
) -> bool:
    if not db_path.exists():
        return False
    with duckdb.connect(str(db_path), read_only=True) as con:
        return bool(
            con.execute(
                """
                SELECT COUNT(*) FROM prediction_scoring_tiles
                WHERE scoring_run_id=? AND result_id=? AND tile_id=?
                  AND status='completed'
                """,
                [scoring_run_id, result_id, tile_id],
            ).fetchone()[0]
        )


def _completed_score_is_valid(
    db_path: Path,
    *,
    scoring_run_id: str,
    result_id: str,
    tile: Mapping[str, Any],
    output_path: Path,
    manifest_path: Path,
    model: Mapping[str, Any],
    pool: Mapping[str, Any],
    input_hash_cache: dict[Path, str],
) -> bool:
    if not db_path.exists():
        return False
    with duckdb.connect(str(db_path), read_only=True) as con:
        row = con.execute(
            """
            SELECT status, input_sha256, output_sha256, manifest_sha256,
                   rows_scored
            FROM prediction_scoring_tiles
            WHERE scoring_run_id=? AND result_id=? AND tile_id=?
            """,
            [scoring_run_id, result_id, tile["tile_id"]],
        ).fetchone()
    if row is None or row[0] != "completed":
        return False
    input_path = Path(str(tile["parquet_path"]))
    if not input_path.exists() or not output_path.exists() or not manifest_path.exists():
        return False
    observed_input = _cached_file_sha256(
        input_path, input_hash_cache
    )
    if (
        observed_input != row[1]
        or observed_input != tile["acquisition_sha256"]
        or file_sha256(output_path) != row[2]
        or file_sha256(manifest_path) != row[3]
    ):
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "scoring_run_id": scoring_run_id,
        "result_id": result_id,
        "dataset_variant": model["variant"],
        "variant_bit": model["bit"],
        "pool_build_id": pool["pool_build_id"],
        "bitmask_schema_sha256": pool["schema"].sha256,
        "model_sha256": model["model_sha256"],
        "feature_columns_sha256": model["feature_columns_sha256"],
        "input_acquisition_sha256": tile["acquisition_sha256"],
        "output_sha256": row[2],
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        return False
    with duckdb.connect() as con:
        physical = con.execute(
            "SELECT COUNT(*) FROM read_parquet(?)",
            [str(output_path)],
        ).fetchone()[0]
    return int(physical) == int(row[4])


def _refresh_scoring_views(db_path: Path, output_dir: Path) -> None:
    files = list(output_dir.glob("*/*/*.parquet"))
    if not files:
        return
    glob_path = (output_dir / "*" / "*" / "*.parquet").as_posix()
    escaped = glob_path.replace("'", "''")
    with duckdb.connect(str(db_path)) as con:
        con.execute(
            f"""
            CREATE OR REPLACE VIEW prediction_pool_scores AS
            SELECT * FROM read_parquet(
                '{escaped}', union_by_name=true, hive_partitioning=false
            );
            CREATE OR REPLACE VIEW prediction_pool_scoring_effective_tiles AS
            SELECT * FROM prediction_scoring_tiles
            WHERE status='completed';
            """
        )


def _update_scoring_run_status(
    db_path: Path,
    *,
    scoring_run_id: str,
    expected_work_units: int,
) -> None:
    with duckdb.connect(str(db_path)) as con:
        completed = con.execute(
            """
            SELECT COUNT(*) FROM prediction_scoring_tiles
            WHERE scoring_run_id=? AND status='completed'
            """,
            [scoring_run_id],
        ).fetchone()[0]
        status = (
            "completed"
            if int(completed) == int(expected_work_units)
            else "partial"
        )
        con.execute(
            """
            UPDATE prediction_scoring_runs
            SET status=?, updated_at=?
            WHERE scoring_run_id=?
            """,
            [status, datetime.now(UTC), scoring_run_id],
        )


def _as_json(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def _canonical_sha256(value: Any) -> str:
    return sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cached_file_sha256(
    path: Path,
    cache: dict[Path, str],
) -> str:
    if path not in cache:
        cache[path] = file_sha256(path)
    return cache[path]


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _quote(identifier: str) -> str:
    return '"' + str(identifier).replace('"', '""') + '"'


def _validate_path_component(value: str, *, label: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+\-]*", str(value)):
        raise ValueError(
            f"{label} contains unsafe path characters: {value!r}"
        )
