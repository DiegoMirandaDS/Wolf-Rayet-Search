"""Acquisition-first, exact-union prediction-pool construction.

The physical layer keeps every source returned by the broad intra-mission
color envelope. Exact locus, photometric-quality and astrometric rules are
stored as versioned masks and exposed through logical DuckDB views.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import platform
from pathlib import Path
import subprocess
from time import monotonic
from typing import Any, Iterable, Mapping

import duckdb
import pandas as pd

from wr_detector.config import resolve_path
from wr_detector.pipelines.exact_variant_union import (
    ExactLocus,
    VariantMaskSchema,
    build_tile_count_record,
    build_variant_mask_schema,
    evaluate_exact_variant_union,
    load_exact_loci_from_exports,
)
from wr_detector.pipelines.prediction_pool import (
    BASE_COLUMNS,
    SkyTile,
    audit_color_envelope_reference_coverage,
    build_prediction_pool_adql,
    derive_color_envelope,
    download_prediction_tile,
    load_prediction_pool_config,
    local_storage_bytes,
    make_sky_tiles,
)


TMASS_CROSSMATCH_LINEAGE_FIELDS = (
    "tmass_candidate_count",
    "tmass_alternatives_json",
    "tmass_selection_rule",
)
SOURCE_HASH_V1_FIELDS = tuple(BASE_COLUMNS)
SOURCE_HASH_V1_ALGORITHM = "pandas_hash_pandas_object_uint64_v1"
TMASS_SELECTION_RULE = (
    "tmass_candidate_v1:max_AB_bands,max_A_bands,min_error_sum,tmass_id"
)


def build_exact_union_prediction_pool(
    config_path: str | Path,
    *,
    dry_run: bool = False,
    max_tiles: int | None = None,
    row_limit: int | None = None,
    confirm_full_build: bool = False,
) -> dict[str, Any]:
    """Build or resume the immutable acquisition layer tile by tile."""
    config_path = Path(config_path)
    config = load_prediction_pool_config(config_path)
    build = config.get("build", {})
    pool_build_id = str(build["pool_build_id"])
    if str(build.get("eligibility_policy")) != "exact_variant_union_annotations":
        raise ValueError(
            "Exact-union builds require eligibility_policy="
            "'exact_variant_union_annotations'."
        )
    if not bool(build.get("allow_mutation", False)) and not dry_run:
        raise RuntimeError(f"Pool build {pool_build_id!r} does not allow mutation.")
    if (
        not dry_run
        and max_tiles is None
        and (
            not confirm_full_build
            or not bool(build.get("full_build_enabled", False))
        )
    ):
        raise RuntimeError(
            "The full-sky exact-union build is gated. Complete the smoke build, "
            "set build.full_build_enabled=true, and pass --confirm-full-build. "
            "Use --max-tiles for a bounded build."
        )

    variants = list(config["exact_union"]["variants"])
    schema = build_variant_mask_schema(
        variants,
        major_version=int(config["exact_union"].get("mask_major_version", 1)),
    )
    loci = load_exact_loci_from_exports(
        reference_dir=resolve_path(config["paths"]["processed_reference_dir"]),
        reference_output_template=config["filters"]["color_locus"][
            "reference_output_template"
        ],
        variants=variants,
        aggregate_min_outlier_planes=int(
            config["filters"]["color_locus"].get(
                "aggregate_min_outlier_planes", 1
            )
        ),
    )
    envelope = derive_color_envelope(config)
    coverage = audit_color_envelope_reference_coverage(config, envelope)
    expected_tiles = make_sky_tiles(config)
    tiles = expected_tiles
    if max_tiles is not None:
        tiles = tiles[: int(max_tiles)]

    sample_adql = (
        build_prediction_pool_adql(
            tiles[0], envelope, config, row_limit=row_limit
        )
        if tiles
        else ""
    )
    if dry_run:
        return {
            "dry_run": True,
            "pool_build_id": pool_build_id,
            "tiles": len(tiles),
            "envelope_sha256": envelope["sha256"],
            "envelope_reference_population": envelope["reference_population"],
            "envelope_coverage": coverage,
            "mask_schema": schema.as_manifest(),
            "sample_adql": sample_adql,
        }

    db_path = resolve_path(config["output_db"])
    staging_dir = resolve_path(config["staging_dir"])
    acquisition_dir = resolve_path(config["output_dir"])
    manifest_dir = resolve_path(config["manifest_dir"])
    for directory in [db_path.parent, staging_dir, acquisition_dir, manifest_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    known_sources = load_known_source_labels(config)
    initialize_exact_union_database(
        db_path,
        pool_build_id=pool_build_id,
        config_path=config_path,
        envelope=envelope,
        coverage=coverage,
        schema=schema,
        loci=loci,
    )
    completed = 0
    skipped = 0
    failed = 0
    pending: list[tuple[SkyTile, str]] = []
    for tile in tiles:
        adql = build_prediction_pool_adql(
            tile, envelope, config, row_limit=row_limit
        )
        if completed_tile_is_valid(
            db_path,
            acquisition_dir=acquisition_dir,
            pool_build_id=pool_build_id,
            tile_id=tile.tile_id,
            adql_sha256=sha256(adql.encode("utf-8")).hexdigest(),
            row_limit=row_limit,
            envelope_sha256=envelope["sha256"],
            bitmask_schema_version=schema.version,
            bitmask_schema_sha256=schema.sha256,
        ):
            skipped += 1
            continue
        pending.append((tile, adql))

    max_workers = max(1, int(config.get("safety", {}).get("max_workers", 1)))
    max_bytes = int(config.get("safety", {}).get("max_local_bytes", 0))
    progress_every = max(
        1, int(config.get("monitoring", {}).get("progress_every_tiles", 5))
    )
    started_at = monotonic()
    handled = 0
    print(
        f"[exact-union] build={pool_build_id} pending={len(pending)} "
        f"skipped={skipped} download_workers={max_workers}",
        flush=True,
    )
    next_item = 0
    futures: dict[Any, tuple[SkyTile, str]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while next_item < len(pending) or futures:
            while next_item < len(pending) and len(futures) < max_workers:
                used_bytes = local_storage_bytes(
                    db_path, staging_dir, acquisition_dir
                )
                if max_bytes and used_bytes >= max_bytes:
                    raise RuntimeError(
                        "Exact-union build reached safety.max_local_bytes before "
                        f"{pending[next_item][0].tile_id}: "
                        f"{used_bytes} >= {max_bytes}."
                    )
                tile, adql = pending[next_item]
                next_item += 1
                mark_exact_union_tile_running(
                    db_path,
                    pool_build_id=pool_build_id,
                    tile=tile,
                    adql=adql,
                    row_limit=row_limit,
                )
                future = executor.submit(
                    download_prediction_tile,
                    tile,
                    envelope,
                    config,
                    staging_dir,
                    row_limit,
                )
                futures[future] = (tile, adql)

            done, _ = wait(set(futures), return_when=FIRST_COMPLETED)
            for future in done:
                tile, adql = futures.pop(future)
                try:
                    csv_path, gaia_job_id = future.result()
                    raw = pd.read_csv(csv_path)
                    frame = prepare_acquisition_frame(
                        raw,
                        tile=tile,
                        loci=loci,
                        schema=schema,
                        known_source_labels=known_sources,
                    )
                    tile_result = persist_acquisition_tile(
                        frame,
                        tile=tile,
                        pool_build_id=pool_build_id,
                        acquisition_dir=acquisition_dir,
                        manifest_dir=manifest_dir,
                        compression=str(
                            config.get("parquet", {}).get(
                                "compression", "zstd"
                            )
                        ),
                        schema=schema,
                        envelope=envelope,
                        loci=loci,
                        adql=adql,
                        gaia_job_id=gaia_job_id,
                        row_limit=row_limit,
                    )
                    register_completed_tile(db_path, tile_result)
                    if bool(
                        config.get("gaia", {}).get(
                            "cleanup_staging_after_validated_ingest", True
                        )
                    ):
                        csv_path.unlink(missing_ok=True)
                    completed += 1
                except Exception as exc:
                    failed += 1
                    mark_exact_union_tile_failed(
                        db_path,
                        pool_build_id=pool_build_id,
                        tile_id=tile.tile_id,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    if not bool(
                        config.get("gaia", {}).get(
                            "continue_on_tile_error", True
                        )
                    ):
                        for pending_future in futures:
                            pending_future.cancel()
                        raise
                handled += 1
                if (
                    handled % progress_every == 0
                    or handled == len(pending)
                ):
                    elapsed = max(monotonic() - started_at, 1e-9)
                    rate = handled / elapsed
                    remaining = len(pending) - handled
                    eta_hours = remaining / rate / 3600 if rate else None
                    print(
                        f"[exact-union] handled={handled}/{len(pending)} "
                        f"completed={completed} failed={failed} "
                        f"rate_tiles_per_hour={rate * 3600:.2f} "
                        f"eta_hours={eta_hours:.2f}",
                        flush=True,
                    )

    refresh_exact_union_views(db_path, acquisition_dir)
    build_status = finalize_exact_union_build_status(
        db_path,
        pool_build_id=pool_build_id,
        expected_tile_ids=[tile.tile_id for tile in expected_tiles],
    )
    build_manifest_path = write_build_manifest(
        db_path,
        config_path=config_path,
        manifest_dir=manifest_dir,
        pool_build_id=pool_build_id,
        build_status=build_status,
        expected_tiles=len(expected_tiles),
        envelope=envelope,
        coverage=coverage,
        schema=schema,
        loci=loci,
    )
    return {
        "pool_build_id": pool_build_id,
        "db_path": str(db_path),
        "acquisition_dir": str(acquisition_dir),
        "build_manifest": str(build_manifest_path),
        "tiles_completed_this_run": completed,
        "tiles_skipped_as_valid": skipped,
        "tiles_failed_this_run": failed,
        "status": build_status,
    }


def audit_exact_union_prediction_pool(
    config_path: str | Path,
) -> dict[str, Any]:
    """Validate physical artifacts, lineage and exact-union invariants."""
    config = load_prediction_pool_config(config_path)
    pool_build_id = str(config["build"]["pool_build_id"])
    db_path = resolve_path(config["output_db"])
    acquisition_dir = resolve_path(config["output_dir"])
    manifest_dir = resolve_path(config["manifest_dir"])
    errors: list[str] = []
    if not db_path.exists():
        raise FileNotFoundError(db_path)

    with duckdb.connect(str(db_path), read_only=True) as con:
        tiles = con.execute(
            """
            SELECT tile_id, status, acquired_pre_locus, accepted_union,
                   known_excluded, written, parquet_path, parquet_bytes,
                   acquisition_sha256, manifest_path, manifest_sha256,
                   adql_sha256
            FROM exact_union_tiles
            WHERE pool_build_id=?
            ORDER BY tile_id
            """,
            [pool_build_id],
        ).fetchdf()
        build_row = con.execute(
            """
            SELECT status, envelope_sha256, bitmask_schema_version,
                   bitmask_schema_sha256
            FROM exact_union_builds WHERE pool_build_id=?
            """,
            [pool_build_id],
        ).fetchone()
    if build_row is None:
        errors.append("pool_build_id is missing from exact_union_builds")
        build_status = None
    else:
        build_status = str(build_row[0])
        if build_status != "completed":
            errors.append(f"build_status={build_status}")

    expected_tile_ids = {
        tile.tile_id for tile in make_sky_tiles(config)
    }
    registered_tile_ids = set(tiles["tile_id"].astype(str))
    completed_tile_ids = set(
        tiles.loc[tiles["status"].eq("completed"), "tile_id"].astype(str)
    )
    missing_registered = sorted(expected_tile_ids - registered_tile_ids)
    extra_registered = sorted(registered_tile_ids - expected_tile_ids)
    incomplete_expected = sorted(expected_tile_ids - completed_tile_ids)
    if missing_registered:
        errors.append(
            f"missing_registered_tiles={len(missing_registered)}"
        )
    if extra_registered:
        errors.append(f"extra_registered_tiles={len(extra_registered)}")
    if incomplete_expected:
        errors.append(
            f"incomplete_expected_tiles={len(incomplete_expected)}"
        )

    tile_results: list[dict[str, Any]] = []
    for row in tiles.to_dict("records"):
        tile_id = str(row["tile_id"])
        parquet_path = Path(str(row["parquet_path"]))
        manifest_path = Path(str(row["manifest_path"]))
        tile_errors: list[str] = []
        if row["status"] != "completed":
            tile_errors.append(f"status={row['status']}")
        if not parquet_path.exists():
            tile_errors.append("parquet_missing")
        elif file_sha256(parquet_path) != row["acquisition_sha256"]:
            tile_errors.append("parquet_sha256_mismatch")
        if not manifest_path.exists():
            tile_errors.append("manifest_missing")
            manifest = {}
        else:
            if file_sha256(manifest_path) != row["manifest_sha256"]:
                tile_errors.append("manifest_sha256_mismatch")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("adql_sha256") != row["adql_sha256"]:
                tile_errors.append("adql_sha256_mismatch")
            if manifest.get("acquisition_sha256") != row["acquisition_sha256"]:
                tile_errors.append("manifest_parquet_sha256_mismatch")

        physical: tuple[Any, ...] | None = None
        if parquet_path.exists():
            with duckdb.connect() as con:
                physical = con.execute(
                    """
                    SELECT
                        COUNT(*) AS rows,
                        COUNT(DISTINCT source_id) AS unique_sources,
                        COUNT(*) FILTER (
                            WHERE compatible_variant_count <>
                                  bit_count(compatible_variant_mask)
                        ) AS count_mask_errors,
                        COUNT(*) FILTER (
                            WHERE passes_any_exact_variant <>
                                  (compatible_variant_count > 0)
                        ) AS union_errors,
                        COUNT(*) FILTER (
                            WHERE compatible_variant_mask <>
                                  (
                                    exact_locus_variant_mask &
                                    photometry_variant_mask &
                                    astrometry_variant_mask
                                  )
                        ) AS component_mask_errors,
                        COUNT(*) FILTER (
                            WHERE passes_any_exact_variant
                        ) AS accepted_union,
                        COUNT(*) FILTER (
                            WHERE passes_any_exact_variant
                              AND known_source_exclusion IS NOT NULL
                        ) AS known_excluded,
                        COUNT(*) FILTER (
                            WHERE passes_any_exact_variant
                              AND known_source_exclusion IS NULL
                        ) AS eligible_written
                    FROM read_parquet(?)
                    """,
                    [str(parquet_path)],
                ).fetchone()
            expected = [
                int(row["acquired_pre_locus"]),
                int(row["acquired_pre_locus"]),
                0,
                0,
                0,
                int(row["accepted_union"]),
                int(row["known_excluded"]),
                int(row["written"]),
            ]
            if [int(value) for value in physical] != expected:
                tile_errors.append(
                    f"physical_counts_mismatch={tuple(physical)} expected={tuple(expected)}"
                )
            if parquet_path.stat().st_size != int(row["parquet_bytes"]):
                tile_errors.append("parquet_bytes_mismatch")
        if tile_errors:
            errors.extend(f"{tile_id}: {error}" for error in tile_errors)
        tile_results.append(
            {
                "tile_id": tile_id,
                "status": row["status"],
                "physical_rows": int(physical[0]) if physical else None,
                "errors": tile_errors,
            }
        )

    global_counts: dict[str, int] = {}
    parquet_files = sorted(acquisition_dir.glob("*.parquet"))
    if parquet_files:
        with duckdb.connect(str(db_path), read_only=True) as con:
            values = con.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM acquisition_sources),
                    (SELECT COUNT(DISTINCT source_id) FROM acquisition_sources),
                    (SELECT COUNT(*) FROM prediction_pool_sources),
                    (SELECT COUNT(*) FROM prediction_pool_effective_tiles)
                """
            ).fetchone()
        global_counts = {
            "acquisition_rows": int(values[0]),
            "distinct_source_ids": int(values[1]),
            "operational_eligible_rows": int(values[2]),
            "effective_tiles": int(values[3]),
        }
        if global_counts["acquisition_rows"] != global_counts["distinct_source_ids"]:
            errors.append("source_id values overlap across acquisition tiles")
        if global_counts["effective_tiles"] != len(tiles):
            errors.append("effective tile count does not match registered tile count")

    build_manifest_path = manifest_dir / "build_manifest.json"
    if not build_manifest_path.exists():
        errors.append("build_manifest.json is missing")
    else:
        build_manifest = json.loads(
            build_manifest_path.read_text(encoding="utf-8")
        )
        if build_manifest.get("status") != "completed":
            errors.append(
                "build_manifest_status="
                f"{build_manifest.get('status')}"
            )
        manifest_counts = build_manifest.get("counts", {})
        if int(manifest_counts.get("expected_tiles", -1)) != len(
            expected_tile_ids
        ):
            errors.append("build_manifest_expected_tiles_mismatch")
        if int(manifest_counts.get("completed_tiles", -1)) != len(
            completed_tile_ids
        ):
            errors.append("build_manifest_completed_tiles_mismatch")
    return {
        "status": "passed" if not errors else "failed",
        "build_status": build_status,
        "pool_build_id": pool_build_id,
        "database_path": str(db_path),
        "expected_tiles": int(len(expected_tile_ids)),
        "registered_tiles": int(len(tiles)),
        "parquet_files": int(len(parquet_files)),
        "global_counts": global_counts,
        "errors": errors,
        "tiles": tile_results,
    }


def prepare_acquisition_frame(
    raw: pd.DataFrame,
    *,
    tile: SkyTile,
    loci: Mapping[str, ExactLocus],
    schema: VariantMaskSchema,
    known_source_labels: Mapping[int, str] | None = None,
) -> pd.DataFrame:
    """Annotate every acquired row without deleting locus-ineligible sources."""
    missing = sorted(set(BASE_COLUMNS) - set(raw.columns))
    if missing:
        raise ValueError(f"Acquisition is missing required source columns: {missing}")
    frame = deduplicate_tmass_crossmatches(raw)
    if frame["source_id"].duplicated().any():
        raise ValueError(
            f"Acquisition remains duplicated after deterministic 2MASS "
            f"resolution in {tile.tile_id}."
        )
    frame.insert(1, "tile_id", tile.tile_id)
    frame = evaluate_exact_variant_union(frame, loci, schema)
    labels = known_source_labels or {}
    numeric_source_id = pd.to_numeric(frame["source_id"], errors="raise").astype(
        "int64"
    )
    frame["known_source_exclusion"] = numeric_source_id.map(labels).astype(
        "string"
    )
    frame["source_hash_v1"] = source_hash_v1(frame)
    frame["annotated_at"] = datetime.now(UTC)
    return frame


def deduplicate_tmass_crossmatches(raw: pd.DataFrame) -> pd.DataFrame:
    """Choose one 2MASS counterpart while preserving every alternative as JSON."""
    frame = raw.copy()
    frame["_tmass_ab_count"] = frame["tmass_quality"].astype("string").map(
        lambda value: _quality_count(value, {"A", "B"}, 3)
    )
    frame["_tmass_a_count"] = frame["tmass_quality"].astype("string").map(
        lambda value: _quality_count(value, {"A"}, 3)
    )
    error_columns = ["J_error", "H_error", "Ks_error"]
    errors = frame[error_columns].apply(pd.to_numeric, errors="coerce")
    frame["_tmass_error_sum"] = errors.fillna(float("inf")).sum(axis=1)
    frame["_tmass_id_sort"] = frame["tmass_id"].astype("string").fillna("")
    counts_by_source = frame.groupby("source_id", dropna=False).size()
    candidate_count = frame["source_id"].map(counts_by_source)
    alternative_fields = [
        "tmass_id",
        "tmass_quality",
        "J",
        "H",
        "Ks",
        "J_error",
        "H_error",
        "Ks_error",
    ]
    alternatives: dict[Any, str] = {}
    duplicate_rows = frame[candidate_count.gt(1)]
    for source_id, group in duplicate_rows.groupby("source_id", sort=False):
        records = []
        for record in group[alternative_fields].to_dict("records"):
            records.append(
                {
                    key: (
                        None
                        if pd.isna(value)
                        else value.item()
                        if hasattr(value, "item")
                        else value
                    )
                    for key, value in record.items()
                }
            )
        alternatives[source_id] = json.dumps(
            records, sort_keys=True, separators=(",", ":")
        )
    frame = frame.sort_values(
        [
            "source_id",
            "_tmass_ab_count",
            "_tmass_a_count",
            "_tmass_error_sum",
            "_tmass_id_sort",
        ],
        ascending=[True, False, False, True, True],
        kind="mergesort",
    ).drop_duplicates("source_id", keep="first")
    frame["tmass_candidate_count"] = frame["source_id"].map(
        counts_by_source
    ).astype("int16")
    frame["tmass_alternatives_json"] = frame["source_id"].map(alternatives).astype(
        "string"
    )
    frame["tmass_selection_rule"] = TMASS_SELECTION_RULE
    return frame.drop(
        columns=[
            "_tmass_ab_count",
            "_tmass_a_count",
            "_tmass_error_sum",
            "_tmass_id_sort",
        ]
    ).reset_index(drop=True)


def _quality_count(
    value: object,
    allowed: set[str],
    required_bands: int,
) -> int:
    if value is None or pd.isna(value):
        return 0
    text = str(value)
    return sum(
        1
        for position in range(min(required_bands, len(text)))
        if text[position] in allowed
    )


def source_hash_v1(frame: pd.DataFrame) -> pd.Series:
    """Return a compact deterministic conflict-detection hash for source payloads."""
    missing = sorted(set(SOURCE_HASH_V1_FIELDS) - set(frame.columns))
    if missing:
        raise ValueError(f"source_hash_v1 fields are missing: {missing}")
    payload = frame.loc[:, SOURCE_HASH_V1_FIELDS].copy()
    hashes = pd.util.hash_pandas_object(
        payload, index=False, categorize=True
    ).astype("uint64")
    return hashes.map(lambda value: f"{int(value):016x}")


def persist_acquisition_tile(
    frame: pd.DataFrame,
    *,
    tile: SkyTile,
    pool_build_id: str,
    acquisition_dir: Path,
    manifest_dir: Path,
    compression: str,
    schema: VariantMaskSchema,
    envelope: Mapping[str, Any],
    loci: Mapping[str, ExactLocus],
    adql: str,
    gaia_job_id: str | None,
    row_limit: int | None,
) -> dict[str, Any]:
    """Atomically write and validate one immutable acquisition Parquet."""
    acquisition_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = acquisition_dir / f"{tile.tile_id}.parquet"
    temporary_path = parquet_path.with_suffix(".parquet.tmp")
    frame.to_parquet(
        temporary_path,
        index=False,
        compression=compression,
    )
    with duckdb.connect() as con:
        physical_rows, unique_sources = con.execute(
            "SELECT COUNT(*), COUNT(DISTINCT source_id) FROM read_parquet(?)",
            [str(temporary_path)],
        ).fetchone()
    if int(physical_rows) != len(frame) or int(unique_sources) != len(frame):
        temporary_path.unlink(missing_ok=True)
        raise ValueError(
            f"Parquet validation failed for {tile.tile_id}: "
            f"rows={physical_rows}, unique_sources={unique_sources}, expected={len(frame)}"
        )
    temporary_path.replace(parquet_path)
    parquet_sha256 = file_sha256(parquet_path)
    counts = _tile_counts(frame, pool_build_id, tile.tile_id, schema)
    manifest = {
        **counts,
        "status": "completed",
        "ra_min": tile.ra_min,
        "ra_max": tile.ra_max,
        "dec_min": tile.dec_min,
        "dec_max": tile.dec_max,
        "row_limit": row_limit,
        "acquisition_envelope_sha256": envelope["sha256"],
        "adql": adql,
        "adql_sha256": sha256(adql.encode("utf-8")).hexdigest(),
        "gaia_job_id": gaia_job_id,
        "parquet_path": str(parquet_path),
        "parquet_bytes": parquet_path.stat().st_size,
        "acquisition_sha256": parquet_sha256,
        "source_hash_v1_algorithm": SOURCE_HASH_V1_ALGORITHM,
        "source_hash_v1_fields": SOURCE_HASH_V1_FIELDS,
        "loci": {
            variant: {
                "locus_run_id": locus.locus_run_id,
                "source_sha256": locus.source_sha256,
            }
            for variant, locus in loci.items()
        },
        "completed_at": datetime.now(UTC).isoformat(),
    }
    manifest_path = manifest_dir / f"{tile.tile_id}.json"
    atomic_write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    manifest["manifest_sha256"] = file_sha256(manifest_path)
    return manifest


def _tile_counts(
    frame: pd.DataFrame,
    pool_build_id: str,
    tile_id: str,
    schema: VariantMaskSchema,
) -> dict[str, Any]:
    known = frame["known_source_exclusion"].notna()
    union = frame["passes_any_exact_variant"].fillna(False).astype(bool)
    annotated = frame.copy()
    annotated["_known_union_exclusion"] = known
    record = build_tile_count_record(
        annotated,
        pool_build_id=pool_build_id,
        tile_id=tile_id,
        schema=schema,
        known_excluded_column="_known_union_exclusion",
    )
    record["acquisition_rows_written"] = int(len(frame))
    record["known_sources_acquired"] = int(known.sum())
    record["known_sources_eligible"] = int((known & union).sum())
    return record


def initialize_exact_union_database(
    db_path: Path,
    *,
    pool_build_id: str,
    config_path: Path,
    envelope: Mapping[str, Any],
    coverage: Mapping[str, Any],
    schema: VariantMaskSchema,
    loci: Mapping[str, ExactLocus],
) -> None:
    with duckdb.connect(str(db_path)) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS exact_union_builds (
                pool_build_id VARCHAR PRIMARY KEY,
                status VARCHAR,
                config_path VARCHAR,
                envelope_sha256 VARCHAR,
                bitmask_schema_version VARCHAR,
                bitmask_schema_sha256 VARCHAR,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS exact_union_tiles (
                pool_build_id VARCHAR,
                tile_id VARCHAR,
                ra_min DOUBLE,
                ra_max DOUBLE,
                dec_min DOUBLE,
                dec_max DOUBLE,
                status VARCHAR,
                row_limit BIGINT,
                adql_sha256 VARCHAR,
                gaia_job_id VARCHAR,
                acquired_pre_locus BIGINT,
                accepted_union BIGINT,
                known_excluded BIGINT,
                written BIGINT,
                parquet_path VARCHAR,
                parquet_bytes BIGINT,
                acquisition_sha256 VARCHAR,
                manifest_path VARCHAR,
                manifest_sha256 VARCHAR,
                error_message VARCHAR,
                updated_at TIMESTAMP,
                PRIMARY KEY (pool_build_id, tile_id)
            );
            """
        )
        existing = con.execute(
            """
            SELECT envelope_sha256, bitmask_schema_sha256
            FROM exact_union_builds
            WHERE pool_build_id = ?
            """,
            [pool_build_id],
        ).fetchone()
        if existing and tuple(existing) != (envelope["sha256"], schema.sha256):
            raise ValueError(
                "The pool_build_id already exists with a different envelope or "
                "bitmask schema. Use a new pool_build_id."
            )
        now = datetime.now(UTC)
        con.execute(
            """
            INSERT INTO exact_union_builds VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (pool_build_id) DO UPDATE SET
                updated_at = excluded.updated_at
            """,
            [
                pool_build_id,
                "running",
                str(config_path),
                envelope["sha256"],
                schema.version,
                schema.sha256,
                now,
                now,
            ],
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE exact_union_contract AS
            SELECT ?::JSON AS envelope, ?::JSON AS coverage,
                   ?::JSON AS bitmask_schema, ?::JSON AS loci
            """,
            [
                json.dumps(envelope),
                json.dumps(coverage),
                json.dumps(schema.as_manifest()),
                json.dumps(
                    {
                        key: {
                            "locus_run_id": value.locus_run_id,
                            "source_path": value.source_path,
                            "source_sha256": value.source_sha256,
                            "required_colors": value.required_colors,
                        }
                        for key, value in loci.items()
                    }
                ),
            ],
        )


def mark_exact_union_tile_running(
    db_path: Path,
    *,
    pool_build_id: str,
    tile: SkyTile,
    adql: str,
    row_limit: int | None,
) -> None:
    with duckdb.connect(str(db_path)) as con:
        con.execute(
            """
            INSERT INTO exact_union_tiles (
                pool_build_id, tile_id, ra_min, ra_max, dec_min, dec_max,
                status, row_limit, adql_sha256, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, 'running', ?, ?, ?)
            ON CONFLICT (pool_build_id, tile_id) DO UPDATE SET
                status = 'running',
                row_limit = excluded.row_limit,
                adql_sha256 = excluded.adql_sha256,
                error_message = NULL,
                updated_at = excluded.updated_at
            """,
            [
                pool_build_id,
                tile.tile_id,
                tile.ra_min,
                tile.ra_max,
                tile.dec_min,
                tile.dec_max,
                row_limit,
                sha256(adql.encode("utf-8")).hexdigest(),
                datetime.now(UTC),
            ],
        )


def register_completed_tile(db_path: Path, manifest: Mapping[str, Any]) -> None:
    with duckdb.connect(str(db_path)) as con:
        con.execute(
            """
            UPDATE exact_union_tiles SET
                status = 'completed',
                gaia_job_id = ?,
                acquired_pre_locus = ?,
                accepted_union = ?,
                known_excluded = ?,
                written = ?,
                parquet_path = ?,
                parquet_bytes = ?,
                acquisition_sha256 = ?,
                manifest_path = ?,
                manifest_sha256 = ?,
                error_message = NULL,
                updated_at = ?
            WHERE pool_build_id = ? AND tile_id = ?
            """,
            [
                manifest.get("gaia_job_id"),
                manifest["acquired_pre_locus"],
                manifest["accepted_union"],
                manifest["known_excluded"],
                manifest["written"],
                manifest["parquet_path"],
                manifest["parquet_bytes"],
                manifest["acquisition_sha256"],
                manifest["manifest_path"],
                manifest["manifest_sha256"],
                datetime.now(UTC),
                manifest["pool_build_id"],
                manifest["tile_id"],
            ],
        )


def mark_exact_union_tile_failed(
    db_path: Path,
    *,
    pool_build_id: str,
    tile_id: str,
    error: str,
) -> None:
    with duckdb.connect(str(db_path)) as con:
        con.execute(
            """
            UPDATE exact_union_tiles
            SET status='failed', error_message=?, updated_at=?
            WHERE pool_build_id=? AND tile_id=?
            """,
            [error, datetime.now(UTC), pool_build_id, tile_id],
        )


def completed_tile_is_valid(
    db_path: Path,
    *,
    acquisition_dir: Path,
    pool_build_id: str,
    tile_id: str,
    adql_sha256: str | None = None,
    row_limit: int | None = None,
    envelope_sha256: str | None = None,
    bitmask_schema_version: str | None = None,
    bitmask_schema_sha256: str | None = None,
) -> bool:
    if not db_path.exists():
        return False
    with duckdb.connect(str(db_path), read_only=True) as con:
        tables = {
            row[0]
            for row in con.execute(
                "SELECT table_name FROM information_schema.tables"
            ).fetchall()
        }
        if "exact_union_tiles" not in tables:
            return False
        row = con.execute(
            """
            SELECT status, acquired_pre_locus, acquisition_sha256,
                   manifest_path, manifest_sha256, adql_sha256, row_limit
            FROM exact_union_tiles
            WHERE pool_build_id=? AND tile_id=?
            """,
            [pool_build_id, tile_id],
        ).fetchone()
    if not row or row[0] != "completed":
        return False
    if adql_sha256 is not None and str(row[5]) != adql_sha256:
        return False
    if row_limit is not None or row[6] is not None:
        if row_limit != row[6]:
            return False
    path = acquisition_dir / f"{tile_id}.parquet"
    if not path.exists() or file_sha256(path) != row[2]:
        return False
    manifest_path = Path(str(row[3]))
    if (
        not manifest_path.exists()
        or not row[4]
        or file_sha256(manifest_path) != row[4]
    ):
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        envelope_sha256 is not None
        and manifest.get("acquisition_envelope_sha256") != envelope_sha256
    ):
        return False
    if (
        bitmask_schema_version is not None
        and manifest.get("bitmask_schema_version") != bitmask_schema_version
    ):
        return False
    if (
        bitmask_schema_sha256 is not None
        and manifest.get("bitmask_schema_sha256") != bitmask_schema_sha256
    ):
        return False
    with duckdb.connect() as con:
        count = con.execute(
            "SELECT COUNT(*) FROM read_parquet(?)", [str(path)]
        ).fetchone()[0]
        columns = {
            str(item[0])
            for item in con.execute(
                "DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]
            ).fetchall()
        }
    required_columns = {
        "source_id",
        "source_hash_v1",
        "exact_locus_variant_mask",
        "photometry_variant_mask",
        "astrometry_variant_mask",
        "compatible_variant_mask",
        "compatible_variant_count",
        "passes_any_exact_variant",
        *TMASS_CROSSMATCH_LINEAGE_FIELDS,
    }
    if not required_columns.issubset(columns):
        return False
    return int(count) == int(row[1])


def refresh_exact_union_views(db_path: Path, acquisition_dir: Path) -> None:
    files = sorted(acquisition_dir.glob("*.parquet"))
    if not files:
        return
    glob_path = (acquisition_dir / "*.parquet").as_posix().replace("'", "''")
    with duckdb.connect(str(db_path)) as con:
        con.execute(
            f"""
            CREATE OR REPLACE VIEW acquisition_sources AS
            SELECT * FROM read_parquet('{glob_path}', union_by_name=true);

            CREATE OR REPLACE VIEW prediction_pool_sources AS
            SELECT *
            FROM acquisition_sources
            WHERE passes_any_exact_variant
              AND known_source_exclusion IS NULL;

            CREATE OR REPLACE VIEW prediction_pool_effective_tiles AS
            SELECT *
            FROM exact_union_tiles
            WHERE status = 'completed';
            """
        )


def finalize_exact_union_build_status(
    db_path: Path,
    *,
    pool_build_id: str,
    expected_tile_ids: Iterable[str],
) -> str:
    """Mark a build completed only for the exact configured terminal tile set."""
    expected = {str(value) for value in expected_tile_ids}
    with duckdb.connect(str(db_path)) as con:
        rows = con.execute(
            """
            SELECT tile_id, status
            FROM exact_union_tiles
            WHERE pool_build_id=?
            """,
            [pool_build_id],
        ).fetchall()
        registered = {str(tile_id) for tile_id, _status in rows}
        completed = {
            str(tile_id)
            for tile_id, status in rows
            if str(status) == "completed"
        }
        status = (
            "completed"
            if registered == expected and completed == expected
            else "partial"
        )
        con.execute(
            """
            UPDATE exact_union_builds
            SET status=?, updated_at=?
            WHERE pool_build_id=?
            """,
            [status, datetime.now(UTC), pool_build_id],
        )
    return status


def write_build_manifest(
    db_path: Path,
    *,
    config_path: Path,
    manifest_dir: Path,
    pool_build_id: str,
    build_status: str,
    expected_tiles: int,
    envelope: Mapping[str, Any],
    coverage: Mapping[str, Any],
    schema: VariantMaskSchema,
    loci: Mapping[str, ExactLocus],
) -> Path:
    with duckdb.connect(str(db_path)) as con:
        totals = con.execute(
            """
            SELECT
                COUNT(*) FILTER (WHERE status='completed'),
                COUNT(*) FILTER (WHERE status='failed'),
                COALESCE(SUM(acquired_pre_locus) FILTER (WHERE status='completed'), 0),
                COALESCE(SUM(accepted_union) FILTER (WHERE status='completed'), 0),
                COALESCE(SUM(known_excluded) FILTER (WHERE status='completed'), 0),
                COALESCE(SUM(written) FILTER (WHERE status='completed'), 0)
            FROM exact_union_tiles
            WHERE pool_build_id=?
            """,
            [pool_build_id],
        ).fetchone()
    manifest = {
        "pool_build_id": pool_build_id,
        "status": build_status,
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "acquisition_envelope": envelope,
        "acquisition_envelope_coverage": coverage,
        "bitmask_schema": schema.as_manifest(),
        "source_hash_v1_algorithm": SOURCE_HASH_V1_ALGORITHM,
        "source_hash_v1_fields": SOURCE_HASH_V1_FIELDS,
        "runtime_lineage": runtime_lineage(),
        "loci": {
            variant: {
                "locus_run_id": locus.locus_run_id,
                "source_path": locus.source_path,
                "source_sha256": locus.source_sha256,
                "required_colors": locus.required_colors,
            }
            for variant, locus in loci.items()
        },
        "counts": {
            "expected_tiles": int(expected_tiles),
            "completed_tiles": int(totals[0]),
            "failed_tiles": int(totals[1]),
            "acquired_pre_locus": int(totals[2]),
            "accepted_union": int(totals[3]),
            "known_excluded": int(totals[4]),
            "eligible_written": int(totals[5]),
        },
        "written_at": datetime.now(UTC).isoformat(),
    }
    path = manifest_dir / "build_manifest.json"
    atomic_write_json(path, manifest)
    return path


def load_known_source_labels(config: Mapping[str, Any]) -> dict[int, str]:
    labels: dict[int, str] = {}
    sources = [
        (
            resolve_path(config["paths"]["wr_reference_db"]),
            "wr_reference",
            "known_wr",
        ),
        (
            resolve_path(config["paths"]["simbad_negative_db"]),
            "simbad_negative_sources",
            "known_non_wr_simbad",
        ),
    ]
    for path, table, label in sources:
        if not path.exists():
            continue
        with duckdb.connect(str(path), read_only=True) as con:
            tables = {
                row[0]
                for row in con.execute(
                    "SELECT table_name FROM information_schema.tables"
                ).fetchall()
            }
            if table not in tables:
                continue
            rows = con.execute(
                f"SELECT DISTINCT source_id FROM {table} WHERE source_id IS NOT NULL"
            ).fetchall()
        for (source_id,) in rows:
            labels.setdefault(int(source_id), label)
    return labels


def runtime_lineage() -> dict[str, Any]:
    """Record reproducible code/runtime identifiers without environment secrets."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        commit = None
        dirty = None
    packages: dict[str, str | None] = {}
    for package in ["wolf-rayet-detector", "pandas", "pyarrow", "duckdb", "astropy"]:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
