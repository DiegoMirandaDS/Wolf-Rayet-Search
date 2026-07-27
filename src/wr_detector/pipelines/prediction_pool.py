"""Tiled Gaia prediction-pool construction, ingestion, exclusion, and coverage auditing."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
import json
from pathlib import Path
from threading import Lock, local
from time import sleep
from typing import Any, Mapping

import duckdb
import pandas as pd

from wr_detector.config import load_yaml, load_yaml_with_extends, resolve_path
from wr_detector.db import connect


COLOR_EXPRESSIONS = {
    "G_BP": "gaia.phot_g_mean_mag - gaia.phot_bp_mean_mag",
    "G_RP": "gaia.phot_g_mean_mag - gaia.phot_rp_mean_mag",
    "BP_RP": "gaia.phot_bp_mean_mag - gaia.phot_rp_mean_mag",
    "J_H": "tmass.j_m - tmass.h_m",
    "J_K": "tmass.j_m - tmass.ks_m",
    "H_K": "tmass.h_m - tmass.ks_m",
    "W1_W2": "wise.w1mpro - wise.w2mpro",
}

LOCAL_COLOR_EXPRESSIONS = {
    "G_BP": "G - BP",
    "G_RP": "G - RP",
    "BP_RP": "BP - RP",
    "J_H": "J - H",
    "J_K": "J - Ks",
    "H_K": "H - Ks",
    "W1_W2": "W1 - W2",
}

BASE_COLUMNS = [
    "source_id",
    "gaia_designation",
    "ra",
    "dec",
    "G",
    "BP",
    "RP",
    "G_flux",
    "G_flux_error",
    "BP_flux",
    "BP_flux_error",
    "RP_flux",
    "RP_flux_error",
    "J",
    "H",
    "Ks",
    "J_error",
    "H_error",
    "Ks_error",
    "W1",
    "W2",
    "W3",
    "W4",
    "W1_error",
    "W2_error",
    "W3_error",
    "W4_error",
    "parallax",
    "parallax_error",
    "parallax_over_error",
    "pmra",
    "pmdec",
    "tmass_id",
    "tmass_quality",
    "wise_id",
    "wise_quality",
]


_THREAD_LOCAL = local()
_DB_WRITE_LOCK = Lock()


@dataclass(frozen=True)
class SkyTile:
    tile_id: str
    ra_min: float
    ra_max: float
    dec_min: float
    dec_max: float


def build_prediction_pool(
    config_path: str | Path,
    *,
    dry_run: bool = False,
    max_tiles: int | None = None,
    row_limit: int | None = None,
) -> dict[str, Any]:
    config = load_prediction_pool_config(config_path)
    assert_prediction_pool_build_allowed(config, dry_run=dry_run)
    db_path = resolve_path(config["output_db"])
    staging_dir = resolve_path(config["staging_dir"])
    output_dir = resolve_path(config["output_dir"])
    staging_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    envelope = derive_color_envelope(config)
    tiles = make_sky_tiles(config)
    if max_tiles is not None:
        tiles = tiles[:max_tiles]

    if dry_run:
        sample_tile = tiles[0] if tiles else None
        return {
            "dry_run": True,
            "tiles": len(tiles),
            "sample_adql": build_prediction_pool_adql(sample_tile, envelope, config, row_limit=row_limit)
            if sample_tile
            else "",
        }

    con = connect(db_path)
    try:
        initialize_prediction_pool_database(con)
        refresh_prediction_pool_view(con, output_dir)
        refresh_known_sources(con, config)
        ensure_tiles(con, tiles)
        reset_stale_running_tiles(con, config)
        subdivide_failed_tiles(con, config)
    finally:
        con.close()

    completed = 0
    skipped_for_limit = 0
    max_workers = max(1, int(config.get("safety", {}).get("max_workers", 1)))
    ingest_workers = max(1, int(config.get("safety", {}).get("ingest_workers", 1)))
    if get_gaia_credentials_file(config) and max_workers == 1:
        login_to_gaia(config)
    pending_tiles = load_runnable_tiles(db_path)
    if max_tiles is not None:
        pending_tiles = pending_tiles[:max_tiles]

    next_tile = 0
    skipped_limit_applied = False
    download_futures = {}
    ingest_futures = {}
    with (
        ThreadPoolExecutor(max_workers=max_workers) as download_executor,
        ThreadPoolExecutor(max_workers=ingest_workers) as ingest_executor,
    ):
        while next_tile < len(pending_tiles) or download_futures or ingest_futures:
            while next_tile < len(pending_tiles) and len(download_futures) < max_workers:
                if local_storage_bytes(db_path, staging_dir, output_dir) >= int(config["safety"]["max_local_bytes"]):
                    skipped_for_limit += mark_remaining_skipped(db_path, pending_tiles[next_tile:], reason="local_byte_limit")
                    next_tile = len(pending_tiles)
                    skipped_limit_applied = True
                    break
                tile = pending_tiles[next_tile]
                next_tile += 1
                mark_tile_running(db_path, tile)
                future = download_executor.submit(download_prediction_tile, tile, envelope, config, staging_dir, row_limit)
                download_futures[future] = tile

            active_futures = set(download_futures) | set(ingest_futures)
            if not active_futures:
                break

            done, _ = wait(active_futures, return_when=FIRST_COMPLETED)
            for future in done:
                if future in download_futures:
                    tile = download_futures.pop(future)
                    try:
                        csv_path, job_id = future.result()
                    except Exception as exc:
                        mark_tile_failed(db_path, tile, str(exc))
                        if not config.get("gaia", {}).get("continue_on_tile_error", True):
                            raise
                    else:
                        ingest_future = ingest_executor.submit(
                            ingest_downloaded_prediction_tile,
                            db_path,
                            csv_path,
                            tile,
                            envelope,
                            config,
                            job_id,
                        )
                        ingest_futures[ingest_future] = tile
                elif future in ingest_futures:
                    tile = ingest_futures.pop(future)
                    try:
                        future.result()
                        completed += 1
                    except Exception as exc:
                        mark_tile_failed(db_path, tile, str(exc))
                        if not config.get("gaia", {}).get("continue_on_tile_error", True):
                            raise

            if (
                not skipped_limit_applied
                and local_storage_bytes(db_path, staging_dir, output_dir) >= int(config["safety"]["max_local_bytes"])
            ):
                skipped_for_limit += mark_remaining_skipped(db_path, pending_tiles[next_tile:], reason="local_byte_limit")
                next_tile = len(pending_tiles)
                skipped_limit_applied = True

    return {
        "db_path": str(db_path),
        "staging_dir": str(staging_dir),
        "output_dir": str(output_dir),
        "tiles_requested": len(tiles),
        "tiles_completed_this_run": completed,
        "tiles_skipped_for_limit": skipped_for_limit,
        "local_storage_bytes": local_storage_bytes(db_path, staging_dir, output_dir),
    }


def assert_prediction_pool_build_allowed(
    config: dict[str, Any],
    *,
    dry_run: bool,
) -> None:
    """Prevent mutation of a pool configuration registered as legacy/read-only."""
    build = config.get("build", {})
    status = str(build.get("status", "")).strip().lower()
    mutation_allowed = bool(build.get("allow_mutation", True))
    if not dry_run and (status == "legacy_read_only" or not mutation_allowed):
        pool_build_id = str(build.get("pool_build_id", "unregistered_legacy_pool"))
        policy = str(build.get("eligibility_policy", "unknown"))
        raise RuntimeError(
            f"Prediction-pool build {pool_build_id!r} is read-only "
            f"(eligibility_policy={policy!r}). The current full-build pipeline "
            "still implements the legacy aggregate and must not overwrite this pool. "
            "Use --dry-run for inspection or audit-prediction-pool for read-only review. "
            "A separate exact-union build configuration must be implemented and validated "
            "before the Gaia-scale reconstruction."
        )


def audit_prediction_pool(db_path: str | Path) -> dict[str, Any]:
    path = resolve_path(db_path)
    con = duckdb.connect(str(path))
    try:
        tables = {
            row[0]
            for row in con.execute("SELECT table_name FROM information_schema.tables").fetchall()
        }
        if "prediction_pool_tiles" in tables and "prediction_pool_parquet_files" in tables and "prediction_pool_effective_tiles" not in tables:
            refresh_prediction_pool_metadata_views(con)
            tables = {
                row[0]
                for row in con.execute("SELECT table_name FROM information_schema.tables").fetchall()
            }
        counts = {
            table: int(con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in [
                "prediction_pool_sources",
                "prediction_pool_effective_tiles",
                "prediction_pool_tile_coverage",
                "prediction_pool_parquet_files",
                "prediction_pool_tiles",
                "prediction_pool_query_log",
                "prediction_pool_exclusions",
                "prediction_pool_known_sources",
            ]
            if table in tables
        }
        tile_status = {}
        if "prediction_pool_tiles" in tables:
            tile_status = dict(
                con.execute(
                    "SELECT status, COUNT(*) FROM prediction_pool_tiles GROUP BY status ORDER BY status"
                ).fetchall()
            )
        exclusions = {}
        if "prediction_pool_exclusions" in tables:
            exclusions = dict(
                con.execute(
                    "SELECT exclusion_label, COUNT(*) FROM prediction_pool_exclusions GROUP BY exclusion_label ORDER BY exclusion_label"
                ).fetchall()
            )
        return {
            "db_path": str(path),
            "db_size_bytes": path.stat().st_size if path.exists() else 0,
            "counts": counts,
            "tile_status": tile_status,
            "exclusions": exclusions,
        }
    finally:
        con.close()


def load_prediction_pool_config(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml_with_extends(config_path)
    config["paths"] = load_yaml(config["paths_config"])
    config["filters"] = load_yaml(config["filters_config"])
    env_file = config.get("gaia", {}).get("credentials_env_file")
    if env_file:
        config["_env"] = load_env_file(resolve_path(env_file))
    else:
        config["_env"] = {}
    return config


def load_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def derive_color_envelope(config: dict[str, Any]) -> dict[str, Any]:
    paths = config["paths"]
    filters = config["filters"]
    reference_dir = resolve_path(paths["processed_reference_dir"])
    variants = config.get("color_envelope", {}).get("variants", [])
    if not variants:
        variants = [v for v in filters["color_locus"]["dataset_variants"] if v.startswith("relaxed_")]

    frames = []
    fits: dict[tuple[str, str], dict[str, Any]] = {}
    for variant in variants:
        output = filters["color_locus"]["reference_output_template"].format(variant=variant)
        path = reference_dir / output
        df = pd.read_parquet(path)
        frames.append(df)
        for plane in str(df["color_locus_planes"].iloc[0]).split(","):
            if not plane:
                continue
            prefix = f"color_locus_{plane}"
            x_color, y_color = plane.split("__", 1)
            key = (x_color, y_color)
            fits.setdefault(
                key,
                {
                    "x": x_color,
                    "y": y_color,
                    "slope": [],
                    "intercept": [],
                    "threshold": [],
                    "transform": str(df["color_locus_transform"].iloc[0]),
                },
            )
            fits[key]["slope"].append(float(df[f"{prefix}_slope"].iloc[0]))
            fits[key]["intercept"].append(float(df[f"{prefix}_intercept"].iloc[0]))
            fits[key]["threshold"].append(float(df[f"{prefix}_threshold"].iloc[0]))

    if not frames:
        raise ValueError("No color-locus variants configured for prediction pool envelope.")

    union = pd.concat(frames, ignore_index=True)
    if config.get("color_envelope", {}).get("use_kept_rows", True) and "color_locus_keep" in union:
        kept = union[union["color_locus_keep"]].copy()
        if not kept.empty:
            union = kept

    configured_colors = config.get("color_envelope", {}).get(
        "colors", list(COLOR_EXPRESSIONS)
    )
    unknown_colors = sorted(set(configured_colors) - set(COLOR_EXPRESSIONS))
    if unknown_colors:
        raise ValueError(
            "Prediction-pool acquisition supports only the declared intra-mission "
            f"colors; unknown colors: {unknown_colors}"
        )
    color_bounds = {}
    padding_config = config.get("color_envelope", {}).get("padding", {})
    span_fraction = float(padding_config.get("span_fraction", 0.0))
    absolute_padding = padding_config.get("absolute", {})
    if span_fraction < 0:
        raise ValueError("color_envelope.padding.span_fraction cannot be negative.")
    source_rows = int(len(union))
    source_hashes: dict[str, str] = {}
    for variant in variants:
        output = filters["color_locus"]["reference_output_template"].format(
            variant=variant
        )
        path = reference_dir / output
        source_hashes[variant] = sha256(path.read_bytes()).hexdigest()
    for color in configured_colors:
        values = pd.to_numeric(union[color], errors="coerce").dropna()
        if values.empty:
            continue
        observed_min = float(values.min())
        observed_max = float(values.max())
        padding = max(
            float(absolute_padding.get(color, 0.0)),
            (observed_max - observed_min) * span_fraction,
        )
        color_bounds[color] = {
            "min": observed_min - padding,
            "max": observed_max + padding,
            "observed_min": observed_min,
            "observed_max": observed_max,
            "padding": padding,
            "finite_reference_rows": int(len(values)),
        }

    fit_envelopes = []
    for fit in fits.values():
        fit_envelopes.append(
            {
                "x": fit["x"],
                "y": fit["y"],
                "slope_min": float(min(fit["slope"])),
                "slope_max": float(max(fit["slope"])),
                "intercept_min": float(min(fit["intercept"])),
                "intercept_max": float(max(fit["intercept"])),
                "threshold_max": float(max(fit["threshold"])),
                "transform": fit["transform"],
            }
        )

    envelope = {
        "schema_version": "wr_intra_mission_acquisition_envelope_v1",
        "color_bounds": color_bounds,
        "fits": fit_envelopes,
        "variants": variants,
        "reference_population": (
            "color_locus_kept_rows"
            if config.get("color_envelope", {}).get("use_kept_rows", True)
            else "all_finite_variant_reference_rows"
        ),
        "reference_rows": source_rows,
        "reference_artifact_sha256": source_hashes,
        "padding": {
            "span_fraction": span_fraction,
            "absolute": {
                color: float(absolute_padding.get(color, 0.0))
                for color in configured_colors
            },
        },
    }
    envelope["sha256"] = _canonical_json_sha256(envelope)
    return envelope


def audit_color_envelope_reference_coverage(
    config: dict[str, Any],
    envelope: dict[str, Any],
) -> dict[str, Any]:
    """Prove coverage for every finite WR row used to define the envelope."""
    filters = config["filters"]
    reference_dir = resolve_path(config["paths"]["processed_reference_dir"])
    frames = []
    for variant in envelope["variants"]:
        filename = filters["color_locus"]["reference_output_template"].format(
            variant=variant
        )
        frame = pd.read_parquet(reference_dir / filename)
        frame = frame.assign(_coverage_variant=variant)
        frames.append(frame)
    controls = pd.concat(frames, ignore_index=True)
    colors = list(envelope["color_bounds"])
    finite = pd.Series(True, index=controls.index)
    inside = pd.Series(True, index=controls.index)
    for color in colors:
        values = pd.to_numeric(controls[color], errors="coerce")
        finite &= values.notna()
        bounds = envelope["color_bounds"][color]
        inside &= values.between(
            float(bounds["min"]), float(bounds["max"]), inclusive="both"
        )
    eligible = controls[finite]
    outside = eligible[~inside[finite]]
    source_column = "source_id" if "source_id" in controls else None
    report = {
        "colors": colors,
        "reference_rows": int(len(controls)),
        "finite_reference_rows": int(finite.sum()),
        "covered_reference_rows": int((finite & inside).sum()),
        "outside_reference_rows": int(len(outside)),
        "coverage_fraction": (
            float((finite & inside).sum() / finite.sum()) if finite.any() else 0.0
        ),
        "missing_required_color_rows": int((~finite).sum()),
        "outside_source_ids": (
            [
                int(value)
                for value in pd.to_numeric(
                    outside[source_column], errors="coerce"
                ).dropna().drop_duplicates().head(100)
            ]
            if source_column
            else []
        ),
    }
    required = bool(
        config.get("color_envelope", {})
        .get("coverage_validation", {})
        .get("require_all_finite_reference_rows", False)
    )
    if required and report["outside_reference_rows"]:
        raise ValueError(
            "Acquisition envelope does not cover every finite WR reference row: "
            f"{report['outside_reference_rows']} rows are outside."
        )
    return report


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def make_sky_tiles(config: dict[str, Any]) -> list[SkyTile]:
    tiling = config.get("tiling", {})
    explicit_tiles = tiling.get("explicit_tiles", [])
    source_config = tiling.get("explicit_tiles_from_config")
    if source_config:
        source = load_yaml(source_config)
        source_key = str(tiling.get("explicit_tiles_key", "regions"))
        source_id_key = str(tiling.get("explicit_tile_id_key", "name"))
        explicit_tiles = [
            {
                "tile_id": item[source_id_key],
                "ra_min": item["ra_min"],
                "ra_max": item["ra_max"],
                "dec_min": item["dec_min"],
                "dec_max": item["dec_max"],
            }
            for item in source[source_key]
        ]
    if explicit_tiles:
        tiles = [
            SkyTile(
                tile_id=str(item["tile_id"]),
                ra_min=float(item["ra_min"]),
                ra_max=float(item["ra_max"]),
                dec_min=float(item["dec_min"]),
                dec_max=float(item["dec_max"]),
            )
            for item in explicit_tiles
        ]
        if len({tile.tile_id for tile in tiles}) != len(tiles):
            raise ValueError("tiling.explicit_tiles contains duplicate tile_id values.")
        for tile in tiles:
            if not (
                0 <= tile.ra_min < tile.ra_max <= 360
                and -90 <= tile.dec_min < tile.dec_max <= 90
            ):
                raise ValueError(f"Invalid explicit sky tile bounds: {tile}")
        return tiles
    ra_step = float(tiling["ra_step_deg"])
    dec_step = float(tiling["dec_step_deg"])
    tiles: list[SkyTile] = []
    dec = -90.0
    while dec < 90.0:
        ra = 0.0
        dec_max = min(90.0, dec + dec_step)
        while ra < 360.0:
            ra_max = min(360.0, ra + ra_step)
            tile_id = format_tile_id(ra, ra_max, dec, dec_max)
            tiles.append(SkyTile(tile_id=tile_id, ra_min=ra, ra_max=ra_max, dec_min=dec, dec_max=dec_max))
            ra = ra_max
        dec = dec_max
    parent_ids = set(tiling.get("pre_subdivide_parent_tile_ids", []))
    if not parent_ids:
        return tiles
    unknown = sorted(parent_ids - {tile.tile_id for tile in tiles})
    if unknown:
        raise ValueError(
            f"Unknown pre-subdivision parent tile ids: {unknown}"
        )
    child_ra_step = float(tiling.get("subtile_ra_step_deg", ra_step / 2))
    child_dec_step = float(tiling.get("subtile_dec_step_deg", dec_step / 2))
    expanded: list[SkyTile] = []
    for tile in tiles:
        if tile.tile_id not in parent_ids:
            expanded.append(tile)
            continue
        child_dec = tile.dec_min
        while child_dec < tile.dec_max:
            child_dec_max = min(tile.dec_max, child_dec + child_dec_step)
            child_ra = tile.ra_min
            while child_ra < tile.ra_max:
                child_ra_max = min(tile.ra_max, child_ra + child_ra_step)
                expanded.append(
                    SkyTile(
                        tile_id=format_tile_id(
                            child_ra,
                            child_ra_max,
                            child_dec,
                            child_dec_max,
                        ),
                        ra_min=child_ra,
                        ra_max=child_ra_max,
                        dec_min=child_dec,
                        dec_max=child_dec_max,
                    )
                )
                child_ra = child_ra_max
            child_dec = child_dec_max
    if len({tile.tile_id for tile in expanded}) != len(expanded):
        raise ValueError("Pre-subdivision generated duplicate child tile ids.")
    return expanded


def format_tile_id(ra_min: float, ra_max: float, dec_min: float, dec_max: float) -> str:
    return f"ra{ra_min:06.2f}_{ra_max:06.2f}__dec{dec_min:+06.2f}_{dec_max:+06.2f}".replace(".", "p")


def load_runnable_tiles(db_path: Path) -> list[SkyTile]:
    con = connect(db_path)
    try:
        rows = con.execute(
            """
            SELECT tile_id, ra_min, ra_max, dec_min, dec_max
            FROM prediction_pool_tiles
            WHERE status NOT IN ('completed', 'skipped')
            ORDER BY dec_min, ra_min
            """
        ).fetchall()
        return [
            SkyTile(tile_id=str(tile_id), ra_min=float(ra_min), ra_max=float(ra_max), dec_min=float(dec_min), dec_max=float(dec_max))
            for tile_id, ra_min, ra_max, dec_min, dec_max in rows
        ]
    finally:
        con.close()


def build_prediction_pool_adql(
    tile: SkyTile,
    envelope: dict[str, Any],
    config: dict[str, Any],
    *,
    row_limit: int | None = None,
    extra_gaia_columns: Mapping[str, str] | None = None,
) -> str:
    top = f"TOP {int(row_limit)} " if row_limit is not None else ""
    quality = (
        build_quality_predicate(config)
        if config.get("color_envelope", {}).get(
            "server_side_quality_filter", True
        )
        else ""
    )
    bounds = build_color_bounds_predicate(envelope["color_bounds"])
    astrometry = build_astrometry_predicate(config)
    predicates = [
        f"gaia.ra >= {tile.ra_min}",
        f"gaia.ra < {tile.ra_max}",
        f"gaia.dec >= {tile.dec_min}",
        f"gaia.dec < {tile.dec_max}",
        "gaia.phot_g_mean_mag IS NOT NULL",
        "gaia.phot_bp_mean_mag IS NOT NULL",
        "gaia.phot_rp_mean_mag IS NOT NULL",
        "tmass.j_m IS NOT NULL",
        "tmass.h_m IS NOT NULL",
        "tmass.ks_m IS NOT NULL",
        "wise.w1mpro IS NOT NULL",
        "wise.w2mpro IS NOT NULL",
        bounds,
    ]
    if quality:
        predicates.append(quality)
    if config.get("color_envelope", {}).get("server_side_locus_filter", False):
        predicates.append(build_color_locus_predicate(envelope["fits"]))
    if astrometry:
        predicates.append(astrometry)
    where = "\n        AND ".join(f"({p})" for p in predicates if p)
    configured_extra_gaia = {
        str(alias): str(column)
        for alias, column in config.get("gaia", {})
        .get("extra_source_columns", {})
        .items()
    }
    configured_extra_gaia.update(extra_gaia_columns or {})
    extra_select = ""
    for alias, column in configured_extra_gaia.items():
        if not alias.replace("_", "").isalnum() or not column.replace("_", "").isalnum():
            raise ValueError(f"Invalid Gaia column or alias: {column!r} AS {alias!r}")
        extra_select += f"        gaia.{column} AS {alias},\n"
    ap_columns = {
        str(alias): str(column)
        for alias, column in config.get("gaia", {})
        .get("astrophysical_parameters_columns", {})
        .items()
    }
    ap_select = ""
    for alias, column in ap_columns.items():
        if not alias.replace("_", "").isalnum() or not column.replace("_", "").isalnum():
            raise ValueError(
                f"Invalid astrophysical-parameters column or alias: "
                f"{column!r} AS {alias!r}"
            )
        ap_select += f"        ap.{column} AS {alias},\n"
    wise_extra_columns = {
        str(alias): str(column)
        for alias, column in config.get("photometry", {})
        .get("allwise_extra_columns", {})
        .items()
    }
    wise_extra_select = ""
    for alias, column in wise_extra_columns.items():
        if not alias.replace("_", "").isalnum() or not column.replace("_", "").isalnum():
            raise ValueError(
                f"Invalid AllWISE column or alias: {column!r} AS {alias!r}"
            )
        wise_extra_select += f"        wise.{column} AS {alias},\n"
    crossmatch_select = ""
    if bool(
        config.get("photometry", {}).get(
            "include_crossmatch_diagnostics", False
        )
    ):
        crossmatch_select = (
            "        xmatch.angular_distance AS tmass_angular_distance,\n"
            "        xmatch.number_of_neighbours AS tmass_number_of_neighbours,\n"
            "        xmatch.number_of_mates AS tmass_number_of_mates,\n"
            "        xmatch.xm_flag AS tmass_xm_flag,\n"
            "        wise_match.angular_distance AS wise_angular_distance,\n"
            "        wise_match.number_of_neighbours AS wise_number_of_neighbours,\n"
            "        wise_match.number_of_mates AS wise_number_of_mates,\n"
            "        wise_match.xm_flag AS wise_xm_flag,\n"
        )
    ap_join = (
        "\n    LEFT OUTER JOIN gaiadr3.astrophysical_parameters AS ap "
        "USING (source_id)"
        if ap_columns
        else ""
    )
    return f"""
    SELECT {top}
        gaia.source_id,
        gaia.designation AS gaia_designation,
        gaia.ra,
        gaia.dec,
{extra_select}{ap_select}        gaia.phot_g_mean_mag AS G,
        gaia.phot_bp_mean_mag AS BP,
        gaia.phot_rp_mean_mag AS RP,
        gaia.phot_g_mean_flux AS G_flux,
        gaia.phot_g_mean_flux_error AS G_flux_error,
        gaia.phot_bp_mean_flux AS BP_flux,
        gaia.phot_bp_mean_flux_error AS BP_flux_error,
        gaia.phot_rp_mean_flux AS RP_flux,
        gaia.phot_rp_mean_flux_error AS RP_flux_error,
        tmass.j_m AS J,
        tmass.h_m AS H,
        tmass.ks_m AS Ks,
        tmass.j_msigcom AS J_error,
        tmass.h_msigcom AS H_error,
        tmass.ks_msigcom AS Ks_error,
        wise.w1mpro AS W1,
        wise.w2mpro AS W2,
        wise.w3mpro AS W3,
        wise.w4mpro AS W4,
        wise.w1mpro_error AS W1_error,
        wise.w2mpro_error AS W2_error,
        wise.w3mpro_error AS W3_error,
        wise.w4mpro_error AS W4_error,
{wise_extra_select}        gaia.parallax,
        gaia.parallax_error,
        gaia.parallax_over_error,
        gaia.pmra,
        gaia.pmdec,
{crossmatch_select}        xjoin.original_psc_source_id AS tmass_id,
        tmass.ph_qual AS tmass_quality,
        wise.designation AS wise_id,
        wise.ph_qual AS wise_quality
    FROM gaiadr3.gaia_source AS gaia
    JOIN gaiadr3.tmass_psc_xsc_best_neighbour AS xmatch USING (source_id)
    JOIN gaiadr3.tmass_psc_xsc_join AS xjoin USING (clean_tmass_psc_xsc_oid)
    JOIN gaiadr1.tmass_original_valid AS tmass
        ON xjoin.original_psc_source_id = tmass.designation
    JOIN gaiadr3.allwise_best_neighbour AS wise_match USING (source_id)
    JOIN gaiadr1.allwise_original_valid AS wise
        ON wise_match.allwise_oid = wise.allwise_oid{ap_join}
    WHERE {where}
    """


def build_quality_predicate(config: dict[str, Any]) -> str:
    phot = config["photometry"]
    tmass_patterns = _quality_like_patterns(phot["twomass_allowed_qualities"], required_bands=3)
    wise_patterns = _quality_like_patterns(phot["wise_allowed_qualities"], required_bands=2)
    return (
        "(" + " OR ".join(f"tmass.ph_qual LIKE '{pattern}%'" for pattern in tmass_patterns) + ") "
        "AND "
        "(" + " OR ".join(f"wise.ph_qual LIKE '{pattern}%'" for pattern in wise_patterns) + ")"
    )


def build_color_bounds_predicate(color_bounds: dict[str, dict[str, float]]) -> str:
    parts = []
    for color, bounds in color_bounds.items():
        expr = COLOR_EXPRESSIONS[color]
        parts.append(f"{expr} BETWEEN {bounds['min']:.12g} AND {bounds['max']:.12g}")
    return " AND ".join(parts)


def build_color_locus_predicate(fits: list[dict[str, Any]]) -> str:
    parts = []
    for fit in fits:
        x_expr = signed_log1p_adql(COLOR_EXPRESSIONS[fit["x"]])
        y_expr = signed_log1p_adql(COLOR_EXPRESSIONS[fit["y"]])
        slopes = [fit["slope_min"], fit["slope_max"]]
        intercepts = [fit["intercept_min"], fit["intercept_max"]]
        residual_checks = []
        for slope in slopes:
            for intercept in intercepts:
                residual_checks.append(
                    f"ABS({y_expr} - ({intercept:.12g} + {slope:.12g} * {x_expr})) <= {fit['threshold_max']:.12g}"
                )
        parts.append("(" + " OR ".join(residual_checks) + ")")
    return "(" + " OR ".join(parts) + ")"


def signed_log1p_adql(expr: str) -> str:
    return f"(CASE WHEN ({expr}) >= 0 THEN LOG(1 + ABS({expr})) ELSE -LOG(1 + ABS({expr})) END)"


def build_astrometry_predicate(config: dict[str, Any]) -> str:
    astrometry = config.get("astrometry", {})
    parts = []
    if astrometry.get("min_parallax") is not None:
        parts.append(f"gaia.parallax > {float(astrometry['min_parallax'])}")
    if astrometry.get("min_parallax_over_error") is not None:
        parts.append(f"gaia.parallax_over_error >= {float(astrometry['min_parallax_over_error'])}")
    return " AND ".join(parts)


def download_prediction_tile(
    tile: SkyTile,
    envelope: dict[str, Any],
    config: dict[str, Any],
    staging_dir: Path,
    row_limit: int | None,
    *,
    query_override: str | None = None,
) -> tuple[Path, str | None]:
    gaia_client = get_thread_gaia_tap_client(config)
    credentials_file = get_gaia_credentials_file(config)
    query = query_override or build_prediction_pool_adql(tile, envelope, config, row_limit=row_limit)
    output_file = staging_dir / f"{tile.tile_id}.csv"
    max_retries = max(1, int(config.get("gaia", {}).get("max_retries", 1)))
    retry_sleep_seconds = max(0, int(config.get("gaia", {}).get("retry_sleep_seconds", 0)))
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        job_id = None
        try:
            job = gaia_client.launch_job_async(
                query,
                name=f"wr_prediction_pool_{tile.tile_id}",
                dump_to_file=True,
                output_file=str(output_file),
                output_format=config.get("gaia", {}).get("output_format", "csv"),
            )
            job_id = getattr(job, "jobid", None)
            if not output_file.exists():
                results = job.get_results()
                results.to_pandas().to_csv(output_file, index=False)
            if credentials_file and config.get("gaia", {}).get("cleanup_async_jobs", True) and job_id:
                gaia_client.remove_jobs([job_id])
            return output_file, job_id
        except Exception as exc:
            last_error = exc
            if credentials_file and config.get("gaia", {}).get("cleanup_async_jobs", True) and job_id:
                try:
                    gaia_client.remove_jobs([job_id])
                except Exception:
                    pass
            if attempt < max_retries and retry_sleep_seconds:
                sleep(retry_sleep_seconds)

    if last_error:
        raise last_error
    raise RuntimeError(f"Gaia query failed for tile {tile.tile_id}")


def ingest_downloaded_prediction_tile(
    db_path: Path,
    csv_path: Path,
    tile: SkyTile,
    envelope: dict[str, Any],
    config: dict[str, Any],
    job_id: str | None,
) -> tuple[Path, int, int]:
    parquet_path, inserted, excluded = ingest_prediction_tile(db_path, csv_path, tile, envelope, config)
    cleanup_staging_file(csv_path, config)
    mark_tile_completed(db_path, tile, parquet_path, inserted=inserted, excluded=excluded, job_id=job_id)
    return parquet_path, inserted, excluded


def login_to_gaia(config: dict[str, Any]) -> None:
    credentials_file = get_gaia_credentials_file(config)
    if not credentials_file:
        return

    from astroquery.gaia import Gaia
    from astroquery.utils.tap.core import TapPlus

    if config.get("gaia", {}).get("login_data_server", False):
        Gaia.login(credentials_file=str(credentials_file))
    else:
        TapPlus.login(Gaia, credentials_file=str(credentials_file))


def make_gaia_tap_client(config: dict[str, Any]):
    credentials_file = get_gaia_credentials_file(config)
    if not credentials_file:
        from astroquery.gaia import Gaia

        return Gaia

    from astroquery.utils.tap.core import TapPlus

    client = TapPlus(url="https://gea.esac.esa.int/tap-server/tap")
    TapPlus.login(client, credentials_file=str(credentials_file))
    return client


def get_thread_gaia_tap_client(config: dict[str, Any]):
    client = getattr(_THREAD_LOCAL, "gaia_tap_client", None)
    if client is None:
        client = make_gaia_tap_client(config)
        _THREAD_LOCAL.gaia_tap_client = client
    return client


def get_gaia_credentials_file(config: dict[str, Any]) -> Path | None:
    env_key = config.get("gaia", {}).get("credentials_file_env", "GAIA_CREDENTIALS_FILE")
    raw = config.get("_env", {}).get(env_key)
    if not raw:
        return None
    path = resolve_path(raw)
    return path if path.exists() else None


def initialize_prediction_pool_database(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_pool_parquet_files (
            tile_id VARCHAR PRIMARY KEY,
            parquet_path VARCHAR,
            row_count INTEGER,
            file_size_bytes BIGINT,
            written_at TIMESTAMP
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_pool_tiles (
            tile_id VARCHAR PRIMARY KEY,
            ra_min DOUBLE,
            ra_max DOUBLE,
            dec_min DOUBLE,
            dec_max DOUBLE,
            status VARCHAR,
            row_count INTEGER,
            excluded_count INTEGER,
            output_file VARCHAR,
            job_id VARCHAR,
            error_message VARCHAR,
            updated_at TIMESTAMP
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_pool_query_log (
            tile_id VARCHAR,
            job_id VARCHAR,
            output_file VARCHAR,
            row_count INTEGER,
            excluded_count INTEGER,
            status VARCHAR,
            message VARCHAR,
            logged_at TIMESTAMP
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_pool_exclusions (
            source_id BIGINT,
            tile_id VARCHAR,
            exclusion_label VARCHAR,
            excluded_at TIMESTAMP
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_pool_known_sources (
            source_id BIGINT PRIMARY KEY,
            exclusion_label VARCHAR
        )
        """
    )


def refresh_prediction_pool_view(con: duckdb.DuckDBPyConnection, output_dir: Path) -> None:
    drop_relation_if_exists(con, "prediction_pool_sources")
    files = sorted(output_dir.glob("*.parquet")) if output_dir.exists() else []
    if files:
        glob_path = (output_dir / "*.parquet").as_posix()
        con.execute(
            f"""
            CREATE VIEW prediction_pool_sources AS
            SELECT *
            FROM read_parquet('{glob_path}', union_by_name=true)
            """
        )
    refresh_prediction_pool_metadata_views(con)


def refresh_prediction_pool_metadata_views(con: duckdb.DuckDBPyConnection) -> None:
    drop_relation_if_exists(con, "prediction_pool_effective_tiles")
    drop_relation_if_exists(con, "prediction_pool_tile_coverage")
    con.execute(
        """
        CREATE VIEW prediction_pool_effective_tiles AS
        SELECT
            t.tile_id,
            t.ra_min,
            t.ra_max,
            t.dec_min,
            t.dec_max,
            t.status,
            t.row_count,
            t.excluded_count,
            p.parquet_path,
            p.file_size_bytes,
            p.written_at
        FROM prediction_pool_tiles t
        INNER JOIN prediction_pool_parquet_files p USING (tile_id)
        WHERE t.status = 'completed'
        ORDER BY t.ra_min, t.dec_min
        """
    )
    con.execute(
        """
        CREATE VIEW prediction_pool_tile_coverage AS
        SELECT
            t.tile_id,
            t.ra_min,
            t.ra_max,
            t.dec_min,
            t.dec_max,
            t.status,
            t.row_count,
            t.excluded_count,
            t.output_file,
            t.error_message,
            CASE
                WHEN t.status = 'completed' THEN true
                WHEN t.status = 'skipped' AND t.error_message = 'subdivided_into_subtiles' THEN true
                ELSE false
            END AS coverage_terminal,
            CASE
                WHEN t.status = 'completed' THEN 'effective_tile'
                WHEN t.status = 'skipped' AND t.error_message = 'subdivided_into_subtiles' THEN 'subdivided_parent'
                ELSE 'incomplete'
            END AS coverage_role
        FROM prediction_pool_tiles t
        ORDER BY t.ra_min, t.dec_min
        """
    )


def drop_relation_if_exists(con: duckdb.DuckDBPyConnection, name: str) -> None:
    row = con.execute(
        """
        SELECT table_type
        FROM information_schema.tables
        WHERE table_name = ?
        """,
        [name],
    ).fetchone()
    if not row:
        return
    if str(row[0]).upper() == "VIEW":
        con.execute(f"DROP VIEW {name}")
    else:
        con.execute(f"DROP TABLE {name}")


def refresh_known_sources(con: duckdb.DuckDBPyConnection, config: dict[str, Any]) -> None:
    con.execute("DELETE FROM prediction_pool_known_sources")
    paths = config["paths"]
    reference_db = resolve_path(paths["wr_reference_db"])
    simbad_db = resolve_path(paths["simbad_negative_db"])
    if reference_db.exists():
        con.execute(f"ATTACH '{reference_db.as_posix()}' AS wr_ref (READ_ONLY)")
        con.execute(
            """
            INSERT OR REPLACE INTO prediction_pool_known_sources
            SELECT DISTINCT source_id, 'known_wr'
            FROM wr_ref.wr_reference
            WHERE source_id IS NOT NULL
            """
        )
        con.execute("DETACH wr_ref")
    if simbad_db.exists():
        con.execute(f"ATTACH '{simbad_db.as_posix()}' AS simbad_neg (READ_ONLY)")
        con.execute(
            """
            INSERT OR REPLACE INTO prediction_pool_known_sources
            SELECT DISTINCT source_id, 'known_non_wr_simbad'
            FROM simbad_neg.simbad_negative_sources
            WHERE source_id IS NOT NULL
            """
        )
        con.execute("DETACH simbad_neg")


def ensure_tiles(con: duckdb.DuckDBPyConnection, tiles: list[SkyTile]) -> None:
    now = now_utc()
    rows = [
        {
            "tile_id": tile.tile_id,
            "ra_min": tile.ra_min,
            "ra_max": tile.ra_max,
            "dec_min": tile.dec_min,
            "dec_max": tile.dec_max,
            "status": "pending",
            "row_count": 0,
            "excluded_count": 0,
            "output_file": None,
            "job_id": None,
            "error_message": None,
            "updated_at": now,
        }
        for tile in tiles
    ]
    con.register("_tiles", pd.DataFrame(rows))
    con.execute(
        """
        INSERT INTO prediction_pool_tiles
        SELECT * FROM _tiles
        ON CONFLICT (tile_id) DO NOTHING
        """
    )
    con.unregister("_tiles")


def reset_stale_running_tiles(con: duckdb.DuckDBPyConnection, config: dict[str, Any]) -> int:
    stale_minutes = int(config.get("tiling", {}).get("stale_running_minutes", 0) or 0)
    if stale_minutes <= 0:
        return 0
    cutoff = now_utc() - timedelta(minutes=stale_minutes)
    rows = con.execute(
        """
        UPDATE prediction_pool_tiles
        SET status = 'failed',
            error_message = 'stale_running_reset',
            updated_at = ?
        WHERE status = 'running'
          AND updated_at < ?
        RETURNING tile_id
        """,
        [now_utc(), cutoff],
    ).fetchall()
    return len(rows)


def subdivide_failed_tiles(con: duckdb.DuckDBPyConnection, config: dict[str, Any]) -> int:
    cfg = config.get("tiling", {}).get("subdivide_failed", {})
    if not cfg.get("enabled", False):
        return 0

    sub_ra_step = float(cfg.get("ra_step_deg", 5.0))
    sub_dec_step = float(cfg.get("dec_step_deg", 5.0))
    failed_rows = con.execute(
        """
        SELECT tile_id, ra_min, ra_max, dec_min, dec_max
        FROM prediction_pool_tiles
        WHERE status = 'failed'
          AND (ra_max - ra_min) > ?
          AND (dec_max - dec_min) > ?
        ORDER BY dec_min, ra_min
        """,
        [sub_ra_step, sub_dec_step],
    ).fetchall()
    if not failed_rows:
        return 0

    now = now_utc()
    subtile_rows = []
    parent_ids = []
    for parent_id, ra_min, ra_max, dec_min, dec_max in failed_rows:
        parent_ids.append(str(parent_id))
        ra = float(ra_min)
        while ra < float(ra_max):
            child_ra_max = min(float(ra_max), ra + sub_ra_step)
            dec = float(dec_min)
            while dec < float(dec_max):
                child_dec_max = min(float(dec_max), dec + sub_dec_step)
                tile_id = format_tile_id(ra, child_ra_max, dec, child_dec_max)
                subtile_rows.append(
                    {
                        "tile_id": tile_id,
                        "ra_min": ra,
                        "ra_max": child_ra_max,
                        "dec_min": dec,
                        "dec_max": child_dec_max,
                        "status": "pending",
                        "row_count": 0,
                        "excluded_count": 0,
                        "output_file": None,
                        "job_id": None,
                        "error_message": f"subtile_of:{parent_id}",
                        "updated_at": now,
                    }
                )
                dec = child_dec_max
            ra = child_ra_max

    if subtile_rows:
        con.register("_subtiles", pd.DataFrame(subtile_rows))
        con.execute(
            """
            INSERT INTO prediction_pool_tiles
            SELECT * FROM _subtiles
            ON CONFLICT (tile_id) DO NOTHING
            """
        )
        con.unregister("_subtiles")

    con.executemany(
        """
        UPDATE prediction_pool_tiles
        SET status = 'skipped',
            error_message = 'subdivided_into_subtiles',
            updated_at = ?
        WHERE tile_id = ?
        """,
        [(now, parent_id) for parent_id in parent_ids],
    )
    return len(subtile_rows)


def ingest_prediction_tile(
    db_path: str | Path,
    csv_path: Path,
    tile: SkyTile,
    envelope: dict[str, Any],
    config: dict[str, Any],
) -> tuple[Path, int, int]:
    output_dir = resolve_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / f"{tile.tile_id}.parquet"
    compression = config.get("parquet", {}).get("compression", "zstd")
    with _DB_WRITE_LOCK:
        con = connect(db_path)
        try:
            initialize_prediction_pool_database(con)
            con.execute("DROP TABLE IF EXISTS _tile_raw")
            con.execute(
                """
                CREATE TEMP TABLE _tile_raw AS
                SELECT * FROM read_csv_auto(?, header=true, union_by_name=true)
                """,
                [str(csv_path)],
            )
            add_local_annotation_table(con, tile, envelope)
            con.execute(
                """
                INSERT INTO prediction_pool_exclusions
                SELECT s.source_id, s.tile_id, k.exclusion_label, ?
                FROM _tile_stage s
                JOIN prediction_pool_known_sources k USING (source_id)
                """,
                [now_utc()],
            )
            excluded = int(con.execute("SELECT COUNT(*) FROM _tile_stage s JOIN prediction_pool_known_sources k USING (source_id)").fetchone()[0])
            final_columns = prediction_pool_output_columns()
            con.execute(
                f"""
                COPY (
                    SELECT {", ".join("s." + col for col in final_columns)}
                    FROM _tile_stage s
                    LEFT JOIN prediction_pool_known_sources k USING (source_id)
                    WHERE k.source_id IS NULL
                )
                TO ?
                (FORMAT PARQUET, COMPRESSION {compression.upper()})
                """,
                [str(parquet_path)],
            )
            inserted = int(con.execute("SELECT COUNT(*) FROM _tile_stage s LEFT JOIN prediction_pool_known_sources k USING (source_id) WHERE k.source_id IS NULL").fetchone()[0])
            con.execute(
                """
                INSERT OR REPLACE INTO prediction_pool_parquet_files
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    tile.tile_id,
                    str(parquet_path),
                    inserted,
                    parquet_path.stat().st_size if parquet_path.exists() else 0,
                    now_utc(),
                ],
            )
            refresh_prediction_pool_view(con, output_dir)
            return parquet_path, inserted, excluded
        finally:
            con.close()


def cleanup_staging_file(csv_path: Path, config: dict[str, Any]) -> None:
    if config.get("gaia", {}).get("cleanup_staging_after_ingest", True) and csv_path.exists():
        csv_path.unlink()


def prediction_pool_output_columns() -> list[str]:
    return [
        "source_id",
        "tile_id",
        "gaia_designation",
        "ra",
        "dec",
        "G",
        "BP",
        "RP",
        "G_flux",
        "G_flux_error",
        "BP_flux",
        "BP_flux_error",
        "RP_flux",
        "RP_flux_error",
        "J",
        "H",
        "Ks",
        "J_error",
        "H_error",
        "Ks_error",
        "W1",
        "W2",
        "W3",
        "W4",
        "W1_error",
        "W2_error",
        "W3_error",
        "W4_error",
        "parallax",
        "parallax_error",
        "parallax_over_error",
        "pmra",
        "pmdec",
        "tmass_id",
        "tmass_quality",
        "wise_id",
        "wise_quality",
        "G_BP",
        "G_RP",
        "BP_RP",
        "J_H",
        "J_K",
        "H_K",
        "W1_W2",
        "color_locus_valid",
        "color_locus_outlier_plane_count",
        "color_locus_keep",
        "inserted_at",
    ]


def add_local_annotation_table(con: duckdb.DuckDBPyConnection, tile: SkyTile, envelope: dict[str, Any]) -> None:
    select_cols = ", ".join([f"r.{col}" for col in BASE_COLUMNS if col != "source_id"])
    color_cols = ", ".join([f"({expr}) AS {name}" for name, expr in LOCAL_COLOR_EXPRESSIONS.items()])
    valid_checks = [f"{name} IS NOT NULL" for name in LOCAL_COLOR_EXPRESSIONS]
    outlier_checks = []
    for fit in envelope["fits"]:
        x_expr = signed_log1p_sql(fit["x"])
        y_expr = signed_log1p_sql(fit["y"])
        slope = (float(fit["slope_min"]) + float(fit["slope_max"])) / 2
        intercept = (float(fit["intercept_min"]) + float(fit["intercept_max"])) / 2
        threshold = float(fit["threshold_max"])
        outlier_checks.append(
            f"CASE WHEN ABS({y_expr} - ({intercept:.12g} + {slope:.12g} * {x_expr})) > {threshold:.12g} THEN 1 ELSE 0 END"
        )
    outlier_sum = " + ".join(outlier_checks) if outlier_checks else "0"
    con.execute("DROP TABLE IF EXISTS _tile_stage")
    con.execute(
        f"""
        CREATE TEMP TABLE _tile_stage AS
        WITH colors AS (
            SELECT
                r.source_id,
                CAST(? AS VARCHAR) AS tile_id,
                {select_cols},
                {color_cols}
            FROM _tile_raw r
        ),
        annotated AS (
            SELECT
                *,
                ({' AND '.join(valid_checks)}) AS color_locus_valid,
                CAST(({outlier_sum}) AS INTEGER) AS color_locus_outlier_plane_count
            FROM colors
        )
        SELECT
            *,
            color_locus_valid AND color_locus_outlier_plane_count < 2 AS color_locus_keep,
            CAST(? AS TIMESTAMP) AS inserted_at
        FROM annotated
        WHERE color_locus_valid AND color_locus_outlier_plane_count < 2
        """,
        [tile.tile_id, now_utc()],
    )


def signed_log1p_sql(color: str) -> str:
    return f"(CASE WHEN {color} >= 0 THEN LN(1 + ABS({color})) ELSE -LN(1 + ABS({color})) END)"


def get_tile_status(db_path: Path, tile_id: str) -> str | None:
    con = connect(db_path)
    try:
        initialize_prediction_pool_database(con)
        row = con.execute("SELECT status FROM prediction_pool_tiles WHERE tile_id = ?", [tile_id]).fetchone()
        return str(row[0]) if row else None
    finally:
        con.close()


def mark_tile_running(db_path: Path, tile: SkyTile) -> None:
    update_tile_status(db_path, tile.tile_id, "running")


def mark_tile_completed(
    db_path: Path,
    tile: SkyTile,
    output_file: Path,
    *,
    inserted: int,
    excluded: int,
    job_id: str | None,
) -> None:
    with _DB_WRITE_LOCK:
        con = connect(db_path)
        try:
            initialize_prediction_pool_database(con)
            con.execute(
                """
                INSERT INTO prediction_pool_tiles
                VALUES (?, ?, ?, ?, ?, 'completed', ?, ?, ?, ?, NULL, ?)
                ON CONFLICT (tile_id) DO UPDATE SET
                    status = 'completed',
                    row_count = excluded.row_count,
                    excluded_count = excluded.excluded_count,
                    output_file = excluded.output_file,
                    job_id = excluded.job_id,
                    error_message = NULL,
                    updated_at = excluded.updated_at
                """,
                [
                    tile.tile_id,
                    tile.ra_min,
                    tile.ra_max,
                    tile.dec_min,
                    tile.dec_max,
                    inserted,
                    excluded,
                    str(output_file),
                    job_id,
                    now_utc(),
                ],
            )
            con.execute(
                """
                INSERT INTO prediction_pool_query_log
                VALUES (?, ?, ?, ?, ?, 'completed', NULL, ?)
                """,
                [tile.tile_id, job_id, str(output_file), inserted, excluded, now_utc()],
            )
            refresh_prediction_pool_metadata_views(con)
        finally:
            con.close()


def mark_tile_failed(db_path: Path, tile: SkyTile, message: str) -> None:
    with _DB_WRITE_LOCK:
        con = connect(db_path)
        try:
            con.execute(
                """
                UPDATE prediction_pool_tiles
                SET status = 'failed', error_message = ?, updated_at = ?
                WHERE tile_id = ?
                """,
                [message, now_utc(), tile.tile_id],
            )
            con.execute(
                """
                INSERT INTO prediction_pool_query_log
                VALUES (?, NULL, NULL, 0, 0, 'failed', ?, ?)
                """,
                [tile.tile_id, message, now_utc()],
            )
        finally:
            con.close()


def update_tile_status(db_path: Path, tile_id: str, status: str) -> None:
    with _DB_WRITE_LOCK:
        con = connect(db_path)
        try:
            con.execute(
                "UPDATE prediction_pool_tiles SET status = ?, updated_at = ? WHERE tile_id = ?",
                [status, now_utc(), tile_id],
            )
        finally:
            con.close()


def mark_remaining_skipped(db_path: Path, tiles: list[SkyTile], *, reason: str) -> int:
    if not tiles:
        return 0
    with _DB_WRITE_LOCK:
        con = connect(db_path)
        try:
            rows = [(tile.tile_id,) for tile in tiles]
            con.executemany(
                """
                UPDATE prediction_pool_tiles
                SET status = 'skipped', error_message = ?, updated_at = ?
                WHERE tile_id = ? AND status IN ('pending', 'running')
                """,
                [(reason, now_utc(), tile_id) for (tile_id,) in rows],
            )
            return len(rows)
        finally:
            con.close()


def local_storage_bytes(db_path: Path, staging_dir: Path, output_dir: Path | None = None) -> int:
    total = 0
    for path in [db_path, Path(f"{db_path}.wal")]:
        if path.exists():
            total += path.stat().st_size
    for directory in [staging_dir, output_dir]:
        if directory and directory.exists():
            total += sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
    return total


def now_utc() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _quoted_values(values: list[str]) -> str:
    return ", ".join(f"'{value}'" for value in values)


def _quality_like_patterns(values: list[str], *, required_bands: int) -> list[str]:
    patterns = [""]
    for _ in range(required_bands):
        patterns = [prefix + value for prefix in patterns for value in values]
    return patterns
