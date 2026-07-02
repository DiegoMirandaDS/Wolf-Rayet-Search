"""Reference WR catalogue build pipeline with Gaia, 2MASS, WISE, and VizieR enrichment."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from wr_detector.catalogs.gaia import (
    normalize_gaia_result,
    normalize_gaia_tmass_result,
    normalize_gaia_wise_result,
    query_gaia_reference,
    query_gaia_tmass,
    query_gaia_wise,
)
from wr_detector.catalogs.gwrc import extract_gaia_dr3_reference, fetch_gwrc_snapshot
from wr_detector.catalogs.vizier import (
    TWOMASS_COLUMNS,
    WISE_COLUMNS,
    normalize_twomass_vizier,
    normalize_wise_vizier,
    query_catalog_by_coordinates,
    select_duplicates_by_radius,
)
from wr_detector.config import load_reference_config, resolve_path
from wr_detector.db import write_reference_database
from wr_detector.features import export_reference_datasets


def build_reference(config_path: str | Path) -> dict[str, object]:
    config = load_reference_config(config_path)
    paths = config["paths"]
    catalogs = config["catalogs"]
    crossmatch = config["crossmatch"]

    raw_dir = resolve_path(paths["raw_reference_dir"])
    db_path = resolve_path(paths["wr_reference_db"])

    snapshot, raw_catalog = fetch_gwrc_snapshot(catalogs["gwrc"]["url"], raw_dir)
    wr_reference = extract_gaia_dr3_reference(raw_catalog)
    source_ids = wr_reference["source_id"].astype("int64").tolist()

    chunk_size = int(catalogs["gaia"]["chunk_size"])
    gaia_raw = query_gaia_reference(source_ids, chunk_size=chunk_size)
    tmass_raw = query_gaia_tmass(source_ids, chunk_size=chunk_size)
    wise_raw = query_gaia_wise(source_ids, chunk_size=chunk_size)
    gaia_sources = normalize_gaia_result(gaia_raw)
    twomass_gaia = normalize_gaia_tmass_result(tmass_raw)
    wise_gaia = normalize_gaia_wise_result(wise_raw)

    twomass_all, twomass_log = _complete_twomass_matches(
        gaia_sources=gaia_sources,
        existing=twomass_gaia,
        catalog=catalogs["vizier"]["twomass_catalog"],
        radius_arcsec=float(crossmatch["vizier_radius_arcsec"]),
        primary_radius_arcsec=float(crossmatch["duplicate_primary_radius_arcsec"]),
        secondary_radius_arcsec=float(crossmatch["duplicate_secondary_radius_arcsec"]),
        timeout_seconds=int(catalogs["vizier"]["timeout_seconds"]),
    )
    wise_all, wise_log = _complete_wise_matches(
        gaia_sources=gaia_sources,
        existing=wise_gaia,
        catalog=catalogs["vizier"]["wise_catalog"],
        radius_arcsec=float(crossmatch["vizier_radius_arcsec"]),
        primary_radius_arcsec=float(crossmatch["duplicate_primary_radius_arcsec"]),
        secondary_radius_arcsec=float(crossmatch["duplicate_secondary_radius_arcsec"]),
        timeout_seconds=int(catalogs["vizier"]["timeout_seconds"]),
    )

    log_rows = [
        {"stage": "gwrc_snapshot", "method": "http_read_html", "rows": len(raw_catalog), "notes": snapshot.version},
        {"stage": "wr_reference", "method": "alias1_gaia_dr3_filter", "rows": len(wr_reference), "notes": ""},
        {"stage": "gaia_sources", "method": "gaia_source_id", "rows": len(gaia_sources), "notes": ""},
        {"stage": "twomass_matches", "method": "gaia_xmatch", "rows": len(twomass_gaia), "notes": ""},
        {"stage": "wise_matches", "method": "gaia_xmatch", "rows": len(wise_gaia), "notes": ""},
        *twomass_log,
        *wise_log,
    ]
    crossmatch_log = pd.DataFrame(log_rows)

    snapshot_dict = asdict(snapshot)
    snapshot_dict["html_path"] = str(snapshot.html_path)
    snapshot_dict["csv_path"] = str(snapshot.csv_path)
    write_reference_database(
        db_path,
        snapshot=snapshot_dict,
        wr_catalog_raw=raw_catalog,
        wr_reference=wr_reference,
        gaia_sources=gaia_sources,
        twomass_matches=twomass_all,
        wise_matches=wise_all,
        crossmatch_log=crossmatch_log,
    )

    export_counts = export_reference_datasets(config["filters_config"])
    return {
        "db_path": str(db_path),
        "snapshot_version": snapshot.version,
        "snapshot_rows": snapshot.row_count,
        "wr_with_gaia_id": len(wr_reference),
        "gaia_sources": len(gaia_sources),
        "twomass_matches": len(twomass_all),
        "wise_matches": len(wise_all),
        "exports": export_counts,
    }


def _complete_twomass_matches(
    *,
    gaia_sources: pd.DataFrame,
    existing: pd.DataFrame,
    catalog: str,
    radius_arcsec: float,
    primary_radius_arcsec: float,
    secondary_radius_arcsec: float,
    timeout_seconds: int,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    if gaia_sources.empty or "source_id" not in gaia_sources.columns:
        return existing.copy(), [
            {"stage": "twomass_fallback_raw", "method": "vizier_cone", "rows": 0, "notes": "missing=0"},
            {"stage": "twomass_fallback_selected", "method": "vizier_cone", "rows": 0, "notes": "{}"},
        ]
    existing_ids = existing["source_id"] if "source_id" in existing.columns else pd.Series(dtype="int64")
    missing = gaia_sources[~gaia_sources["source_id"].isin(existing_ids)]
    raw = query_catalog_by_coordinates(
        missing,
        catalog=catalog,
        columns=TWOMASS_COLUMNS,
        radius_arcsec=radius_arcsec,
        timeout_seconds=timeout_seconds,
    )
    selected, stats = select_duplicates_by_radius(
        raw,
        primary_radius_arcsec=primary_radius_arcsec,
        secondary_radius_arcsec=secondary_radius_arcsec,
    )
    fallback = normalize_twomass_vizier(selected)
    out = pd.concat([existing, fallback], ignore_index=True) if not fallback.empty else existing.copy()
    if "source_id" not in out.columns:
        return out, log
    out = out.drop_duplicates("source_id", keep="first").reset_index(drop=True)
    log = [
        {"stage": "twomass_fallback_raw", "method": "vizier_cone", "rows": len(raw), "notes": f"missing={len(missing)}"},
        {"stage": "twomass_fallback_selected", "method": "vizier_cone", "rows": len(fallback), "notes": str(stats)},
    ]
    return out, log


def _complete_wise_matches(
    *,
    gaia_sources: pd.DataFrame,
    existing: pd.DataFrame,
    catalog: str,
    radius_arcsec: float,
    primary_radius_arcsec: float,
    secondary_radius_arcsec: float,
    timeout_seconds: int,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    if gaia_sources.empty or "source_id" not in gaia_sources.columns:
        return existing.copy(), [
            {"stage": "wise_fallback_raw", "method": "vizier_cone", "rows": 0, "notes": "missing=0"},
            {"stage": "wise_fallback_selected", "method": "vizier_cone", "rows": 0, "notes": "{}"},
        ]
    existing_ids = existing["source_id"] if "source_id" in existing.columns else pd.Series(dtype="int64")
    missing = gaia_sources[~gaia_sources["source_id"].isin(existing_ids)]
    raw = query_catalog_by_coordinates(
        missing,
        catalog=catalog,
        columns=WISE_COLUMNS,
        radius_arcsec=radius_arcsec,
        timeout_seconds=timeout_seconds,
    )
    selected, stats = select_duplicates_by_radius(
        raw,
        primary_radius_arcsec=primary_radius_arcsec,
        secondary_radius_arcsec=secondary_radius_arcsec,
    )
    fallback = normalize_wise_vizier(selected)
    out = pd.concat([existing, fallback], ignore_index=True) if not fallback.empty else existing.copy()
    if "source_id" not in out.columns:
        return out, log
    out = out.drop_duplicates("source_id", keep="first").reset_index(drop=True)
    log = [
        {"stage": "wise_fallback_raw", "method": "vizier_cone", "rows": len(raw), "notes": f"missing={len(missing)}"},
        {"stage": "wise_fallback_selected", "method": "vizier_cone", "rows": len(fallback), "notes": str(stats)},
    ]
    return out, log
