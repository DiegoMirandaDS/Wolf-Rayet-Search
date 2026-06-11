from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pandas as pd

from wr_detector.catalogs.gaia import (
    normalize_gaia_result,
    normalize_gaia_tmass_result,
    normalize_gaia_wise_result,
    query_gaia_reference,
    query_gaia_tmass,
    query_gaia_wise,
)
from wr_detector.catalogs.simbad import (
    deduplicate_simbad_sources,
    exclude_wolf_rayet_rows,
    normalize_simbad_result,
    query_simbad_by_types,
)
from wr_detector.config import load_simbad_negative_config, resolve_path
from wr_detector.db import write_simbad_negative_database
from wr_detector.features import export_simbad_negative_datasets
from wr_detector.pipelines.reference import _complete_twomass_matches, _complete_wise_matches


def build_simbad_negative(config_path: str | Path) -> dict[str, object]:
    config = load_simbad_negative_config(config_path)
    paths = config["paths"]
    catalogs = config["catalogs"]
    simbad_cfg = config["simbad"]
    crossmatch = config["crossmatch"]

    raw_dir = resolve_path(paths["raw_simbad_negative_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    db_path = resolve_path(paths["simbad_negative_db"])

    raw = query_simbad_by_types(
        simbad_cfg["object_types"],
        row_limit=simbad_cfg.get("row_limit"),
        timeout_seconds=int(catalogs["simbad"]["timeout_seconds"]),
    )
    raw_path = raw_dir / f"simbad_negative_{_timestamp()}.csv"
    raw.to_csv(raw_path, index=False)

    sources = normalize_simbad_result(raw)
    wr_source_ids = _load_wr_source_ids(resolve_path(paths["wr_reference_db"]))
    sources = exclude_wolf_rayet_rows(
        sources,
        wr_source_ids=wr_source_ids,
        patterns=simbad_cfg.get("exclude_wolf_rayet_patterns"),
    )
    sources = deduplicate_simbad_sources(sources)
    source_ids = sources["source_id"].astype("int64").tolist()

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
        {"stage": "simbad_raw", "method": "simbad_otype", "rows": len(raw), "notes": ",".join(simbad_cfg["object_types"])},
        {"stage": "simbad_negative_sources", "method": "explicit_gaia_dr3_ids", "rows": len(sources), "notes": str(raw_path)},
        {"stage": "gaia_sources", "method": "gaia_source_id", "rows": len(gaia_sources), "notes": ""},
        {"stage": "twomass_matches", "method": "gaia_xmatch", "rows": len(twomass_gaia), "notes": ""},
        {"stage": "wise_matches", "method": "gaia_xmatch", "rows": len(wise_gaia), "notes": ""},
        *twomass_log,
        *wise_log,
    ]
    crossmatch_log = pd.DataFrame(log_rows)

    write_simbad_negative_database(
        db_path,
        simbad_negative_raw=raw,
        simbad_negative_sources=sources,
        gaia_sources=gaia_sources,
        twomass_matches=twomass_all,
        wise_matches=wise_all,
        crossmatch_log=crossmatch_log,
    )

    export_counts = export_simbad_negative_datasets(config["filters_config"])
    return {
        "db_path": str(db_path),
        "raw_path": str(raw_path),
        "simbad_raw_rows": len(raw),
        "simbad_with_gaia_id_non_wr": len(sources),
        "gaia_sources": len(gaia_sources),
        "twomass_matches": len(twomass_all),
        "wise_matches": len(wise_all),
        "exports": export_counts,
    }


def _load_wr_source_ids(db_path: Path) -> list[int]:
    if not db_path.exists():
        return []
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        exists = bool(
            con.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'wr_reference'").fetchone()[0]
        )
        if not exists:
            return []
        return [int(row[0]) for row in con.execute("SELECT source_id FROM wr_reference").fetchall()]
    finally:
        con.close()


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
