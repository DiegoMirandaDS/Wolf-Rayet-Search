from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


TABLES = [
    "catalog_snapshots",
    "wr_catalog_raw",
    "wr_reference",
    "gaia_sources",
    "twomass_matches",
    "wise_matches",
    "crossmatch_log",
]

SIMBAD_NEGATIVE_TABLES = [
    "simbad_negative_raw",
    "simbad_negative_sources",
    "gaia_sources",
    "twomass_matches",
    "wise_matches",
    "crossmatch_log",
]


def connect(db_path: str | Path) -> duckdb.DuckDBPyConnection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path))


def replace_table(con: duckdb.DuckDBPyConnection, name: str, df: pd.DataFrame) -> None:
    con.execute(f"DROP TABLE IF EXISTS {name}")
    if len(df.columns) == 0:
        con.execute(f"CREATE TABLE {name} (_empty_placeholder BOOLEAN)")
        con.execute(f"DELETE FROM {name}")
        return
    con.register("_df", df)
    con.execute(f"CREATE TABLE {name} AS SELECT * FROM _df")
    con.unregister("_df")


def write_reference_database(
    db_path: str | Path,
    *,
    snapshot: dict[str, Any],
    wr_catalog_raw: pd.DataFrame,
    wr_reference: pd.DataFrame,
    gaia_sources: pd.DataFrame,
    twomass_matches: pd.DataFrame,
    wise_matches: pd.DataFrame,
    crossmatch_log: pd.DataFrame,
) -> None:
    con = connect(db_path)
    try:
        replace_table(con, "catalog_snapshots", pd.DataFrame([snapshot]))
        replace_table(con, "wr_catalog_raw", wr_catalog_raw)
        replace_table(con, "wr_reference", wr_reference)
        replace_table(con, "gaia_sources", gaia_sources)
        replace_table(con, "twomass_matches", twomass_matches)
        replace_table(con, "wise_matches", wise_matches)
        replace_table(con, "crossmatch_log", crossmatch_log)
    finally:
        con.close()


def write_simbad_negative_database(
    db_path: str | Path,
    *,
    simbad_negative_raw: pd.DataFrame,
    simbad_negative_sources: pd.DataFrame,
    gaia_sources: pd.DataFrame,
    twomass_matches: pd.DataFrame,
    wise_matches: pd.DataFrame,
    crossmatch_log: pd.DataFrame,
) -> None:
    con = connect(db_path)
    try:
        replace_table(con, "simbad_negative_raw", simbad_negative_raw)
        replace_table(con, "simbad_negative_sources", simbad_negative_sources)
        replace_table(con, "gaia_sources", gaia_sources)
        replace_table(con, "twomass_matches", twomass_matches)
        replace_table(con, "wise_matches", wise_matches)
        replace_table(con, "crossmatch_log", crossmatch_log)
    finally:
        con.close()


def table_count(con: duckdb.DuckDBPyConnection, table: str) -> int:
    return int(con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def audit_reference_database(db_path: str | Path) -> dict[str, Any]:
    con = connect(db_path)
    try:
        counts = {table: table_count(con, table) for table in TABLES if _table_exists(con, table)}
        duplicates = 0
        if _table_exists(con, "wr_reference"):
            duplicates = int(
                con.execute(
                    """
                    SELECT COUNT(*)
                    FROM (
                        SELECT source_id, COUNT(*) AS n
                        FROM wr_reference
                        GROUP BY source_id
                        HAVING n > 1
                    )
                    """
                ).fetchone()[0]
            )
        return {"counts": counts, "duplicate_source_ids": duplicates}
    finally:
        con.close()


def _table_exists(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    return bool(
        con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?", [table]
        ).fetchone()[0]
    )
