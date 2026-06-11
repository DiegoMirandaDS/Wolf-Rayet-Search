from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd


GAIA_DR3_ID_RE = re.compile(r"\bGaia\s+DR3\s+(\d+)\b", re.IGNORECASE)
WOLF_RAYET_RE = re.compile(r"wolf\s*-?\s*rayet|wolfrayet|\bwr\*", re.IGNORECASE)


def build_simbad_otype_criteria(object_type: str) -> str:
    escaped = object_type.replace("'", "''")
    return f"otype = '{escaped}'"


def build_simbad_otype_tap_query(object_type: str, *, row_limit: int | None = None) -> str:
    escaped = object_type.replace("'", "''")
    top = "" if row_limit is None else f" TOP {int(row_limit)}"
    return f"""
    SELECT DISTINCT{top}
        basic.main_id AS simbad_main_id,
        basic.ra AS simbad_ra,
        basic.dec AS simbad_dec,
        basic.otype AS simbad_main_type,
        basic.sp_type AS simbad_sp_type,
        ident.id AS simbad_ids,
        '{escaped}' AS simbad_query_type
    FROM basic
    JOIN ident ON basic.oid = ident.oidref
    WHERE basic.otype = '{escaped}'
      AND ident.id LIKE 'Gaia DR3 %'
    """


def query_simbad_by_types(
    object_types: Iterable[str],
    *,
    row_limit: int | None = None,
    timeout_seconds: int = 180,
) -> pd.DataFrame:
    from astroquery.simbad import Simbad

    frames: list[pd.DataFrame] = []
    client = Simbad()
    client.TIMEOUT = timeout_seconds
    for object_type in object_types:
        result = client.query_tap(build_simbad_otype_tap_query(object_type, row_limit=row_limit))
        if result is None:
            continue
        frame = result.to_pandas()
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def normalize_simbad_result(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "source_id",
                "gaia_designation",
                "simbad_main_id",
                "simbad_ra",
                "simbad_dec",
                "simbad_main_type",
                "simbad_other_types",
                "simbad_sp_type",
                "simbad_ids",
                "simbad_query_type",
            ]
        )

    out = pd.DataFrame(index=df.index)
    out["simbad_main_id"] = _first_present(df, ["simbad_main_id", "main_id", "MAIN_ID"])
    out["simbad_ra"] = _first_present(df, ["simbad_ra", "ra", "RA", "RA_d"])
    out["simbad_dec"] = _first_present(df, ["simbad_dec", "dec", "DEC", "DEC_d"])
    out["simbad_main_type"] = _first_present(df, ["simbad_main_type", "main_type", "OTYPE", "otype"])
    out["simbad_other_types"] = _first_present(df, ["simbad_other_types", "other_types", "OTYPES", "otypes"])
    out["simbad_sp_type"] = _first_present(df, ["simbad_sp_type", "sp_type", "SP_TYPE", "SP_TYPE_2", "sp"])
    out["simbad_ids"] = _first_present(df, ["simbad_ids", "ids", "IDS", "ID"])
    out["simbad_query_type"] = _first_present(df, ["simbad_query_type"])
    out["source_id"] = out["simbad_ids"].map(extract_gaia_dr3_source_id).astype("Int64")
    out["gaia_designation"] = out["source_id"].map(lambda value: pd.NA if pd.isna(value) else f"Gaia DR3 {int(value)}")
    out = out[out["source_id"].notna()].copy()
    out["source_id"] = out["source_id"].astype("int64")
    return out.reset_index(drop=True)


def extract_gaia_dr3_source_id(value: object) -> int | None:
    if pd.isna(value):
        return None
    match = GAIA_DR3_ID_RE.search(str(value))
    return int(match.group(1)) if match else None


def exclude_wolf_rayet_rows(
    df: pd.DataFrame,
    *,
    wr_source_ids: Iterable[int] = (),
    patterns: Iterable[str] | None = None,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    wr_ids = {int(value) for value in wr_source_ids}
    pattern = _compile_exclusion_pattern(patterns)
    type_text = (
        df.get("simbad_main_type", pd.Series("", index=df.index)).astype("string").fillna("")
        + " "
        + df.get("simbad_other_types", pd.Series("", index=df.index)).astype("string").fillna("")
    )
    is_wr_type = type_text.str.contains(pattern, na=False)
    is_known_wr = df["source_id"].isin(wr_ids) if wr_ids else pd.Series(False, index=df.index)
    return df[~is_wr_type & ~is_known_wr].reset_index(drop=True)


def deduplicate_simbad_sources(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    sort_cols = [c for c in ["source_id", "simbad_query_type", "simbad_main_id"] if c in df.columns]
    out = df.sort_values(sort_cols).drop_duplicates("source_id", keep="first")
    return out.reset_index(drop=True)


def _first_present(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for candidate in candidates:
        if candidate in df.columns:
            return df[candidate]
    return pd.Series(pd.NA, index=df.index)


def _compile_exclusion_pattern(patterns: Iterable[str] | None) -> re.Pattern[str]:
    if patterns is None:
        return WOLF_RAYET_RE
    escaped = []
    for value in patterns:
        token = str(value)
        if token == "WR*":
            escaped.append(r"\bwr\*")
        else:
            escaped.append(re.escape(token).replace(r"\-", r"-?").replace(r"\ ", r"\s*"))
    return re.compile("|".join(escaped), re.IGNORECASE)
