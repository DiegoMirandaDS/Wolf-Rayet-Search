from __future__ import annotations

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.vizier import Vizier


TWOMASS_COLUMNS = [
    "2MASS",
    "RAJ2000",
    "DEJ2000",
    "Jmag",
    "Hmag",
    "Kmag",
    "e_Jmag",
    "e_Hmag",
    "e_Kmag",
    "Qflg",
    "_r",
]
WISE_COLUMNS = [
    "AllWISE",
    "RAJ2000",
    "DEJ2000",
    "W1mag",
    "W2mag",
    "W3mag",
    "W4mag",
    "e_W1mag",
    "e_W2mag",
    "e_W3mag",
    "e_W4mag",
    "qph",
    "_r",
]


def query_catalog_by_coordinates(
    gaia_sources: pd.DataFrame,
    *,
    catalog: str,
    columns: list[str],
    radius_arcsec: float,
    timeout_seconds: int = 180,
) -> pd.DataFrame:
    if gaia_sources.empty:
        return pd.DataFrame()

    source = gaia_sources.reset_index(drop=True).copy()
    coords = SkyCoord(source["ra"].to_numpy() * u.deg, source["dec"].to_numpy() * u.deg, frame="icrs")
    vizier = Vizier(columns=columns)
    vizier.ROW_LIMIT = -1
    vizier.TIMEOUT = timeout_seconds
    result = vizier.query_region(coords, radius=radius_arcsec * u.arcsec, catalog=catalog)
    if not result:
        return pd.DataFrame()

    out = result[0].to_pandas()
    qcol = "__q" if "__q" in out.columns else ("_q" if "_q" in out.columns else None)
    if qcol is None:
        out["query_idx"] = pd.NA
    else:
        q = pd.to_numeric(out[qcol], errors="coerce")
        one_based = (q.min() == 1) or (q.max() == len(source))
        out["query_idx"] = (q - 1 if one_based else q).astype("Int64")
        out = out.drop(columns=[qcol])

    idx_map = source[["source_id", "ra", "dec"]].copy()
    idx_map["query_idx"] = np.arange(len(idx_map), dtype=int)
    out = out.merge(idx_map, on="query_idx", how="left", suffixes=("", "_gaia"))

    if {"RAJ2000", "DEJ2000", "query_idx"}.issubset(out.columns):
        valid = out["query_idx"].notna()
        if valid.any():
            source_idx = out.loc[valid, "query_idx"].astype(int).to_numpy()
            target = SkyCoord(
                out.loc[valid, "RAJ2000"].to_numpy() * u.deg,
                out.loc[valid, "DEJ2000"].to_numpy() * u.deg,
                frame="icrs",
            )
            out.loc[valid, "sep_arcsec"] = coords[source_idx].separation(target).arcsecond
    return out


def select_duplicates_by_radius(
    df: pd.DataFrame,
    *,
    primary_radius_arcsec: float,
    secondary_radius_arcsec: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if df.empty:
        return pd.DataFrame(), {
            "n_input": 0,
            "n_groups": 0,
            "n_singletons_kept": 0,
            "n_dupe_groups": 0,
            "n_resolved_at_primary": 0,
            "n_resolved_at_secondary": 0,
            "n_dropped_ambiguous": 0,
            "dropped_source_ids": [],
        }

    required = {"source_id", "sep_arcsec"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing duplicate-resolution columns: {missing}")

    counts = df.groupby("source_id", dropna=False).size()
    single_ids = set(counts[counts == 1].index)
    dupe_ids = set(counts[counts > 1].index)

    picks = [df[df["source_id"].isin(single_ids)].copy()]
    dupes = df[df["source_id"].isin(dupe_ids)].copy()
    primary = dupes[dupes["sep_arcsec"] <= primary_radius_arcsec].copy()

    resolved_primary = []
    still_dupes = []
    for source_id, group in primary.groupby("source_id", dropna=False):
        if len(group) == 1:
            resolved_primary.append(group.index[0])
        else:
            still_dupes.append(source_id)

    secondary = primary[primary["source_id"].isin(still_dupes)]
    secondary = secondary[secondary["sep_arcsec"] <= secondary_radius_arcsec].copy()

    resolved_secondary = []
    ambiguous = []
    for source_id, group in secondary.groupby("source_id", dropna=False):
        if len(group) == 1:
            resolved_secondary.append(group.index[0])
        else:
            ambiguous.append(source_id)

    zero_after_primary = sorted(set(dupe_ids) - set(primary["source_id"].dropna().unique()))
    zero_after_secondary = sorted(set(still_dupes) - set(secondary["source_id"].dropna().unique()))
    ambiguous = sorted(set(ambiguous) | set(zero_after_primary) | set(zero_after_secondary))

    if resolved_primary:
        picks.append(primary.loc[resolved_primary])
    if resolved_secondary:
        picks.append(secondary.loc[resolved_secondary])

    out = pd.concat(picks, ignore_index=True) if picks else pd.DataFrame(columns=df.columns)
    stats = {
        "n_input": int(len(df)),
        "n_groups": int(len(counts)),
        "n_singletons_kept": int(len(single_ids)),
        "n_dupe_groups": int(len(dupe_ids)),
        "n_resolved_at_primary": int(len(resolved_primary)),
        "n_resolved_at_secondary": int(len(resolved_secondary)),
        "n_dropped_ambiguous": int(len(ambiguous)),
        "dropped_source_ids": [int(v) for v in ambiguous if pd.notna(v)],
    }
    return out.reset_index(drop=True), stats


def normalize_twomass_vizier(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = df.rename(
        columns={
            "2MASS": "tmass_id",
            "RAJ2000": "ra_tmass",
            "DEJ2000": "dec_tmass",
            "Jmag": "J",
            "Hmag": "H",
            "Kmag": "Ks",
            "e_Jmag": "J_error",
            "e_Hmag": "H_error",
            "e_Kmag": "Ks_error",
            "Qflg": "tmass_quality",
        }
    )
    keep = ["source_id", "tmass_id", "ra_tmass", "dec_tmass", "J", "H", "Ks", "J_error", "H_error", "Ks_error", "tmass_quality", "sep_arcsec"]
    out = out[[c for c in keep if c in out.columns]].copy()
    out["match_method"] = "vizier_cone"
    return out


def normalize_wise_vizier(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = df.rename(
        columns={
            "AllWISE": "wise_id",
            "RAJ2000": "ra_wise",
            "DEJ2000": "dec_wise",
            "W1mag": "W1",
            "W2mag": "W2",
            "W3mag": "W3",
            "W4mag": "W4",
            "e_W1mag": "W1_error",
            "e_W2mag": "W2_error",
            "e_W3mag": "W3_error",
            "e_W4mag": "W4_error",
            "qph": "wise_quality",
        }
    )
    keep = [
        "source_id",
        "wise_id",
        "ra_wise",
        "dec_wise",
        "W1",
        "W2",
        "W3",
        "W4",
        "W1_error",
        "W2_error",
        "W3_error",
        "W4_error",
        "wise_quality",
        "sep_arcsec",
    ]
    out = out[[c for c in keep if c in out.columns]].copy()
    out["match_method"] = "vizier_cone"
    return out
