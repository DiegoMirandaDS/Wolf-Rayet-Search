from __future__ import annotations

from itertools import combinations
from pathlib import Path

import duckdb
import pandas as pd


GAIA_BANDS = ["G", "BP", "RP"]
PHOTOMETRY_BANDS = ["J", "H", "Ks", "W1", "W2"]
TMASS_BANDS = ["J", "H", "Ks"]
WISE_BANDS = ["W1", "W2"]

QUALITY_ALLOWED = {
    "A": {"A"},
    "B": {"A", "B"},
}


def load_available_photometry(db_path: str | Path) -> pd.DataFrame:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return con.execute(
            """
            SELECT
                g.source_id,
                r.wr_id,
                g.G,
                g.BP,
                g.RP,
                g.parallax,
                g.parallax_over_error,
                g.ruwe,
                t.J,
                t.H,
                t.Ks,
                t.tmass_quality,
                w.W1,
                w.W2,
                w.W3,
                w.W4,
                w.wise_quality
            FROM wr_reference r
            INNER JOIN gaia_sources g USING (source_id)
            LEFT JOIN twomass_matches t USING (source_id)
            LEFT JOIN wise_matches w USING (source_id)
            """
        ).fetchdf()
    finally:
        con.close()


def quality_char(series: pd.Series, position: int) -> pd.Series:
    return series.astype("string").str[position]


def band_quality_mask(df: pd.DataFrame, band: str, min_quality: str) -> pd.Series:
    allowed = QUALITY_ALLOWED[min_quality]
    if band in TMASS_BANDS:
        position = TMASS_BANDS.index(band)
        return quality_char(df["tmass_quality"], position).isin(allowed)
    if band in WISE_BANDS:
        position = WISE_BANDS.index(band)
        return quality_char(df["wise_quality"], position).isin(allowed)
    raise ValueError(f"Unsupported quality band: {band}")


def availability_mask(
    df: pd.DataFrame,
    *,
    bands: list[str],
    min_quality: str,
    require_gaia: bool = True,
) -> pd.Series:
    required = list(GAIA_BANDS) if require_gaia else []
    required.extend(bands)
    mask = pd.Series(True, index=df.index)
    for band in required:
        mask &= df[band].notna()
    for band in bands:
        mask &= band_quality_mask(df, band, min_quality)
    return mask


def parallax_masks(df: pd.DataFrame, min_parallax: float = 0.0) -> dict[str, pd.Series]:
    soft = df["parallax"] > min_parallax
    return {
        "photometry": pd.Series(True, index=df.index),
        "parallax_soft": soft,
        "poe_1": soft & (df["parallax_over_error"] >= 1),
        "poe_2": soft & (df["parallax_over_error"] >= 2),
        "poe_3": soft & (df["parallax_over_error"] >= 3),
        "poe_5": soft & (df["parallax_over_error"] >= 5),
    }


def enumerate_band_combinations(
    df: pd.DataFrame,
    *,
    min_quality: str,
    min_parallax: float = 0.0,
    min_bands: int = 1,
) -> pd.DataFrame:
    rows = []
    pmasks = parallax_masks(df, min_parallax=min_parallax)
    total = len(df)
    for size in range(min_bands, len(PHOTOMETRY_BANDS) + 1):
        for combo in combinations(PHOTOMETRY_BANDS, size):
            bands = list(combo)
            base_mask = availability_mask(df, bands=bands, min_quality=min_quality)
            for parallax_name, parallax_mask in pmasks.items():
                mask = base_mask & parallax_mask
                rows.append(
                    {
                        "min_quality": min_quality,
                        "bands": "+".join(bands),
                        "n_required_bands": len(bands),
                        "uses_2mass": any(b in TMASS_BANDS for b in bands),
                        "uses_wise": any(b in WISE_BANDS for b in bands),
                        "parallax_filter": parallax_name,
                        "rows": int(mask.sum()),
                        "retention_vs_wr_reference": float(mask.sum() / total),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["parallax_filter", "rows", "n_required_bands"],
        ascending=[True, False, False],
    )


def per_band_retention(
    df: pd.DataFrame,
    *,
    min_quality: str,
    min_parallax: float = 0.0,
) -> pd.DataFrame:
    rows = []
    pmasks = parallax_masks(df, min_parallax=min_parallax)
    gaia_mask = availability_mask(df, bands=[], min_quality=min_quality)
    for band in PHOTOMETRY_BANDS:
        band_mask = availability_mask(df, bands=[band], min_quality=min_quality)
        for parallax_name, parallax_mask in pmasks.items():
            mask = band_mask & parallax_mask
            rows.append(
                {
                    "min_quality": min_quality,
                    "band": band,
                    "parallax_filter": parallax_name,
                    "rows": int(mask.sum()),
                    "lost_vs_gaia_only": int((gaia_mask & parallax_mask).sum() - mask.sum()),
                }
            )
    return pd.DataFrame(rows)


def compare_quality_thresholds(df: pd.DataFrame, min_parallax: float = 0.0) -> pd.DataFrame:
    frames = [
        enumerate_band_combinations(df, min_quality=quality, min_parallax=min_parallax)
        for quality in ["A", "B"]
    ]
    return pd.concat(frames, ignore_index=True)


def add_mission_band_counts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    split_bands = out["bands"].str.split("+")
    out["n_2mass_bands"] = split_bands.map(lambda bands: sum(b in TMASS_BANDS for b in bands))
    out["n_wise_bands"] = split_bands.map(lambda bands: sum(b in WISE_BANDS for b in bands))
    out["valid_min_2_per_used_mission"] = (
        ((out["n_2mass_bands"] == 0) | (out["n_2mass_bands"] >= 2))
        & ((out["n_wise_bands"] == 0) | (out["n_wise_bands"] >= 2))
    )
    out["valid_3_2mass_2_wise"] = (out["n_2mass_bands"] == 3) & (out["n_wise_bands"] == 2)
    return out
