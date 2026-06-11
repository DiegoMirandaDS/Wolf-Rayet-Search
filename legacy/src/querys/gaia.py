from __future__ import annotations
from typing import Optional, Union, Tuple
from pathlib import Path
import pandas as pd

from src.querys.utils import (
    rename_and_select,
    COL_RENAME_GAIA_XMATCH,
    COL_RENAME_GAIA_CORE,
)


def _in_clause_str(values) -> str:
    return ", ".join([f"'{v}'" for v in values])


def wr_from_gaia(
    df: Optional[pd.DataFrame],
    identifier_col: str,
    df_not_found: bool = False,
    output_route: Optional[Union[str, Path]] = None
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:

    if df is None or identifier_col not in df.columns:
        raise ValueError("A valid DataFrame and identifier_col are required")

    from astroquery.gaia import Gaia

    gaia_identifiers = (
        df[identifier_col]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    if not gaia_identifiers:
        return (pd.DataFrame(), pd.DataFrame()) if df_not_found else pd.DataFrame()

    identifiers_str = _in_clause_str(gaia_identifiers)

    # ==============================
    # Q1: GAIA + 2MASS GAIA + WISE GAIA
    # ==============================
    query_xmatch = f"""
    SELECT
        -- Gaia core
        gaia.source_id,
        gaia.designation,
        gaia.ra, gaia.dec,
        gaia.phot_g_mean_mag, gaia.phot_bp_mean_mag, gaia.phot_rp_mean_mag,
        gaia.phot_g_mean_flux, gaia.phot_g_mean_flux_error,
        gaia.phot_bp_mean_flux, gaia.phot_bp_mean_flux_error,
        gaia.phot_rp_mean_flux, gaia.phot_rp_mean_flux_error,
        gaia.parallax, gaia.parallax_error,
        gaia.pmra, gaia.pmra_error, gaia.pmdec, gaia.pmdec_error,
        gaia.radial_velocity, gaia.radial_velocity_error,
        gaia.teff_gspphot, gaia.logg_gspphot, gaia.mh_gspphot,
        gaia.parallax_over_error, gaia.ruwe,

        -- 2MASS Gaia
        xjoin.original_psc_source_id AS tmass_oid,
        tmass.designation AS designation2,
        tmass.j_m, tmass.h_m, tmass.ks_m,
        tmass.ph_qual,
        tmass.j_msigcom, tmass.h_msigcom, tmass.ks_msigcom,

        -- WISE Gaia
        wise.allwise_oid AS allwise_oid,
        wise.designation AS designation3,
        wise.w1mpro, wise.w2mpro, wise.w3mpro, wise.w4mpro,
        wise.ph_qual AS ph_qual2,
        wise.w1mpro_error, wise.w2mpro_error, wise.w3mpro_error, wise.w4mpro_error

    FROM gaiadr3.gaia_source AS gaia
    JOIN gaiadr3.tmass_psc_xsc_best_neighbour AS xmatch USING (source_id)
    JOIN gaiadr3.tmass_psc_xsc_join AS xjoin USING (clean_tmass_psc_xsc_oid)
    JOIN gaiadr1.tmass_original_valid AS tmass ON xjoin.original_psc_source_id = tmass.designation
    JOIN gaiadr3.allwise_best_neighbour AS wise_match USING (source_id)
    JOIN gaiadr1.allwise_original_valid AS wise ON wise_match.allwise_oid = wise.allwise_oid

    WHERE gaia.designation IN ({identifiers_str})
    """

    print("Launching Gaia crossmatch query (Gaia + 2MASS + WISE)...")
    job1 = Gaia.launch_job_async(query_xmatch)
    results1 = job1.get_results()
    df_xmatch_raw = results1.to_pandas()
    print(f"Gaia crossmatch query completed: {len(df_xmatch_raw)} rows.")

    df_xmatch = rename_and_select(df_xmatch_raw, COL_RENAME_GAIA_XMATCH)
    keep_from_original = [identifier_col]
    if "matching_ident" in df.columns:
        keep_from_original.append("matching_ident")
    df_xmatch = df_xmatch.merge(
        df[keep_from_original],
        left_on="gaia_id", right_on=identifier_col,
        how="left"
    )

    if output_route is not None:
        Path(output_route).parent.mkdir(parents=True, exist_ok=True)
        df_xmatch.to_csv(output_route, index=False)

    if not df_not_found:
        return df_xmatch

    # ==============================
    # Determinar los que faltaron en la Q1 (por los INNER JOIN)
    # ==============================

    found_designations = set(df_xmatch_raw["designation"].astype(str)) if not df_xmatch_raw.empty else set()
    missing = sorted(set(gaia_identifiers) - found_designations)

    if not missing:
        print("No missing identifiers for Gaia-core-only query.")
        return df_xmatch, pd.DataFrame()

    # ==============================
    # Q2: SOLO GAIA CORE para los que faltaron
    # ==============================
    missing_str = _in_clause_str(missing)
    query_core = f"""
    SELECT
        gaia.source_id,
        gaia.designation,
        gaia.ra, gaia.dec,
        gaia.phot_g_mean_mag, gaia.phot_bp_mean_mag, gaia.phot_rp_mean_mag,
        gaia.phot_g_mean_flux, gaia.phot_g_mean_flux_error,
        gaia.phot_bp_mean_flux, gaia.phot_bp_mean_flux_error,
        gaia.phot_rp_mean_flux, gaia.phot_rp_mean_flux_error,
        gaia.parallax, gaia.parallax_error,
        gaia.pmra, gaia.pmra_error, gaia.pmdec, gaia.pmdec_error,
        gaia.radial_velocity, gaia.radial_velocity_error,
        gaia.teff_gspphot, gaia.logg_gspphot, gaia.mh_gspphot,
        gaia.parallax_over_error, gaia.ruwe
    FROM gaiadr3.gaia_source AS gaia
    WHERE gaia.designation IN ({missing_str})
    """

    print(f"Launching Gaia-core-only query for {len(missing)} missing identifiers...")
    job2 = Gaia.launch_job_async(query_core)
    results2 = job2.get_results()
    df_core_raw = results2.to_pandas()
    print(f"Gaia-core-only query completed: {len(df_core_raw)} rows.")

    df_core = rename_and_select(df_core_raw, COL_RENAME_GAIA_CORE)

    df_core = df_core.merge(
        df[keep_from_original],
        left_on="gaia_id", right_on=identifier_col,
        how="left"
    )

    return df_xmatch, df_core
