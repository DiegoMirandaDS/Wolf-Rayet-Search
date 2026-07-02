"""ADQL builders and normalization helpers for Gaia DR3 photometry and crossmatches."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


GAIA_COLUMNS = [
    "source_id",
    "designation",
    "ra",
    "dec",
    "phot_g_mean_mag",
    "phot_bp_mean_mag",
    "phot_rp_mean_mag",
    "phot_g_mean_flux",
    "phot_g_mean_flux_error",
    "phot_bp_mean_flux",
    "phot_bp_mean_flux_error",
    "phot_rp_mean_flux",
    "phot_rp_mean_flux_error",
    "parallax",
    "parallax_error",
    "parallax_over_error",
    "pmra",
    "pmra_error",
    "pmdec",
    "pmdec_error",
    "radial_velocity",
    "radial_velocity_error",
    "ruwe",
    "teff_gspphot",
    "logg_gspphot",
    "mh_gspphot",
]


def _chunks(values: list[int], size: int) -> Iterable[list[int]]:
    for i in range(0, len(values), size):
        yield values[i : i + size]


def _source_id_csv(source_ids: list[int]) -> str:
    return ", ".join(str(int(v)) for v in source_ids)


def build_gaia_core_query(source_ids: list[int]) -> str:
    ids = _source_id_csv(source_ids)
    return f"""
    SELECT
        gaia.source_id,
        gaia.designation,
        gaia.ra,
        gaia.dec,
        gaia.phot_g_mean_mag,
        gaia.phot_bp_mean_mag,
        gaia.phot_rp_mean_mag,
        gaia.phot_g_mean_flux,
        gaia.phot_g_mean_flux_error,
        gaia.phot_bp_mean_flux,
        gaia.phot_bp_mean_flux_error,
        gaia.phot_rp_mean_flux,
        gaia.phot_rp_mean_flux_error,
        gaia.parallax,
        gaia.parallax_error,
        gaia.parallax_over_error,
        gaia.pmra,
        gaia.pmra_error,
        gaia.pmdec,
        gaia.pmdec_error,
        gaia.radial_velocity,
        gaia.radial_velocity_error,
        gaia.ruwe,
        gaia.teff_gspphot,
        gaia.logg_gspphot,
        gaia.mh_gspphot
    FROM gaiadr3.gaia_source AS gaia
    WHERE gaia.source_id IN ({ids})
    """


def build_gaia_tmass_query(source_ids: list[int]) -> str:
    ids = _source_id_csv(source_ids)
    return f"""
    SELECT
        gaia.source_id,
        xjoin.original_psc_source_id AS tmass_id,
        tmass.j_m,
        tmass.h_m,
        tmass.ks_m,
        tmass.j_msigcom,
        tmass.h_msigcom,
        tmass.ks_msigcom,
        tmass.ph_qual AS tmass_quality
    FROM gaiadr3.gaia_source AS gaia
    JOIN gaiadr3.tmass_psc_xsc_best_neighbour AS xmatch USING (source_id)
    JOIN gaiadr3.tmass_psc_xsc_join AS xjoin USING (clean_tmass_psc_xsc_oid)
    JOIN gaiadr1.tmass_original_valid AS tmass
        ON xjoin.original_psc_source_id = tmass.designation
    WHERE gaia.source_id IN ({ids})
    """


def build_gaia_wise_query(source_ids: list[int]) -> str:
    ids = _source_id_csv(source_ids)
    return f"""
    SELECT
        gaia.source_id,
        wise.allwise_oid,
        wise.designation AS wise_id,
        wise.w1mpro,
        wise.w2mpro,
        wise.w3mpro,
        wise.w4mpro,
        wise.w1mpro_error,
        wise.w2mpro_error,
        wise.w3mpro_error,
        wise.w4mpro_error,
        wise.ph_qual AS wise_quality
    FROM gaiadr3.gaia_source AS gaia
    JOIN gaiadr3.allwise_best_neighbour AS wise_match USING (source_id)
    JOIN gaiadr1.allwise_original_valid AS wise
        ON wise_match.allwise_oid = wise.allwise_oid
    WHERE gaia.source_id IN ({ids})
    """


def normalize_gaia_result(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    source_cols = {
        "designation": "gaia_designation",
        "phot_g_mean_mag": "G",
        "phot_bp_mean_mag": "BP",
        "phot_rp_mean_mag": "RP",
        "phot_g_mean_flux": "G_flux",
        "phot_g_mean_flux_error": "G_flux_error",
        "phot_bp_mean_flux": "BP_flux",
        "phot_bp_mean_flux_error": "BP_flux_error",
        "phot_rp_mean_flux": "RP_flux",
        "phot_rp_mean_flux_error": "RP_flux_error",
        "parallax_error": "parallax_error",
        "pmra_error": "pmra_error",
        "pmdec_error": "pmdec_error",
        "radial_velocity_error": "radial_velocity_error",
        "teff_gspphot": "teff",
        "logg_gspphot": "logg",
        "mh_gspphot": "feh",
    }
    gaia_source_cols = [c for c in GAIA_COLUMNS if c in df.columns]
    gaia_sources = df[gaia_source_cols].rename(columns=source_cols).drop_duplicates("source_id")
    return gaia_sources


def normalize_gaia_tmass_result(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    tmass_cols = [
        "source_id",
        "tmass_id",
        "j_m",
        "h_m",
        "ks_m",
        "j_msigcom",
        "h_msigcom",
        "ks_msigcom",
        "tmass_quality",
    ]
    twomass = df[[c for c in tmass_cols if c in df.columns]].copy()
    twomass = twomass.rename(
        columns={
            "j_m": "J",
            "h_m": "H",
            "ks_m": "Ks",
            "j_msigcom": "J_error",
            "h_msigcom": "H_error",
            "ks_msigcom": "Ks_error",
        }
    )
    twomass = twomass[twomass["tmass_id"].notna()].drop_duplicates("source_id")
    twomass["match_method"] = "gaia_xmatch"
    twomass["sep_arcsec"] = pd.NA
    return twomass


def normalize_gaia_wise_result(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    wise_cols = [
        "source_id",
        "allwise_oid",
        "wise_id",
        "w1mpro",
        "w2mpro",
        "w3mpro",
        "w4mpro",
        "w1mpro_error",
        "w2mpro_error",
        "w3mpro_error",
        "w4mpro_error",
        "wise_quality",
    ]
    wise = df[[c for c in wise_cols if c in df.columns]].copy()
    wise = wise.rename(
        columns={
            "w1mpro": "W1",
            "w2mpro": "W2",
            "w3mpro": "W3",
            "w4mpro": "W4",
            "w1mpro_error": "W1_error",
            "w2mpro_error": "W2_error",
            "w3mpro_error": "W3_error",
            "w4mpro_error": "W4_error",
        }
    )
    wise = wise[wise["wise_id"].notna()].drop_duplicates("source_id")
    wise["match_method"] = "gaia_xmatch"
    wise["sep_arcsec"] = pd.NA
    return wise


def query_gaia_reference(source_ids: list[int], chunk_size: int = 400) -> pd.DataFrame:
    from astroquery.gaia import Gaia

    return _query_chunks(Gaia, build_gaia_core_query, source_ids, chunk_size)


def query_gaia_tmass(source_ids: list[int], chunk_size: int = 400) -> pd.DataFrame:
    from astroquery.gaia import Gaia

    return _query_chunks(Gaia, build_gaia_tmass_query, source_ids, chunk_size)


def query_gaia_wise(source_ids: list[int], chunk_size: int = 400) -> pd.DataFrame:
    from astroquery.gaia import Gaia

    return _query_chunks(Gaia, build_gaia_wise_query, source_ids, chunk_size)


def _query_chunks(gaia_client, query_builder, source_ids: list[int], chunk_size: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for chunk in _chunks(source_ids, chunk_size):
        job = gaia_client.launch_job_async(query_builder(chunk))
        frames.append(job.get_results().to_pandas())
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
