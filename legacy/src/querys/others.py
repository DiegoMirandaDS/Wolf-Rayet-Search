### Querys to Wise and 2MASS (maybe SIMBAD) ###

from __future__ import annotations
from typing import Optional, Sequence, List
import numpy as np
import pandas as pd

from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.vizier import Vizier

# Catalogues
CAT_2MASS   = "II/246/out"        # 2MASS PSC
CAT_ALLWISE = "II/328/allwise"    # AllWISE Source Catalog

COLS_2MASS = [
    "2MASS","RAJ2000","DEJ2000",
    "Jmag","Hmag","Kmag","e_Jmag","e_Hmag","e_Kmag",
    "Qflg","_r"
]
COLS_WISE = [
    "AllWISE","RAJ2000","DEJ2000",
    "W1mag","W2mag","W3mag","W4mag",
    "e_W1mag","e_W2mag","e_W3mag","e_W4mag",
    "qph", "_r"
]

def _pick_id_col(df: pd.DataFrame, candidates: Sequence[str] = ("identifier","gaia_id","source_id","designation")) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _query_catalog_bulk(
    gaia_df: pd.DataFrame,
    *,
    catalog: str,
    columns: List[str],
    radius_arcsec: float = 5.0,
    save_csv: Optional[str] = None,
    chunk_size: Optional[int] = None,
) -> pd.DataFrame:
    if gaia_df is None or gaia_df.empty:
        return pd.DataFrame()

    # Coordenadas de entrada (ICRS, grados)
    src_coords = SkyCoord(gaia_df["ra"].to_numpy() * u.deg,
                          gaia_df["dec"].to_numpy() * u.deg, frame="icrs")

    v = Vizier(columns=columns)
    v.ROW_LIMIT = -1
    v.TIMEOUT = 180

    def _run(coords_sub: SkyCoord, start_offset: int) -> pd.DataFrame:
        res = v.query_region(coords_sub, radius=radius_arcsec * u.arcsec, catalog=catalog)
        if not res:
            return pd.DataFrame()
        df = res[0].to_pandas()

        # Normalizar índice de consulta a 0‑based GLOBAL en columna 'query_idx'
        qcol = "__q" if "__q" in df.columns else ("_q" if "_q" in df.columns else None)
        if qcol is not None:
            q = pd.to_numeric(df[qcol], errors="coerce")
            # Detección 1‑based: si hay valores = len(coords_sub) o min==1
            is_one_based = (q.min() == 1) or (q.max() == len(coords_sub))
            q0 = (q - 1) if is_one_based else q
            df["query_idx"] = q0.astype("Int64") + start_offset
            df = df.drop(columns=[qcol])
        else:
            # Raro, pero mantenemos NaN; no calcularemos sep_arcsec para esas filas
            df["query_idx"] = pd.NA

        return df

    # UNA llamada o por chunks (con offset global correcto)
    frames = []
    if chunk_size and chunk_size > 0:
        start = 0
        n = len(src_coords)
        while start < n:
            end = min(start + chunk_size, n)
            part = _run(src_coords[start:end], start_offset=start)
            frames.append(part)
            start = end
        out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    else:
        out = _run(src_coords, start_offset=0)

    # Adjuntar id legible de la fila de origen (opcional, sin renombrar)
    id_col = _pick_id_col(gaia_df)
    if id_col and "query_idx" in out.columns:
        map_df = gaia_df.reset_index(drop=True)[[id_col]].copy()
        map_df["query_idx"] = np.arange(len(map_df), dtype=int)
        out = out.merge(map_df, on="query_idx", how="left")

    # Calcular separación precisa (arcsec) contra la coord de entrada (solo donde hay índice válido)
    if {"RAJ2000", "DEJ2000", "query_idx"}.issubset(out.columns):
        valid = out["query_idx"].notna()
        if valid.any():
            i = out.loc[valid, "query_idx"].astype(int).to_numpy()
            # seguridad por si algún mirror devolvió algo raro
            i = np.clip(i, 0, len(src_coords) - 1)
            trg = SkyCoord(out.loc[valid, "RAJ2000"].to_numpy() * u.deg,
                           out.loc[valid, "DEJ2000"].to_numpy() * u.deg, frame="icrs")
            out.loc[valid, "sep_arcsec"] = src_coords[i].separation(trg).arcsecond

    if save_csv:
        out.to_csv(save_csv, index=False)

    return out

def query_2mass_nf_bulk(
    gaia_df: pd.DataFrame,
    *,
    radius_arcsec: float = 5.0,
    save_csv: Optional[str] = None,
    chunk_size: Optional[int] = None,
) -> pd.DataFrame:
    """Una sola query a 2MASS (VizieR II/246/out) para todas las coordenadas."""
    return _query_catalog_bulk(
        gaia_df,
        catalog=CAT_2MASS,
        columns=COLS_2MASS,
        radius_arcsec=radius_arcsec,
        save_csv=save_csv,
        chunk_size=chunk_size,
    )

def query_wise_nf_bulk(
    gaia_df: pd.DataFrame,
    *,
    radius_arcsec: float = 5.0,
    save_csv: Optional[str] = None,
    chunk_size: Optional[int] = None,
) -> pd.DataFrame:
    """Una sola query a AllWISE (VizieR II/328/allwise) para todas las coordenadas."""
    return _query_catalog_bulk(
        gaia_df,
        catalog=CAT_ALLWISE,
        columns=COLS_WISE,
        radius_arcsec=radius_arcsec,
        save_csv=save_csv,
        chunk_size=chunk_size,
    )
