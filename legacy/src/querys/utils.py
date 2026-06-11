from __future__ import annotations
from typing import Dict, Iterable, Literal, Optional, Sequence
import pandas as pd

# ===== Diccionario maestro (todos los orígenes soportados) =====
_GAIA_CORE_RAW_TO_STD = {
    "source_id": "source_id", "designation": "gaia_id",
    "ra": "ra", "dec": "dec",

    "phot_g_mean_mag": "Gmag",
    "phot_bp_mean_mag": "BPmag",
    "phot_rp_mean_mag": "RPmag",

    "phot_g_mean_flux": "Gflux",
    "phot_g_mean_flux_error": "e_Gflux",
    "phot_bp_mean_flux": "BPflux",
    "phot_bp_mean_flux_error": "e_BPflux",
    "phot_rp_mean_flux": "RPflux",
    "phot_rp_mean_flux_error": "e_RPflux",

    "parallax": "parallax", "parallax_error": "e_parallax",
    "pmra": "pmRA", "pmra_error": "e_pmRA",
    "pmdec": "pmDE", "pmdec_error": "e_pmDE",

    "radial_velocity": "radial_velocity",
    "radial_velocity_error": "e_radial_velocity",

    "teff_gspphot": "teff",
    "logg_gspphot": "logg",
    "mh_gspphot": "feh",

    "parallax_over_error": "parallax_over_error",
    "ruwe": "ruwe",

    # alias de tu CSV de entrada (Crowther)
    "Alias1": "Alias1",
}

# identidades para DFs ya normalizados (como tu gaia_nf)
_GAIA_CORE_STD_ID = {
    "source_id": "source_id", "gaia_id": "gaia_id",
    "ra": "ra", "dec": "dec",
    "Gmag": "Gmag", "BPmag": "BPmag", "RPmag": "RPmag",
    "Gflux": "Gflux", "e_Gflux": "e_Gflux",
    "BPflux": "BPflux", "e_BPflux": "e_BPflux",
    "RPflux": "RPflux", "e_RPflux": "e_RPflux",
    "parallax": "parallax", "e_parallax": "e_parallax",
    "pmRA": "pmRA", "e_pmRA": "e_pmRA",
    "pmDE": "pmDE", "e_pmDE": "e_pmDE",
    "radial_velocity": "radial_velocity", "e_radial_velocity": "e_radial_velocity",
    "teff": "teff", "logg": "logg", "feh": "feh",
    "parallax_over_error": "parallax_over_error",
    "ruwe": "ruwe",
    "Alias1": "Alias1",
}

COL_RENAME_GAIA_CORE = {**_GAIA_CORE_RAW_TO_STD, **_GAIA_CORE_STD_ID}


# --- 2MASS NF (VizieR II/246/out) ---
_TMASS_NF_RAW_TO_STD = {
    "2MASS": "tmass_id",
    "RAJ2000": "ra_2mass", "DEJ2000": "dec_2mass",
    "Jmag": "Jmag", "Hmag": "Hmag", "Kmag": "Kmag",
    "e_Jmag": "e_Jmag", "e_Hmag": "e_Hmag", "e_Kmag": "e_Kmag",
    "Qflg": "qual_tmass",
    "gaia_id": "gaia_id",  # ya viene en tus CSV NF
}
_TMASS_NF_STD_ID = {
    "tmass_id": "tmass_id",
    "ra_2mass": "ra_2mass", "dec_2mass": "dec_2mass",
    "Jmag": "Jmag", "Hmag": "Hmag", "Kmag": "Kmag",
    "e_Jmag": "e_Jmag", "e_Hmag": "e_Hmag", "e_Kmag": "e_Kmag",
    "qual_tmass": "qual_tmass",
    "gaia_id": "gaia_id",
}
COL_RENAME_2MASS_NF = {**_TMASS_NF_RAW_TO_STD, **_TMASS_NF_STD_ID}


# --- WISE NF (VizieR II/328/allwise) ---
_WISE_NF_RAW_TO_STD = {
    "AllWISE": "wise_id",
    "RAJ2000": "ra_wise", "DEJ2000": "dec_wise",
    "qph": "qual_wise",
    "W1mag": "W1mag", "W2mag": "W2mag", "W3mag": "W3mag", "W4mag": "W4mag",
    "e_W1mag": "e_W1mag", "e_W2mag": "e_W2mag", "e_W3mag": "e_W3mag", "e_W4mag": "e_W4mag",
    "gaia_id": "gaia_id",
}
_WISE_NF_STD_ID = {
    "wise_id": "wise_id",
    "ra_wise": "ra_wise", "dec_wise": "dec_wise",
    "qual_wise": "qual_wise",
    "W1mag": "W1mag", "W2mag": "W2mag", "W3mag": "W3mag", "W4mag": "W4mag",
    "e_W1mag": "e_W1mag", "e_W2mag": "e_W2mag", "e_W3mag": "e_W3mag", "e_W4mag": "e_W4mag",
    "gaia_id": "gaia_id",
}
COL_RENAME_WISE_NF = {**_WISE_NF_RAW_TO_STD, **_WISE_NF_STD_ID}


COL_RENAME_2MASS_GAIA = {
    "tmass_oid": "tmass_oid", "designation2": "tmass_id",
    "j_m": "Jmag", "h_m": "Hmag", "ks_m": "Kmag",
    "ph_qual": "qual_tmass",
    "j_msigcom": "e_Jmag", "h_msigcom": "e_Hmag", "ks_msigcom": "e_Kmag",
}
COL_RENAME_WISE_GAIA = {
    "allwise_oid": "allwise_oid", "designation3": "wise_id",
    "w1mpro": "W1mag", "w2mpro": "W2mag", "w3mpro": "W3mag", "w4mpro": "W4mag",
    "ph_qual2": "qual_wise",
    "w1mpro_error": "e_W1mag", "w2mpro_error": "e_W2mag",
    "w3mpro_error": "e_W3mag", "w4mpro_error": "e_W4mag",
}
COL_RENAME_GAIA_XMATCH = {**COL_RENAME_GAIA_CORE, **COL_RENAME_2MASS_GAIA, **COL_RENAME_WISE_GAIA}


def rename_and_select(
    df: pd.DataFrame,
    mapping: Dict[str, str],
    keep_also: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Selecciona columnas presentes en `mapping` y las renombra.
    Si `keep_also` trae columnas extra presentes en `df`, también las conserva."""
    if df is None or df.empty:
        return pd.DataFrame()
    present = [c for c in mapping if c in df.columns]
    out = df[present].rename(columns={k: mapping[k] for k in present})
    if keep_also:
        extra = [c for c in keep_also if c in df.columns and c not in out.columns]
        if extra:
            out = pd.concat([out, df[extra]], axis=1)
    return out


def normalize_catalog(
    df: pd.DataFrame,
    flavor: Literal["gaia_core","gaia_xmatch","2mass_nf","wise_nf","2mass_gaia","wise_gaia"],
    keep_also: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Atajo para normalizar por 'perfil' de columnas de origen."""
    mapping_by_flavor = {
        "gaia_core":   COL_RENAME_GAIA_CORE,
        "gaia_xmatch": COL_RENAME_GAIA_XMATCH,
        "2mass_nf":    COL_RENAME_2MASS_NF,
        "wise_nf":     COL_RENAME_WISE_NF,
        "2mass_gaia":  COL_RENAME_2MASS_GAIA,
        "wise_gaia":   COL_RENAME_WISE_GAIA,
    }
    return rename_and_select(df, mapping_by_flavor[flavor], keep_also=keep_also)


def _pick_join_key(
    left: pd.DataFrame,
    right: pd.DataFrame,
    candidates: Sequence[str] = ("gaia_id","source_id","Alias1"),
) -> str:
    """Devuelve la primera key presente en ambos DF."""
    for c in candidates:
        if c in left.columns and c in right.columns:
            return c
    raise ValueError(f"No common join key found among candidates={candidates}")


def unify_gaia_with_nf(
    gaia_core_df: pd.DataFrame,
    tmass_nf_df: Optional[pd.DataFrame] = None,
    wise_nf_df: Optional[pd.DataFrame] = None,
    keep_also: Iterable[str] = ("gaia_id","Alias1"),
    reorder: bool = True,
    require_all_nf: bool = False,   # <--- nuevo
) -> pd.DataFrame:
    if gaia_core_df is None or gaia_core_df.empty:
        return pd.DataFrame()

    base = normalize_catalog(gaia_core_df, "gaia_core", keep_also=keep_also)

    # --- 2MASS ---
    if tmass_nf_df is not None and not tmass_nf_df.empty:
        t2 = normalize_catalog(tmass_nf_df, "2mass_nf", keep_also=keep_also)
        key = _pick_join_key(base, t2)
        # Evitar columnas duplicadas en el merge (excepto la key)
        overlap = (set(t2.columns) & set(base.columns)) - {key}
        t2 = t2.drop(columns=list(overlap), errors="ignore")
        base = base.merge(t2, on=key, how="left")
        base["has_tmass_nf"] = base[["Jmag","Hmag","Kmag"]].notna().any(axis=1) | base["tmass_id"].notna()

    # --- WISE ---
    if wise_nf_df is not None and not wise_nf_df.empty:
        w2 = normalize_catalog(wise_nf_df, "wise_nf", keep_also=keep_also)
        key = _pick_join_key(base, w2)
        overlap = (set(w2.columns) & set(base.columns)) - {key}
        w2 = w2.drop(columns=list(overlap), errors="ignore")
        base = base.merge(w2, on=key, how="left")
        base["has_wise_nf"] = base[["W1mag","W2mag"]].notna().any(axis=1) | base["wise_id"].notna()

    # --- FILTRO: quedarse solo con quienes tienen TODOS los NF entregados ---
    if require_all_nf:
        mask = pd.Series(True, index=base.index)
        if tmass_nf_df is not None:
            mask &= base["has_tmass_nf"].fillna(False)
        if wise_nf_df is not None:
            mask &= base["has_wise_nf"].fillna(False)
        base = base[mask].reset_index(drop=True)

    if reorder:
        first = [c for c in ("Alias1","matching_ident","gaia_id","source_id") if c in base.columns]
        gaia_cols = [c for c in ("ra","dec","Gmag","BPmag","RPmag","parallax","e_parallax","pmRA","e_pmRA","pmDE","e_pmDE","ruwe",
                                 "teff","logg","feh") if c in base.columns]
        tmass_cols = [c for c in ("tmass_id","Jmag","Hmag","Kmag","e_Jmag","e_Hmag","e_Kmag","qual_tmass") if c in base.columns]
        wise_cols  = [c for c in ("wise_id","W1mag","W2mag","W3mag","W4mag","e_W1mag","e_W2mag","e_W3mag","e_W4mag","qual_wise") if c in base.columns]
        flags = [c for c in ("has_tmass_nf","has_wise_nf") if c in base.columns]
        ordered = first + gaia_cols + tmass_cols + wise_cols + flags
        remaining = [c for c in base.columns if c not in ordered]
        base = base[ordered + remaining]

    return base


# ----------------------------------------
# Select between duplicated data in querys
# ----------------------------------------

def nf_select_duplicates_by_radius(
    df: pd.DataFrame,
    *,
    primary_radius_arcsec: float = 1.5,
    secondary_radius_arcsec: float = 1.0,
) -> tuple[pd.DataFrame, dict]:
    """
    Aplica filtros por radio SOLO a los query_idx con duplicados:
      1) Para grupos con >1 candidatos: limitar a sep_arcsec <= 1.5"
      2) Si aún hay >1: limitar a sep_arcsec <= 1.0"
      3) Si aún hay >1 (o si un grupo queda en 0 tras los cortes): descartar ese query_idx completo.
    Los grupos con 1 solo candidato no se modifican.

    Devuelve:
      - DF resultante con <=1 fila por query_idx (algunos qid pueden desaparecer)
      - dict de métricas
    """
    if df is None or df.empty:
        return pd.DataFrame(), {
            "n_input": 0, "n_groups": 0, "n_singletons_kept": 0,
            "n_dupe_groups": 0, "n_resolved_at_1p5": 0, "n_resolved_at_1p0": 0,
            "n_dropped_ambiguous": 0, "n_dropped_zero_after_1p5": 0,
            "dropped_qids": []
        }

    required = {"query_idx", "sep_arcsec"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    q = pd.to_numeric(df["query_idx"], errors="coerce")
    df = df.copy()
    df["query_idx"] = q.astype("Int64")

    counts = df.groupby("query_idx", dropna=False).size()
    single_qids = set(counts[counts == 1].index.tolist())
    dupe_qids   = set(counts[counts > 1].index.tolist())

    singles = df[df["query_idx"].isin(single_qids)]
    dupes   = df[df["query_idx"].isin(dupe_qids)]

    singles_kept = singles.copy()

    eps = 1e-6
    d15 = dupes[dupes["sep_arcsec"] <= (primary_radius_arcsec + eps)].copy()

    after15_counts = d15.groupby("query_idx", dropna=False).size()
    zero_after_15 = [int(qid) for qid in dupe_qids if (qid not in after15_counts.index)]

    resolved_15_idx = []
    still_dupes_qids = []
    for qid, g in d15.groupby("query_idx", dropna=False):
        if len(g) == 1:
            resolved_15_idx.append(g.index[0])
        else:
            still_dupes_qids.append(qid)

    d10 = d15[d15["query_idx"].isin(still_dupes_qids)]
    d10 = d10[d10["sep_arcsec"] <= (secondary_radius_arcsec + eps)].copy()

    resolved_10_idx = []
    still_dupes_after_10 = []
    for qid, g in d10.groupby("query_idx", dropna=False):
        if len(g) == 1:
            resolved_10_idx.append(g.index[0])
        else:
            still_dupes_after_10.append(qid)

    dropped_qids = [int(qid) for qid in still_dupes_after_10 if pd.notna(qid)]
    dropped_qids.extend(zero_after_15)

    picks = []
    if not singles_kept.empty:
        picks.append(singles_kept)
    if resolved_15_idx:
        picks.append(d15.loc[resolved_15_idx])
    if resolved_10_idx:
        picks.append(d10.loc[resolved_10_idx])

    out = pd.concat(picks, ignore_index=True) if picks else pd.DataFrame(columns=df.columns)

    stats = {
        "n_input": len(df),
        "n_groups": len(counts),
        "n_singletons_kept": len(single_qids) if single_qids else 0,
        "n_dupe_groups": len(dupe_qids),
        "n_resolved_at_1p5": len(resolved_15_idx),
        "n_resolved_at_1p0": len(resolved_10_idx),
        "n_dropped_ambiguous": len(dropped_qids),
        "n_dropped_zero_after_1p5": len(zero_after_15),
        "dropped_qids": dropped_qids,
    }
    return out.reset_index(drop=True), stats
