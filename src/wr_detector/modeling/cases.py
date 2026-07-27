"""Case-level review queries over synchronized training history.

These loaders join per-source ``model_predictions`` rows with the WR
reference and SIMBAD negative databases so that the Model Explorer can
review individual holdout cases (missed WR, false positives, top
candidates) and compute aggregate statistics such as recall by WR
subtype or false-positive composition by SIMBAD object type.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord

from wr_detector.config import load_yaml, resolve_path
from wr_detector.modeling.explorer import explorer_db_path


CASE_SPLITS = ["holdout", "train_oof"]

CASE_KINDS = {
    "true_positive": "WR ranked above threshold",
    "false_negative": "WR ranked below threshold",
    "false_positive": "negative ranked above threshold",
    "true_negative": "negative ranked below threshold",
}

DIAGNOSTIC_BASES = {
    "review_budget": "Review budget (top-K)",
    "operating_threshold": "Operating threshold",
}

DIAGNOSTIC_STATES = {
    "review_budget": [
        "Background",
        "Contaminant @K",
        "WR outside @K",
        "WR recovered @K",
    ],
    "operating_threshold": [
        "True negative",
        "False positive",
        "False negative",
        "True positive",
    ],
}


def reference_db_paths(config_path: str | Path = "configs/models.yaml") -> dict[str, Path | None]:
    """Resolve WR reference and SIMBAD negative DB paths from the models config."""
    config = load_yaml(config_path)
    paths_config_path = config.get("paths_config")
    paths = load_yaml(paths_config_path) if paths_config_path else {}
    wr_db = paths.get("wr_reference_db")
    negative_db = paths.get("simbad_negative_db")
    return {
        "wr": resolve_path(wr_db) if wr_db else None,
        "negative": resolve_path(negative_db) if negative_db else None,
    }


def load_case_predictions(
    config_path: str | Path = "configs/models.yaml",
    *,
    run_id: str,
    result_id: str,
    split: str = "holdout",
) -> pd.DataFrame:
    """Load per-source predictions for one model, enriched with source identity.

    Returns one row per scored source ordered by descending score with a
    1-based ``rank`` column. When the reference databases are available the
    rows include WR identity (``wr_number``, ``spectral_type``,
    ``wr_subtype``), SIMBAD identity (``object_name``,
    ``simbad_main_type``) and Gaia/2MASS/WISE photometry-derived colors.
    """
    db_path = explorer_db_path(config_path)
    with duckdb.connect(str(db_path), read_only=True) as con:
        attached = _attach_reference_dbs(con, reference_db_paths(config_path))
        query = _case_query(attached)
        cases = con.execute(query, [run_id, result_id, split]).fetchdf()
    return _finalize_cases(cases)


def load_case_overlap(
    config_path: str | Path = "configs/models.yaml",
    *,
    run_id: str,
    split: str = "holdout",
    kind: str = "false_positive",
    top_k: int = 100,
) -> pd.DataFrame:
    """Aggregate recurring cases across every model of a run.

    ``kind='false_positive'`` returns negatives appearing inside the top
    ``top_k`` ranked candidates; ``kind='missed_wr'`` returns WR falling
    outside the top ``top_k``. ``n_models`` counts how many of the run's
    models flag each source, so persistent contaminants and persistently
    missed WR float to the top.
    """
    if kind not in {"false_positive", "missed_wr"}:
        raise ValueError(f"Unsupported overlap kind: {kind}")
    db_path = explorer_db_path(config_path)
    with duckdb.connect(str(db_path), read_only=True) as con:
        attached = _attach_reference_dbs(con, reference_db_paths(config_path))
        condition = (
            "target = 0 AND rank <= ?" if kind == "false_positive" else "target = 1 AND rank > ?"
        )
        identity_select, identity_joins = _identity_clauses(attached)
        query = f"""
            WITH ranked AS (
                SELECT
                    result_id,
                    source_id,
                    target,
                    score,
                    ROW_NUMBER() OVER (PARTITION BY result_id ORDER BY score DESC) AS rank
                FROM model_predictions
                WHERE run_id = ? AND split = ?
            ),
            totals AS (
                SELECT COUNT(DISTINCT result_id) AS n_total_models FROM ranked
            ),
            flagged AS (
                SELECT source_id, target, score, rank FROM ranked WHERE {condition}
            ),
            grouped AS (
                SELECT
                    source_id,
                    ANY_VALUE(target) AS target,
                    COUNT(*) AS n_models,
                    AVG(score) AS score_mean,
                    MIN(rank) AS best_rank,
                    MAX(rank) AS worst_rank
                FROM flagged
                GROUP BY source_id
            )
            SELECT
                grouped.*,
                totals.n_total_models,
                grouped.n_models / totals.n_total_models AS model_share
                {identity_select}
            FROM grouped
            CROSS JOIN totals
            {identity_joins}
            ORDER BY n_models DESC, score_mean DESC
        """
        overlap = con.execute(query, [run_id, split, top_k]).fetchdf()
    return _with_case_identity(overlap)


def wr_broad_subtype(spectral_type: object) -> str:
    """Map a WR spectral type string to a broad subtype family."""
    if spectral_type is None or pd.isna(spectral_type):
        return "unknown"
    value = str(spectral_type).strip().upper()
    has_wn = "WN" in value
    has_wc = "WC" in value
    if has_wn and has_wc:
        return "WN/WC"
    if has_wn:
        return "WN"
    if has_wc:
        return "WC"
    if "WO" in value:
        return "WO"
    return "other"


def subtype_recovery(cases: pd.DataFrame, *, ks: list[int] | None = None) -> pd.DataFrame:
    """Per-WR-subtype recovery at each candidate budget for one model's cases."""
    ks = ks or [10, 50, 100, 500, 1000]
    wr = cases[cases["target"].eq(1)]
    if wr.empty or "wr_subtype" not in wr.columns:
        return pd.DataFrame(columns=["wr_subtype", "total", *[f"recovered_at_{k}" for k in ks]])
    rows = []
    for subtype, group in wr.groupby("wr_subtype", dropna=False):
        row: dict[str, object] = {"wr_subtype": subtype, "total": len(group)}
        for k in ks:
            recovered = int(group["rank"].le(k).sum())
            row[f"recovered_at_{k}"] = recovered
            row[f"recovered_at_{k}_pct"] = recovered / len(group) if len(group) else pd.NA
        rows.append(row)
    return pd.DataFrame(rows).sort_values("total", ascending=False).reset_index(drop=True)


def false_positive_composition(cases: pd.DataFrame, *, top_k: int = 100) -> pd.DataFrame:
    """SIMBAD object-type composition of negatives inside the top ``top_k``."""
    if "simbad_main_type" not in cases.columns:
        return pd.DataFrame(columns=["simbad_main_type", "count", "share"])
    contaminants = cases[cases["target"].eq(0) & cases["rank"].le(top_k)]
    if contaminants.empty:
        return pd.DataFrame(columns=["simbad_main_type", "count", "share"])
    counts = (
        contaminants["simbad_main_type"]
        .fillna("unknown")
        .value_counts()
        .rename_axis("simbad_main_type")
        .reset_index(name="count")
    )
    counts["share"] = counts["count"] / counts["count"].sum()
    return counts


def precision_recall_points(cases: pd.DataFrame, *, max_points: int = 400) -> pd.DataFrame:
    """Precision-recall curve points computed from per-source scores.

    Returns one row per retained rank with ``recall``, ``precision`` and the
    ``score`` cutoff, downsampled evenly to ``max_points`` rows.
    """
    ranked = _ranked_targets(cases)
    if ranked.empty:
        return pd.DataFrame(columns=["recall", "precision", "score"])
    positives = int(ranked["target"].sum())
    if positives == 0:
        return pd.DataFrame(columns=["recall", "precision", "score"])
    cumulative_tp = ranked["target"].cumsum()
    counts = pd.Series(range(1, len(ranked) + 1), index=ranked.index)
    curve = pd.DataFrame(
        {
            "recall": cumulative_tp / positives,
            "precision": cumulative_tp / counts,
            "score": ranked["score"],
        }
    )
    return _downsample(curve, max_points)


def roc_points(cases: pd.DataFrame, *, max_points: int = 400) -> pd.DataFrame:
    """ROC curve points (``fpr``, ``tpr``, ``score``) from per-source scores."""
    ranked = _ranked_targets(cases)
    if ranked.empty:
        return pd.DataFrame(columns=["fpr", "tpr", "score"])
    positives = int(ranked["target"].sum())
    negatives = len(ranked) - positives
    if positives == 0 or negatives == 0:
        return pd.DataFrame(columns=["fpr", "tpr", "score"])
    cumulative_tp = ranked["target"].cumsum()
    counts = pd.Series(range(1, len(ranked) + 1), index=ranked.index)
    curve = pd.DataFrame(
        {
            "fpr": (counts - cumulative_tp) / negatives,
            "tpr": cumulative_tp / positives,
            "score": ranked["score"],
        }
    )
    return _downsample(curve, max_points)


def _ranked_targets(cases: pd.DataFrame) -> pd.DataFrame:
    ranked = cases.dropna(subset=["score"]).sort_values("score", ascending=False)
    return ranked[["score", "target"]].astype({"target": int}).reset_index(drop=True)


def _downsample(curve: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(curve) <= max_points:
        return curve
    step = max(1, len(curve) // max_points)
    sampled = curve.iloc[::step]
    if sampled.index[-1] != curve.index[-1]:
        sampled = pd.concat([sampled, curve.tail(1)])
    return sampled.reset_index(drop=True)


def case_confusion(cases: pd.DataFrame) -> dict[str, int]:
    """Threshold-based confusion counts for one model's cases."""
    target = cases["target"].astype(int)
    predicted = cases["predicted"].fillna(0).astype(int)
    return {
        "true_positive": int(((target == 1) & (predicted == 1)).sum()),
        "false_negative": int(((target == 1) & (predicted == 0)).sum()),
        "false_positive": int(((target == 0) & (predicted == 1)).sum()),
        "true_negative": int(((target == 0) & (predicted == 0)).sum()),
    }


def classify_cases(
    cases: pd.DataFrame,
    *,
    basis: str = "review_budget",
    top_k: int = 100,
) -> pd.DataFrame:
    """Add an explicit diagnostic state without conflating rank and threshold.

    ``review_budget`` describes whether a source falls inside a top-K manual
    review list. ``operating_threshold`` uses the model's synchronized
    threshold decision and therefore supports conventional TP/FP/FN/TN names.
    """
    if basis not in DIAGNOSTIC_BASES:
        raise ValueError(f"Unsupported diagnostic basis: {basis}")
    out = cases.copy()
    target = out["target"].fillna(0).astype(int)
    if basis == "review_budget":
        selected = out["rank"].le(top_k)
        out["diagnostic_state"] = np.select(
            [
                target.eq(1) & selected,
                target.eq(0) & selected,
                target.eq(1) & ~selected,
            ],
            ["WR recovered @K", "Contaminant @K", "WR outside @K"],
            default="Background",
        )
    else:
        if "predicted" in out.columns:
            predicted = out["predicted"].fillna(0).astype(int)
        else:
            predicted = (
                out["score"].astype(float)
                >= out["threshold"].astype(float)
            ).astype(int)
        out["diagnostic_state"] = np.select(
            [
                target.eq(1) & predicted.eq(1),
                target.eq(0) & predicted.eq(1),
                target.eq(1) & predicted.eq(0),
            ],
            ["True positive", "False positive", "False negative"],
            default="True negative",
        )
    out["diagnostic_basis"] = basis
    return out


def add_case_spatial_coordinates(
    cases: pd.DataFrame,
    *,
    min_parallax_over_error: float = 2.0,
    max_distance_kpc: float = 15.0,
    sun_distance_kpc: float = 8.122,
) -> pd.DataFrame:
    """Add Galactic sky and qualified top-down plane coordinates.

    Galactic longitude/latitude require only RA/Dec. Plane positions use the
    transparent approximation ``distance_kpc = 1 / parallax_mas`` and are
    populated only for positive parallaxes meeting the requested S/N floor and
    distance cap.
    """
    out = cases.copy()
    spatial_columns = [
        "galactic_l",
        "galactic_b",
        "polar_x",
        "polar_y",
        "mollweide_x",
        "mollweide_y",
        "distance_kpc",
        "galactocentric_x_kpc",
        "galactocentric_y_kpc",
    ]
    for column in spatial_columns:
        out[column] = np.nan
    out["distance_plotted"] = False
    if not {"ra", "dec"}.issubset(out.columns):
        return out

    sky_mask = out["ra"].notna() & out["dec"].notna()
    if sky_mask.any():
        coords = SkyCoord(
            ra=out.loc[sky_mask, "ra"].to_numpy(dtype=float) * u.deg,
            dec=out.loc[sky_mask, "dec"].to_numpy(dtype=float) * u.deg,
            frame="icrs",
        ).galactic
        longitude = coords.l.wrap_at(360 * u.deg).degree
        latitude = coords.b.degree
        radius = 90.0 - latitude
        theta = np.deg2rad(longitude)
        out.loc[sky_mask, "galactic_l"] = longitude
        out.loc[sky_mask, "galactic_b"] = latitude
        out.loc[sky_mask, "polar_x"] = radius * np.sin(theta)
        out.loc[sky_mask, "polar_y"] = radius * np.cos(theta)
        # Standard Mollweide projection with Galactic longitude increasing
        # towards the left, as customary in astronomical all-sky maps.
        from wr_detector.modeling.case_visualization import mollweide_project

        mollweide_x, mollweide_y = mollweide_project(longitude, latitude)
        out.loc[sky_mask, "mollweide_x"] = mollweide_x
        out.loc[sky_mask, "mollweide_y"] = mollweide_y

    required = {"parallax", "parallax_over_error", "galactic_l", "galactic_b"}
    if not required.issubset(out.columns):
        return out
    distance = 1.0 / out["parallax"].astype(float)
    distance_mask = (
        out["parallax"].gt(0)
        & out["parallax_over_error"].ge(float(min_parallax_over_error))
        & distance.le(float(max_distance_kpc))
        & out["galactic_l"].notna()
    )
    if not distance_mask.any():
        return out

    longitude = np.deg2rad(out.loc[distance_mask, "galactic_l"].astype(float))
    latitude = np.deg2rad(out.loc[distance_mask, "galactic_b"].astype(float))
    qualified_distance = distance.loc[distance_mask]
    plane_distance = qualified_distance * np.cos(latitude)
    out.loc[distance_mask, "distance_kpc"] = qualified_distance
    out.loc[distance_mask, "galactocentric_x_kpc"] = (
        float(sun_distance_kpc) - plane_distance * np.cos(longitude)
    )
    out.loc[distance_mask, "galactocentric_y_kpc"] = (
        plane_distance * np.sin(longitude)
    )
    out.loc[distance_mask, "distance_plotted"] = True
    return out


def _attach_reference_dbs(
    con: duckdb.DuckDBPyConnection, paths: dict[str, Path | None]
) -> dict[str, bool]:
    attached = {"wr": False, "negative": False}
    aliases = {"wr": "wr_ref", "negative": "neg_ref"}
    for key, db_path in paths.items():
        if db_path is not None and db_path.exists():
            con.execute(f"ATTACH '{db_path.as_posix()}' AS {aliases[key]} (READ_ONLY)")
            attached[key] = True
    return attached


def _identity_clauses(attached: dict[str, bool]) -> tuple[str, str]:
    select_parts = []
    join_parts = []
    if attached["wr"]:
        select_parts.append(
            ", wr_info.wr_number, wr_info.spectral_type"
        )
        join_parts.append(
            """
            LEFT JOIN (
                SELECT
                    source_id,
                    ANY_VALUE("WR#") AS wr_number,
                    ANY_VALUE("Spectral Type") AS spectral_type
                FROM wr_ref.wr_reference
                WHERE source_id IS NOT NULL
                GROUP BY source_id
            ) AS wr_info USING (source_id)
            """
        )
    if attached["negative"]:
        select_parts.append(
            ", neg_info.simbad_main_id, neg_info.simbad_main_type, neg_info.simbad_sp_type"
        )
        join_parts.append(
            """
            LEFT JOIN (
                SELECT
                    source_id,
                    ANY_VALUE(simbad_main_id) AS simbad_main_id,
                    ANY_VALUE(simbad_main_type) AS simbad_main_type,
                    ANY_VALUE(simbad_sp_type) AS simbad_sp_type
                FROM neg_ref.simbad_negative_sources
                WHERE source_id IS NOT NULL
                GROUP BY source_id
            ) AS neg_info USING (source_id)
            """
        )
    return "".join(select_parts), "".join(join_parts)


def _photometry_clauses(attached: dict[str, bool]) -> tuple[str, str]:
    sources = []
    if attached["wr"]:
        sources.append("wr_ref")
    if attached["negative"]:
        sources.append("neg_ref")
    if not sources:
        return "", ""

    gaia_union = " UNION ALL ".join(
        f"SELECT source_id, ra, dec, G, BP, RP, parallax, parallax_over_error, ruwe FROM {alias}.gaia_sources"
        for alias in sources
    )
    tmass_union = " UNION ALL ".join(
        f"SELECT source_id, J, H, Ks, tmass_quality FROM {alias}.twomass_matches"
        for alias in sources
    )
    wise_union = " UNION ALL ".join(
        f"SELECT source_id, W1, W2, wise_quality FROM {alias}.wise_matches"
        for alias in sources
    )
    select = (
        ", gaia.ra, gaia.dec, gaia.G, gaia.BP, gaia.RP, gaia.parallax,"
        " gaia.parallax_over_error, gaia.ruwe,"
        " tmass.J, tmass.H, tmass.Ks, tmass.tmass_quality,"
        " wise.W1, wise.W2, wise.wise_quality"
    )
    joins = f"""
        LEFT JOIN (
            SELECT * FROM ({gaia_union})
            QUALIFY ROW_NUMBER() OVER (PARTITION BY source_id ORDER BY source_id) = 1
        ) AS gaia USING (source_id)
        LEFT JOIN (
            SELECT * FROM ({tmass_union})
            QUALIFY ROW_NUMBER() OVER (PARTITION BY source_id ORDER BY source_id) = 1
        ) AS tmass USING (source_id)
        LEFT JOIN (
            SELECT * FROM ({wise_union})
            QUALIFY ROW_NUMBER() OVER (PARTITION BY source_id ORDER BY source_id) = 1
        ) AS wise USING (source_id)
    """
    return select, joins


def _case_query(attached: dict[str, bool]) -> str:
    identity_select, identity_joins = _identity_clauses(attached)
    photometry_select, photometry_joins = _photometry_clauses(attached)
    return f"""
        WITH preds AS (
            SELECT source_id, target, score, predicted, threshold
            FROM model_predictions
            WHERE run_id = ? AND result_id = ? AND split = ?
        )
        SELECT
            preds.*
            {identity_select}
            {photometry_select}
        FROM preds
        {identity_joins}
        {photometry_joins}
        ORDER BY preds.score DESC
    """


def _finalize_cases(cases: pd.DataFrame) -> pd.DataFrame:
    cases = cases.copy()
    cases.insert(0, "rank", range(1, len(cases) + 1))
    if {"BP", "RP"}.issubset(cases.columns):
        cases["BP_RP"] = cases["BP"] - cases["RP"]
    if {"G", "BP"}.issubset(cases.columns):
        cases["G_BP"] = cases["G"] - cases["BP"]
    if {"G", "RP"}.issubset(cases.columns):
        cases["G_RP"] = cases["G"] - cases["RP"]
    if {"J", "H"}.issubset(cases.columns):
        cases["J_H"] = cases["J"] - cases["H"]
    if {"J", "Ks"}.issubset(cases.columns):
        cases["J_K"] = cases["J"] - cases["Ks"]
    if {"H", "Ks"}.issubset(cases.columns):
        cases["H_K"] = cases["H"] - cases["Ks"]
    if {"W1", "W2"}.issubset(cases.columns):
        cases["W1_W2"] = cases["W1"] - cases["W2"]
    return _with_case_identity(cases)


def _with_case_identity(cases: pd.DataFrame) -> pd.DataFrame:
    cases = cases.copy()
    if "spectral_type" in cases.columns:
        cases["wr_subtype"] = cases["spectral_type"].map(wr_broad_subtype)
        cases.loc[cases["target"].ne(1), "wr_subtype"] = pd.NA
    name = pd.Series(pd.NA, index=cases.index, dtype="object")
    if "wr_number" in cases.columns:
        wr_mask = cases["target"].eq(1) & cases["wr_number"].notna()
        name.loc[wr_mask] = "WR " + cases.loc[wr_mask, "wr_number"].astype(str)
    if "simbad_main_id" in cases.columns:
        neg_mask = name.isna() & cases["simbad_main_id"].notna()
        name.loc[neg_mask] = cases.loc[neg_mask, "simbad_main_id"].astype(str)
    fallback = name.isna() & cases["source_id"].notna()
    name.loc[fallback] = "Gaia DR3 " + cases.loc[fallback, "source_id"].astype("Int64").astype(str)
    cases["object_name"] = name
    return cases
