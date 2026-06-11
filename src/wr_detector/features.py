from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.metrics import r2_score

from wr_detector.config import load_yaml, resolve_path


BASE_COLUMNS = [
    "source_id",
    "wr_id",
    "ra",
    "dec",
    "G",
    "BP",
    "RP",
    "J",
    "H",
    "Ks",
    "W1",
    "W2",
    "W3",
    "W4",
    "parallax",
    "parallax_error",
    "parallax_over_error",
    "ruwe",
    "pmra",
    "pmdec",
    "tmass_quality",
    "wise_quality",
    "tmass_match_method",
    "wise_match_method",
]


COLOR_DEFINITIONS = {
    "BP_RP": ("BP", "RP"),
    "G_BP": ("G", "BP"),
    "G_RP": ("G", "RP"),
    "J_H": ("J", "H"),
    "H_K": ("H", "Ks"),
    "J_K": ("J", "Ks"),
    "W1_W2": ("W1", "W2"),
    "K_W1": ("Ks", "W1"),
    "G_K": ("G", "Ks"),
    "BP_J": ("BP", "J"),
}


COLOR_LOCUS_COLUMNS = [
    "color_locus_x",
    "color_locus_y",
    "color_locus_estimator",
    "color_locus_transform",
    "color_locus_slope",
    "color_locus_intercept",
    "color_locus_r2",
    "color_locus_sigma_mad",
    "color_locus_threshold",
    "color_locus_log_x",
    "color_locus_log_y",
    "color_locus_predicted_log_y",
    "color_locus_residual",
    "color_locus_normalized_residual",
    "color_locus_valid",
    "color_locus_outlier",
    "color_locus_keep",
]


TMASS_QUALITY_BANDS = {"J": 0, "H": 1, "Ks": 2}
WISE_QUALITY_BANDS = {"W1": 0, "W2": 1}


def load_reference_feature_frame(db_path: str | Path) -> pd.DataFrame:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute(
            """
            SELECT
                g.source_id,
                r.wr_id,
                g.ra,
                g.dec,
                g.G,
                g.BP,
                g.RP,
                t.J,
                t.H,
                t.Ks,
                w.W1,
                w.W2,
                w.W3,
                w.W4,
                g.parallax,
                g.parallax_error,
                g.parallax_over_error,
                g.ruwe,
                g.pmra,
                g.pmdec,
                t.tmass_quality,
                w.wise_quality,
                t.match_method AS tmass_match_method,
                w.match_method AS wise_match_method
            FROM wr_reference r
            INNER JOIN gaia_sources g USING (source_id)
            LEFT JOIN twomass_matches t USING (source_id)
            LEFT JOIN wise_matches w USING (source_id)
            """
        ).fetchdf()
    finally:
        con.close()
    df["sample_label"] = "wr"
    return add_color_features(df)


def load_simbad_negative_feature_frame(db_path: str | Path) -> pd.DataFrame:
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute(
            """
            SELECT
                g.source_id,
                s.simbad_main_id,
                s.simbad_main_type,
                s.simbad_other_types,
                s.simbad_sp_type,
                s.simbad_query_type,
                g.ra,
                g.dec,
                g.G,
                g.BP,
                g.RP,
                t.J,
                t.H,
                t.Ks,
                w.W1,
                w.W2,
                w.W3,
                w.W4,
                g.parallax,
                g.parallax_error,
                g.parallax_over_error,
                g.ruwe,
                g.pmra,
                g.pmdec,
                t.tmass_quality,
                w.wise_quality,
                t.match_method AS tmass_match_method,
                w.match_method AS wise_match_method
            FROM simbad_negative_sources s
            INNER JOIN gaia_sources g USING (source_id)
            LEFT JOIN twomass_matches t USING (source_id)
            LEFT JOIN wise_matches w USING (source_id)
            """
        ).fetchdf()
    finally:
        con.close()
    df["sample_label"] = "non_wr_simbad"
    return add_color_features(df)


def add_color_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for name, (left, right) in COLOR_DEFINITIONS.items():
        out[name] = out[left] - out[right]
    return out


def apply_photometry_filter(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    phot = filters["photometry"]
    if "families" in phot:
        family_name = next(iter(phot["families"]))
        return apply_photometry_family_filter(df, filters, family_name, phot["families"][family_name])

    required = phot["required_columns"]
    mask = pd.Series(True, index=df.index)
    for col in required:
        mask &= df[col].notna()
    mask &= df["tmass_quality"].astype("string").str.startswith(phot["twomass_quality_prefix"], na=False)
    mask &= df["wise_quality"].astype("string").str.startswith(phot["wise_quality_prefix"], na=False)
    return df[mask].reset_index(drop=True)


def apply_photometry_family_filter(
    df: pd.DataFrame,
    filters: dict,
    family_name: str,
    family_config: dict,
) -> pd.DataFrame:
    phot = filters["photometry"]
    required = phot["required_columns"]
    mask = pd.Series(True, index=df.index)
    for col in required:
        mask &= df[col].notna()

    tmass_bands = [band for band in TMASS_QUALITY_BANDS if band in required]
    wise_bands = [band for band in WISE_QUALITY_BANDS if band in required]
    mask &= _quality_mask(
        df["tmass_quality"],
        tmass_bands,
        TMASS_QUALITY_BANDS,
        family_config["twomass_allowed_qualities"],
    )
    mask &= _quality_mask(
        df["wise_quality"],
        wise_bands,
        WISE_QUALITY_BANDS,
        family_config["wise_allowed_qualities"],
    )

    out = df[mask].reset_index(drop=True).copy()
    out["dataset_family"] = family_name
    return out


def make_dataset_variants(df: pd.DataFrame, filters: dict) -> dict[str, pd.DataFrame]:
    phot = filters["photometry"]
    families = phot.get("families")
    if not families:
        photometry = apply_photometry_filter(df, filters)

        soft_cfg = filters["parallax_soft"]
        soft = _apply_parallax_soft_filter(photometry, soft_cfg)

        variants = {
            "photometry": _with_astrometric_subset(photometry, "photometry"),
            "parallax_soft": _with_astrometric_subset(soft, "parallax_soft"),
        }
        for name, cfg in filters["parallax_over_error_variants"].items():
            variants[name] = _with_astrometric_subset(_apply_poe_filter(soft, cfg), name)
        return variants

    variants: dict[str, pd.DataFrame] = {}
    for family_name, family_config in families.items():
        photometry = apply_photometry_family_filter(df, filters, family_name, family_config)
        soft = _apply_parallax_soft_filter(photometry, filters["parallax_soft"])

        variants[f"{family_name}_photometry"] = _with_astrometric_subset(photometry, "photometry")
        variants[f"{family_name}_parallax_soft"] = _with_astrometric_subset(soft, "parallax_soft")
        for subset_name, cfg in filters["parallax_over_error_variants"].items():
            variants[f"{family_name}_{subset_name}"] = _with_astrometric_subset(_apply_poe_filter(soft, cfg), subset_name)
    return variants


def _quality_mask(
    quality: pd.Series,
    bands: list[str],
    band_positions: dict[str, int],
    allowed_qualities: list[str],
) -> pd.Series:
    allowed = set(allowed_qualities)
    values = quality.astype("string")
    mask = pd.Series(True, index=quality.index)
    for band in bands:
        position = band_positions[band]
        mask &= values.str[position].isin(allowed).fillna(False)
    return mask


def _apply_parallax_soft_filter(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    return df[df["parallax"] > float(cfg["min_parallax"])].reset_index(drop=True)


def _apply_poe_filter(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    return df[
        (df["parallax"] > float(cfg["min_parallax"]))
        & (df["parallax_over_error"] >= float(cfg["min_parallax_over_error"]))
    ].reset_index(drop=True)


def _with_astrometric_subset(df: pd.DataFrame, subset_name: str) -> pd.DataFrame:
    out = df.reset_index(drop=True).copy()
    out["astrometric_subset"] = subset_name
    return out


def evaluate_color_locus_planes(
    df: pd.DataFrame,
    candidate_planes: list[dict[str, str]],
    *,
    min_positive_fraction: float,
    estimator: str = "huber",
    transform: str = "log10_positive",
    residual_quantile: float = 0.975,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for plane in candidate_planes:
        fit = fit_log_color_locus(
            df,
            x_color=plane["x"],
            y_color=plane["y"],
            min_positive_fraction=min_positive_fraction,
            estimator=estimator,
            transform=transform,
            residual_quantile=residual_quantile,
        )
        rows.append(fit)
    return pd.DataFrame(rows).sort_values(["valid", "r2"], ascending=[False, False]).reset_index(drop=True)


def fit_log_color_locus(
    df: pd.DataFrame,
    *,
    x_color: str,
    y_color: str,
    min_positive_fraction: float,
    estimator: str = "huber",
    transform: str = "log10_positive",
    residual_quantile: float = 0.975,
) -> dict[str, object]:
    validate_color_locus_inputs(df, x_color=x_color, y_color=y_color)
    valid_mask = make_color_transform_mask(df, x_color=x_color, y_color=y_color, transform=transform)
    valid_fraction = float(valid_mask.mean()) if len(valid_mask) else 0.0
    if valid_fraction < min_positive_fraction:
        return {
            "x": x_color,
            "y": y_color,
            "estimator": estimator,
            "transform": transform,
            "valid": False,
            "n": int(valid_mask.sum()),
            "positive_fraction": valid_fraction,
            "slope": np.nan,
            "intercept": np.nan,
            "r2": np.nan,
            "sigma_mad": np.nan,
            "residual_quantile": residual_quantile,
            "residual_quantile_threshold": np.nan,
        }

    x = transform_color_values(df.loc[valid_mask, x_color], transform=transform).to_numpy().reshape(-1, 1)
    y = transform_color_values(df.loc[valid_mask, y_color], transform=transform).to_numpy()
    model = _make_color_locus_estimator(estimator)
    model.fit(x, y)
    predicted = model.predict(x)
    residual = y - predicted
    abs_residual = np.abs(residual)
    return {
        "x": x_color,
        "y": y_color,
        "estimator": estimator,
        "transform": transform,
        "valid": True,
        "n": int(valid_mask.sum()),
        "positive_fraction": valid_fraction,
        "slope": float(model.coef_[0]),
        "intercept": float(model.intercept_),
        "r2": float(r2_score(y, predicted)),
        "sigma_mad": robust_sigma_mad(residual),
        "residual_quantile": residual_quantile,
        "residual_quantile_threshold": float(np.quantile(abs_residual, residual_quantile)),
    }


def annotate_color_locus(
    df: pd.DataFrame,
    fit: dict[str, object],
    *,
    residual_sigma_threshold: float,
    threshold_method: str = "sigma_mad",
) -> pd.DataFrame:
    if not bool(fit["valid"]):
        raise ValueError(f"Cannot annotate invalid color-locus fit: {fit['x']} -> {fit['y']}")

    out = df.copy()
    x_color = str(fit["x"])
    y_color = str(fit["y"])
    transform = str(fit.get("transform", "log10_positive"))
    valid_mask = make_color_transform_mask(out, x_color=x_color, y_color=y_color, transform=transform)
    log_x = pd.Series(np.nan, index=out.index, dtype="float64")
    log_y = pd.Series(np.nan, index=out.index, dtype="float64")
    predicted = pd.Series(np.nan, index=out.index, dtype="float64")
    residual = pd.Series(np.nan, index=out.index, dtype="float64")
    normalized = pd.Series(np.nan, index=out.index, dtype="float64")

    log_x.loc[valid_mask] = transform_color_values(out.loc[valid_mask, x_color], transform=transform)
    log_y.loc[valid_mask] = transform_color_values(out.loc[valid_mask, y_color], transform=transform)
    predicted.loc[valid_mask] = float(fit["intercept"]) + float(fit["slope"]) * log_x.loc[valid_mask]
    residual.loc[valid_mask] = log_y.loc[valid_mask] - predicted.loc[valid_mask]

    sigma = float(fit["sigma_mad"])
    if sigma > 0:
        normalized.loc[valid_mask] = residual.loc[valid_mask] / sigma

    if threshold_method == "sigma_mad":
        threshold = residual_sigma_threshold * sigma
    elif threshold_method == "empirical_quantile":
        threshold = float(fit["residual_quantile_threshold"])
    else:
        raise ValueError(f"Unsupported color-locus threshold method: {threshold_method}")
    outlier = valid_mask & residual.abs().gt(threshold)

    out["color_locus_x"] = x_color
    out["color_locus_y"] = y_color
    out["color_locus_estimator"] = str(fit["estimator"])
    out["color_locus_transform"] = transform
    out["color_locus_slope"] = float(fit["slope"])
    out["color_locus_intercept"] = float(fit["intercept"])
    out["color_locus_r2"] = float(fit["r2"])
    out["color_locus_sigma_mad"] = sigma
    out["color_locus_threshold"] = threshold
    out["color_locus_log_x"] = log_x
    out["color_locus_log_y"] = log_y
    out["color_locus_predicted_log_y"] = predicted
    out["color_locus_residual"] = residual
    out["color_locus_normalized_residual"] = normalized
    out["color_locus_valid"] = valid_mask
    out["color_locus_outlier"] = outlier
    out["color_locus_keep"] = valid_mask & ~outlier
    return out


def annotate_color_locus_planes(
    df: pd.DataFrame,
    fits: pd.DataFrame,
    *,
    residual_sigma_threshold: float,
    threshold_method: str = "sigma_mad",
    aggregate_min_outlier_planes: int = 1,
) -> pd.DataFrame:
    valid_fits = fits[fits["valid"]]
    if valid_fits.empty:
        raise ValueError("No valid color-locus fits available.")

    out = df.copy()
    plane_names: list[str] = []
    valid_columns: list[str] = []
    outlier_columns: list[str] = []
    for fit in valid_fits.to_dict(orient="records"):
        annotated = annotate_color_locus(
            out,
            fit,
            residual_sigma_threshold=residual_sigma_threshold,
            threshold_method=threshold_method,
        )
        plane_name = _plane_column_name(str(fit["x"]), str(fit["y"]))
        plane_names.append(plane_name)
        for source, suffix in [
            ("color_locus_log_x", "log_x"),
            ("color_locus_log_y", "log_y"),
            ("color_locus_predicted_log_y", "predicted_log_y"),
            ("color_locus_residual", "residual"),
            ("color_locus_normalized_residual", "normalized_residual"),
            ("color_locus_valid", "valid"),
            ("color_locus_outlier", "outlier"),
            ("color_locus_keep", "keep"),
        ]:
            out[f"color_locus_{plane_name}_{suffix}"] = annotated[source]
        out[f"color_locus_{plane_name}_slope"] = float(fit["slope"])
        out[f"color_locus_{plane_name}_intercept"] = float(fit["intercept"])
        out[f"color_locus_{plane_name}_r2"] = float(fit["r2"])
        out[f"color_locus_{plane_name}_sigma_mad"] = float(fit["sigma_mad"])
        out[f"color_locus_{plane_name}_threshold"] = float(annotated["color_locus_threshold"].iloc[0])
        valid_columns.append(f"color_locus_{plane_name}_valid")
        outlier_columns.append(f"color_locus_{plane_name}_outlier")

    out["color_locus_x"] = "multiple"
    out["color_locus_y"] = "multiple"
    out["color_locus_estimator"] = str(valid_fits.iloc[0]["estimator"])
    out["color_locus_transform"] = str(valid_fits.iloc[0]["transform"])
    out["color_locus_slope"] = np.nan
    out["color_locus_intercept"] = np.nan
    out["color_locus_planes"] = ",".join(plane_names)
    out["color_locus_r2"] = float(valid_fits["r2"].mean())
    out["color_locus_sigma_mad"] = float(valid_fits["sigma_mad"].mean())
    out["color_locus_threshold_method"] = threshold_method
    out["color_locus_threshold"] = float(residual_sigma_threshold)
    out["color_locus_log_x"] = np.nan
    out["color_locus_log_y"] = np.nan
    out["color_locus_predicted_log_y"] = np.nan
    out["color_locus_residual"] = np.nan
    out["color_locus_normalized_residual"] = np.nan
    out["color_locus_residual_quantile"] = float(valid_fits["residual_quantile"].iloc[0])
    out["color_locus_aggregate_min_outlier_planes"] = int(aggregate_min_outlier_planes)
    out["color_locus_valid"] = out[valid_columns].all(axis=1)
    out["color_locus_outlier_plane_count"] = out[outlier_columns].sum(axis=1)
    out["color_locus_outlier"] = out["color_locus_outlier_plane_count"].ge(aggregate_min_outlier_planes)
    out["color_locus_keep"] = out["color_locus_valid"] & ~out["color_locus_outlier"]
    return out


def export_color_locus_dataset(filters_config_path: str | Path) -> dict[str, object]:
    return export_color_locus_datasets(filters_config_path)


def export_color_locus_datasets(filters_config_path: str | Path) -> dict[str, object]:
    filters_config = load_yaml(filters_config_path)
    paths = load_yaml(filters_config["paths_config"])
    color_locus_config = filters_config["color_locus"]

    reference_dir = resolve_path(paths["processed_reference_dir"])
    simbad_dir = resolve_path(paths["processed_simbad_negative_dir"])
    variants = color_locus_config.get("dataset_variants")
    if variants is None:
        variants = [color_locus_config.get("base_dataset", "poe_3")]

    outputs: dict[str, object] = {"variants": {}}
    for variant in variants:
        reference_path = reference_dir / filters_config["output_files"][variant]
        simbad_path = simbad_dir / filters_config["simbad_negative_output_files"][variant]
        reference = pd.read_parquet(reference_path)
        simbad = pd.read_parquet(simbad_path)
        result = _export_color_locus_variant(
            variant=variant,
            reference=reference,
            simbad=simbad,
            reference_dir=reference_dir,
            simbad_dir=simbad_dir,
            color_locus_config=color_locus_config,
        )
        outputs["variants"][variant] = result
    return outputs


def _export_color_locus_variant(
    *,
    variant: str,
    reference: pd.DataFrame,
    simbad: pd.DataFrame,
    reference_dir: Path,
    simbad_dir: Path,
    color_locus_config: dict,
) -> dict[str, object]:
    candidate_fits = evaluate_color_locus_planes(
        reference,
        color_locus_config["candidate_planes"],
        min_positive_fraction=float(color_locus_config["min_positive_fraction"]),
        estimator=color_locus_config.get("estimator", "huber"),
        transform=color_locus_config.get("transform", "signed_log1p"),
        residual_quantile=float(color_locus_config.get("residual_quantile", 0.975)),
    )
    valid_fits = candidate_fits[candidate_fits["valid"]]
    if valid_fits.empty:
        raise ValueError(f"No valid color-locus candidate plane met the configured threshold for {variant}.")

    residual_sigma_threshold = float(color_locus_config["residual_sigma_threshold"])
    threshold_method = color_locus_config.get("threshold_method", "sigma_mad")
    aggregate_min_outlier_planes = int(color_locus_config.get("aggregate_min_outlier_planes", 1))

    if color_locus_config.get("selection_policy", "best_plane") == "all_planes":
        reference_annotated = annotate_color_locus_planes(
            reference,
            valid_fits,
            residual_sigma_threshold=residual_sigma_threshold,
            threshold_method=threshold_method,
            aggregate_min_outlier_planes=aggregate_min_outlier_planes,
        )
        simbad_annotated = annotate_color_locus_planes(
            simbad,
            valid_fits,
            residual_sigma_threshold=residual_sigma_threshold,
            threshold_method=threshold_method,
            aggregate_min_outlier_planes=aggregate_min_outlier_planes,
        )
        selected_plane = "all configured valid planes"
    else:
        selected_fit = valid_fits.iloc[0].to_dict()
        reference_annotated = annotate_color_locus(
            reference,
            selected_fit,
            residual_sigma_threshold=residual_sigma_threshold,
            threshold_method=threshold_method,
        )
        simbad_annotated = annotate_color_locus(
            simbad,
            selected_fit,
            residual_sigma_threshold=residual_sigma_threshold,
            threshold_method=threshold_method,
        )
        selected_plane = f"{selected_fit['x']} -> {selected_fit['y']}"

    reference_out = reference_dir / color_locus_config["reference_output_template"].format(variant=variant)
    simbad_out = simbad_dir / color_locus_config["simbad_negative_output_template"].format(variant=variant)
    reference_annotated.to_parquet(reference_out, index=False)
    simbad_annotated.to_parquet(simbad_out, index=False)
    return {
        "reference_rows": len(reference_annotated),
        "simbad_negative_rows": len(simbad_annotated),
        "reference_output_path": str(reference_out),
        "simbad_negative_output_path": str(simbad_out),
        "selected_plane": selected_plane,
        "outliers_reference": int(reference_annotated["color_locus_outlier"].sum()),
        "outliers_simbad_negative": int(simbad_annotated["color_locus_outlier"].sum()),
        "kept_reference": int(reference_annotated["color_locus_keep"].sum()),
        "kept_simbad_negative": int(simbad_annotated["color_locus_keep"].sum()),
        "candidate_fits": candidate_fits.to_dict(orient="records"),
    }


def validate_color_locus_inputs(df: pd.DataFrame, *, x_color: str, y_color: str) -> None:
    missing = [col for col in [x_color, y_color] if col not in df.columns]
    if missing:
        raise KeyError(f"Missing color columns for color-locus fit: {missing}")


def make_positive_color_mask(df: pd.DataFrame, *, x_color: str, y_color: str) -> pd.Series:
    return df[x_color].notna() & df[y_color].notna() & df[x_color].gt(0) & df[y_color].gt(0)


def make_color_transform_mask(df: pd.DataFrame, *, x_color: str, y_color: str, transform: str) -> pd.Series:
    base = df[x_color].notna() & df[y_color].notna()
    if transform == "log10_positive":
        return base & df[x_color].gt(0) & df[y_color].gt(0)
    if transform == "signed_log1p":
        return base
    raise ValueError(f"Unsupported color transform: {transform}")


def transform_color_values(values: pd.Series, *, transform: str) -> pd.Series:
    if transform == "log10_positive":
        return np.log10(values)
    if transform == "signed_log1p":
        return np.sign(values) * np.log1p(np.abs(values))
    raise ValueError(f"Unsupported color transform: {transform}")


def inverse_transform_color_values(values: pd.Series | np.ndarray, *, transform: str) -> pd.Series | np.ndarray:
    if transform == "log10_positive":
        return 10**values
    if transform == "signed_log1p":
        return np.sign(values) * np.expm1(np.abs(values))
    raise ValueError(f"Unsupported color transform: {transform}")


def robust_sigma_mad(values: np.ndarray | pd.Series) -> float:
    values = np.asarray(values, dtype="float64")
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad > 0:
        return float(1.4826 * mad)
    return float(np.std(values, ddof=0))


def _make_color_locus_estimator(estimator: str) -> HuberRegressor | LinearRegression:
    if estimator == "huber":
        return HuberRegressor()
    if estimator == "linear":
        return LinearRegression()
    raise ValueError(f"Unsupported color-locus estimator: {estimator}")


def _plane_column_name(x_color: str, y_color: str) -> str:
    return f"{x_color}__{y_color}".replace("-", "_")


def export_reference_datasets(filters_config_path: str | Path) -> dict[str, int]:
    filters_config = load_yaml(filters_config_path)
    paths = load_yaml(filters_config["paths_config"])
    db_path = resolve_path(paths["wr_reference_db"])
    out_dir = resolve_path(paths["processed_reference_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    full = load_reference_feature_frame(db_path)
    variants = make_dataset_variants(full, filters_config)
    counts: dict[str, int] = {}
    for name, frame in variants.items():
        filename = filters_config["output_files"][name]
        frame.to_parquet(out_dir / filename, index=False)
        counts[name] = len(frame)
    return counts


def export_simbad_negative_datasets(filters_config_path: str | Path) -> dict[str, int]:
    filters_config = load_yaml(filters_config_path)
    paths = load_yaml(filters_config["paths_config"])
    db_path = resolve_path(paths["simbad_negative_db"])
    out_dir = resolve_path(paths["processed_simbad_negative_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    full = load_simbad_negative_feature_frame(db_path)
    variants = make_dataset_variants(full, filters_config)
    counts: dict[str, int] = {}
    for name, frame in variants.items():
        filename = filters_config["simbad_negative_output_files"][name]
        frame.to_parquet(out_dir / filename, index=False)
        counts[name] = len(frame)
    return counts
