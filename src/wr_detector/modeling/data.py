"""Model-ready dataset loading, leakage checks, and feature matrix construction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from wr_detector.config import load_yaml, resolve_path


FORBIDDEN_FEATURES = {
    "source_id",
    "wr_id",
    "simbad_main_id",
    "simbad_main_type",
    "simbad_other_types",
    "simbad_sp_type",
    "simbad_query_type",
    "sample_label",
    "dataset_family",
    "astrometric_subset",
    "tmass_quality",
    "wise_quality",
    "tmass_match_method",
    "wise_match_method",
}
FORBIDDEN_FEATURE_PREFIXES = ("color_locus_",)


def load_modeling_dataset(config: dict, variant: str) -> pd.DataFrame:
    filters = load_yaml(config["filters_config"])
    paths = load_yaml(config["paths_config"])
    dataset_cfg = config.get("modeling_dataset", {})
    reference_path, negative_path = _modeling_dataset_paths(filters, paths, variant, dataset_cfg)
    reference = pd.read_parquet(reference_path).copy()
    negative = pd.read_parquet(negative_path).copy()
    if bool(dataset_cfg.get("require_color_locus_keep", False)):
        reference = _filter_color_locus_keep(reference, reference_path)
        negative = _filter_color_locus_keep(negative, negative_path)
    reference["target"] = 1
    negative["target"] = 0
    reference["dataset_variant"] = variant
    negative["dataset_variant"] = variant
    return pd.concat([reference, negative], ignore_index=True, sort=False)


def _modeling_dataset_paths(filters: dict, paths: dict, variant: str, dataset_cfg: dict) -> tuple[Path, Path]:
    source = str(dataset_cfg.get("source", "base"))
    reference_dir = resolve_path(paths["processed_reference_dir"])
    simbad_dir = resolve_path(paths["processed_simbad_negative_dir"])
    if source == "base":
        return (
            reference_dir / filters["output_files"][variant],
            simbad_dir / filters["simbad_negative_output_files"][variant],
        )
    if source == "color_locus":
        color_locus = filters.get("color_locus", {})
        return (
            reference_dir / color_locus["reference_output_template"].format(variant=variant),
            simbad_dir / color_locus["simbad_negative_output_template"].format(variant=variant),
        )
    raise ValueError(f"Unsupported modeling_dataset source: {source}")


def _filter_color_locus_keep(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    if "color_locus_keep" not in df.columns:
        raise KeyError(f"Missing color_locus_keep in modeling dataset: {path}")
    return df[df["color_locus_keep"].astype(bool)].copy()


def build_model_matrix(
    df: pd.DataFrame,
    feature_columns: list[str],
    *,
    target_column: str = "target",
) -> tuple[pd.DataFrame, pd.Series]:
    _validate_feature_columns(df, feature_columns)
    model_df = df[feature_columns + [target_column]].dropna(axis=0, how="any")
    x = model_df[feature_columns].astype("float64")
    y = model_df[target_column].astype("int8")
    return x, y


def _validate_feature_columns(df: pd.DataFrame, feature_columns: list[str]) -> None:
    missing = [column for column in feature_columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing model feature columns: {missing}")
    forbidden = [
        column
        for column in feature_columns
        if column in FORBIDDEN_FEATURES or any(column.startswith(prefix) for prefix in FORBIDDEN_FEATURE_PREFIXES)
    ]
    if forbidden:
        raise ValueError(f"Forbidden leakage-prone model feature columns: {forbidden}")


def ensure_parent_dir(path: str | Path) -> Path:
    out = resolve_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out
