from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

from wr_detector.config import load_yaml, resolve_path
from wr_detector.modeling.data import ensure_parent_dir, load_modeling_dataset
from wr_detector.modeling.splits import make_global_holdout_mask


def reduce_negative_variants(
    config_path: str | Path = "configs/models.yaml",
    *,
    variants: list[str] | None = None,
    negative_ratios: list[int | str] | None = None,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = load_yaml(config_path)
    selected_variants = variants or list(config["dataset_variants"])
    selected_ratios = negative_ratios or configured_negative_ratios(config)
    summary_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    pool_audit_rows: list[dict[str, object]] = []
    for variant in selected_variants:
        for ratio in selected_ratios:
            reduced, summary, audit, pool_audit = reduce_negative_dataset(config, variant, negative_ratio=ratio)
            out_path = reduced_dataset_path(config, variant, negative_ratio=ratio)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            reduced.to_parquet(out_path, index=False)
            summary_rows.append({**summary, "output_path": str(out_path)})
            audit_rows.extend(audit)
            pool_audit_rows.extend(pool_audit)
            if verbose:
                print(
                    f"{variant} / {summary['negative_ratio_label']}: train WR={summary['train_wr']} "
                    f"train NEG {summary['train_negative_original']} -> {summary['train_negative_reduced']} "
                    f"holdout rows={summary['holdout_rows']}",
                    flush=True,
                )
    summary_df = pd.DataFrame(summary_rows)
    audit_df = pd.DataFrame(audit_rows)
    summary_path = ensure_parent_dir(config["outputs"]["reduction_summary"])
    audit_path = ensure_parent_dir(config["outputs"]["reduction_audit"])
    summary_df.to_csv(summary_path, index=False)
    audit_df.to_csv(audit_path, index=False)
    pool_audit_path = config["outputs"].get("negative_pool_audit")
    if pool_audit_path:
        pd.DataFrame(pool_audit_rows).to_csv(ensure_parent_dir(pool_audit_path), index=False)
    return summary_df, audit_df


def reduce_negative_dataset(
    config: dict,
    variant: str,
    *,
    negative_ratio: int | str | None = None,
) -> tuple[pd.DataFrame, dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    random_state = int(config.get("random_state", 42))
    reduction_cfg = config["negative_reduction"]
    ratio = normalize_negative_ratio(negative_ratio if negative_ratio is not None else reduction_cfg["negative_to_wr_ratio"])
    color_columns = list(reduction_cfg["columns"])
    dataset = load_modeling_dataset(config, variant)
    holdout = make_global_holdout_mask(
        dataset["source_id"],
        holdout_fraction=float(config["holdout_fraction"]),
        random_state=random_state,
        target=dataset["target"],
    )
    train = dataset.loc[~holdout].copy()
    holdout_df = dataset.loc[holdout].copy()
    train_wr = train[train["target"] == 1].copy()
    train_negative = train[train["target"] == 0].copy()
    target_negative = len(train_negative) if ratio == "all" else min(len(train_negative), int(ratio) * len(train_wr))
    sampled_negative = sample_representative_negatives(
        train_negative,
        target_n=target_negative,
        color_columns=color_columns,
        quantile_bins=int(reduction_cfg.get("quantile_bins", 4)),
        random_state=random_state,
    )
    calibration_negative = train_negative[~train_negative["source_id"].isin(sampled_negative["source_id"])].copy()
    train_wr["modeling_split"] = "train"
    sampled_negative["modeling_split"] = "train"
    calibration_negative["modeling_split"] = "threshold_calibration"
    holdout_df["modeling_split"] = "holdout"
    train_wr["reduction_keep"] = True
    sampled_negative["reduction_keep"] = True
    calibration_negative["reduction_keep"] = False
    holdout_df["reduction_keep"] = True
    reduced = pd.concat([train_wr, sampled_negative, calibration_negative, holdout_df], ignore_index=True, sort=False)
    summary = {
        "dataset_variant": variant,
        "negative_ratio": ratio,
        "negative_ratio_label": negative_ratio_label(ratio),
        "train_wr": int(len(train_wr)),
        "train_negative_original": int(len(train_negative)),
        "train_negative_reduced": int(len(sampled_negative)),
        "threshold_calibration_negative": int(len(calibration_negative)),
        "target_negative": int(target_negative),
        "holdout_rows": int(len(holdout_df)),
        "holdout_wr": int((holdout_df["target"] == 1).sum()),
        "holdout_negative": int((holdout_df["target"] == 0).sum()),
        "negative_to_wr_ratio": float(len(sampled_negative) / max(len(train_wr), 1)),
    }
    audit = audit_negative_reduction(
        train_negative,
        sampled_negative,
        variant=variant,
        color_columns=color_columns,
    )
    pool_audit = audit_negative_pools(
        sampled_negative,
        calibration_negative,
        holdout_df[holdout_df["target"] == 0],
        variant=variant,
        color_columns=color_columns,
    )
    return reduced, summary, audit, pool_audit


def sample_representative_negatives(
    negatives: pd.DataFrame,
    *,
    target_n: int,
    color_columns: list[str],
    quantile_bins: int,
    random_state: int,
) -> pd.DataFrame:
    if target_n >= len(negatives):
        return negatives.copy()
    if target_n <= 0:
        return negatives.iloc[0:0].copy()
    working = negatives.copy()
    working["_reduction_stratum"] = make_color_strata(working, color_columns=color_columns, quantile_bins=quantile_bins)
    counts = working["_reduction_stratum"].value_counts(dropna=False).rename("count").reset_index()
    counts["raw_quota"] = counts["count"] / counts["count"].sum() * target_n
    counts["quota"] = np.floor(counts["raw_quota"]).astype(int).clip(upper=counts["count"])
    remainder = int(target_n - counts["quota"].sum())
    if remainder > 0:
        counts["fraction"] = counts["raw_quota"] - np.floor(counts["raw_quota"])
        counts["capacity"] = counts["count"] - counts["quota"]
        for idx in counts[counts["capacity"] > 0].sort_values("fraction", ascending=False).index[:remainder]:
            counts.loc[idx, "quota"] += 1
    sampled_parts = []
    rng = np.random.default_rng(random_state)
    for row in counts.itertuples(index=False):
        quota = int(row.quota)
        if quota <= 0:
            continue
        stratum = getattr(row, "_0")
        group = working[working["_reduction_stratum"] == stratum]
        sampled_parts.append(group.sample(n=quota, random_state=int(rng.integers(0, 2**31 - 1))))
    sampled = pd.concat(sampled_parts, ignore_index=False) if sampled_parts else working.sample(n=target_n, random_state=random_state)
    if len(sampled) < target_n:
        missing = target_n - len(sampled)
        remaining = working.drop(index=sampled.index)
        sampled = pd.concat([sampled, remaining.sample(n=missing, random_state=random_state)], ignore_index=False)
    return sampled.drop(columns=["_reduction_stratum"]).sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def make_color_strata(df: pd.DataFrame, *, color_columns: list[str], quantile_bins: int) -> pd.Series:
    labels = []
    for column in color_columns:
        values = df[column]
        if values.notna().nunique() < 2:
            labels.append(pd.Series("all", index=df.index, dtype="string"))
            continue
        try:
            binned = pd.qcut(values, q=quantile_bins, labels=False, duplicates="drop")
        except ValueError:
            binned = pd.Series(np.nan, index=df.index)
        labels.append(binned.astype("Int64").astype("string").fillna("missing"))
    strata = labels[0]
    for label in labels[1:]:
        strata = strata.str.cat(label, sep="|")
    return strata


def audit_negative_reduction(
    original_negative: pd.DataFrame,
    reduced_negative: pd.DataFrame,
    *,
    variant: str,
    color_columns: list[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for column in color_columns:
        original = original_negative[column].dropna()
        reduced = reduced_negative[column].dropna()
        if original.empty or reduced.empty:
            ks_stat = np.nan
            wasserstein = np.nan
        else:
            ks_stat = float(ks_2samp(original, reduced).statistic)
            wasserstein = float(wasserstein_distance(original, reduced))
        rows.append(
            {
                "dataset_variant": variant,
                "color": column,
                "original_n": int(len(original)),
                "reduced_n": int(len(reduced)),
                "original_mean": float(original.mean()) if not original.empty else np.nan,
                "reduced_mean": float(reduced.mean()) if not reduced.empty else np.nan,
                "original_std": float(original.std()) if len(original) > 1 else np.nan,
                "reduced_std": float(reduced.std()) if len(reduced) > 1 else np.nan,
                "ks_statistic": ks_stat,
                "wasserstein_distance": wasserstein,
            }
        )
    return rows


def audit_negative_pools(
    train_negative: pd.DataFrame,
    calibration_negative: pd.DataFrame,
    holdout_negative: pd.DataFrame,
    *,
    variant: str,
    color_columns: list[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    pools = {
        "train_reduced_negative": train_negative,
        "threshold_calibration_negative": calibration_negative,
        "holdout_negative": holdout_negative,
    }
    reference = pd.concat([train_negative, calibration_negative, holdout_negative], ignore_index=True, sort=False)
    for column in color_columns:
        reference_values = reference[column].dropna()
        for pool_name, pool_df in pools.items():
            values = pool_df[column].dropna()
            if reference_values.empty or values.empty:
                ks_stat = np.nan
                wasserstein = np.nan
            else:
                ks_stat = float(ks_2samp(reference_values, values).statistic)
                wasserstein = float(wasserstein_distance(reference_values, values))
            rows.append(
                {
                    "dataset_variant": variant,
                    "pool": pool_name,
                    "color": column,
                    "n": int(len(values)),
                    "mean": float(values.mean()) if not values.empty else np.nan,
                    "std": float(values.std()) if len(values) > 1 else np.nan,
                    "p05": float(values.quantile(0.05)) if not values.empty else np.nan,
                    "p50": float(values.quantile(0.50)) if not values.empty else np.nan,
                    "p95": float(values.quantile(0.95)) if not values.empty else np.nan,
                    "ks_vs_all_negative": ks_stat,
                    "wasserstein_vs_all_negative": wasserstein,
                }
            )
    return rows


def configured_negative_ratios(config: dict) -> list[int | str]:
    reduction_cfg = config.get("negative_reduction", {})
    ratios = reduction_cfg.get("ratios")
    if ratios:
        return [normalize_negative_ratio(value) for value in ratios]
    return [normalize_negative_ratio(reduction_cfg.get("negative_to_wr_ratio", 10))]


def normalize_negative_ratio(value: int | str) -> int | str:
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"all", "all_train_negatives", "all_train_negative"}:
            return "all"
        if text.endswith("x"):
            text = text[:-1]
        return int(text)
    return int(value)


def negative_ratio_label(value: int | str) -> str:
    ratio = normalize_negative_ratio(value)
    return "all_train_negatives" if ratio == "all" else f"{int(ratio)}x"


def reduced_dataset_path(config: dict, variant: str, *, negative_ratio: int | str | None = None) -> Path:
    out_dir = resolve_path(config["outputs"]["modeling_data_dir"])
    template = config["outputs"]["reduced_dataset_template"]
    ratio = normalize_negative_ratio(negative_ratio if negative_ratio is not None else config["negative_reduction"]["negative_to_wr_ratio"])
    default_ratio = normalize_negative_ratio(config["negative_reduction"].get("negative_to_wr_ratio", 10))
    if ratio == default_ratio:
        return out_dir / template.format(variant=variant)
    stem = Path(template.format(variant=variant)).stem
    return out_dir / f"{stem}__neg_{negative_ratio_label(ratio)}.parquet"
