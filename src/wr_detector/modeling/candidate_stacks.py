"""Read-only evaluation of first-stage models paired with validation layers.

Candidate stacks are evaluated on synchronized first-stage holdout predictions.
The current second layer contributes subtype-aware WN/WC compatibility flags;
it does not replace the first-stage ranking or mutate any model artifact.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from joblib import load

from wr_detector.config import resolve_path
from wr_detector.modeling import cases as case_review
from wr_detector.modeling.explorer import load_run_results
from wr_detector.modeling.layers import ValidationLayer, list_layer_runs, load_layer_results
from wr_detector.modeling.second_layer import add_derived_features, apply_color_locus_keep, score_one_class_model


DEFAULT_STACK_BUDGETS = (10, 50, 100, 500, 1000)
REQUIRED_SUBTYPES = ("WN", "WC")


def list_compatible_stack_runs(
    config_path: str | Path,
    *,
    model_run_id: str,
    result_id: str,
    layer: ValidationLayer,
) -> pd.DataFrame:
    """List layer runs containing validators with matching data lineage."""
    first = _first_stage_result(config_path, model_run_id=model_run_id, result_id=result_id)
    rows: list[dict[str, object]] = []
    for run in list_layer_runs(layer).itertuples(index=False):
        results = compatible_layer_results(first, load_layer_results(layer, str(run.run_id)))
        pairs = validator_pairs(results)
        if pairs.empty:
            continue
        rows.append(
            {
                "run_id": str(run.run_id),
                "modified_at": run.modified_at,
                "source": run.source,
                "results_path": run.results_path,
                "pair_count": int(len(pairs)),
                "dataset_variant": first.get("dataset_variant"),
                "dataset_sha256": first.get("dataset_sha256"),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "run_id",
            "modified_at",
            "source",
            "results_path",
            "pair_count",
            "dataset_variant",
            "dataset_sha256",
        ],
    ).sort_values("modified_at", ascending=False, ignore_index=True)


def compatible_layer_results(first_stage: pd.Series, layer_results: pd.DataFrame) -> pd.DataFrame:
    """Return accepted layer rows that match the first-stage dataset lineage."""
    if layer_results.empty:
        return layer_results.copy()
    compatible = layer_results.copy()
    if "status" in compatible.columns:
        compatible = compatible[compatible["status"].eq("accepted")]
    for column in ["dataset_variant", "dataset_sha256", "models_config_sha256"]:
        expected = first_stage.get(column)
        if column in compatible.columns and expected is not None and pd.notna(expected):
            compatible = compatible[compatible[column].astype(str).eq(str(expected))]
    if "require_color_locus_keep" in compatible.columns:
        compatible = compatible[compatible["require_color_locus_keep"].fillna(False).astype(bool)]
    return compatible.reset_index(drop=True)


def validator_pairs(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize complete WN/WC validator pairs available for stack review."""
    required = {"feature_set", "method", "subtype", "model_path"}
    if results.empty or not required.issubset(results.columns):
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (feature_set, method), group in results.groupby(["feature_set", "method"], dropna=False):
        by_subtype = {
            str(row.subtype): row
            for row in group.itertuples(index=False)
            if str(row.subtype) in REQUIRED_SUBTYPES
        }
        if not all(subtype in by_subtype for subtype in REQUIRED_SUBTYPES):
            continue
        row: dict[str, object] = {
            "pair_key": f"{feature_set}::{method}",
            "pair_label": f"{method} | {feature_set}",
            "feature_set": feature_set,
            "method": method,
        }
        for subtype in REQUIRED_SUBTYPES:
            source = by_subtype[subtype]
            prefix = subtype.lower()
            for column in [
                "holdout_positive_count",
                "holdout_positive_retention",
                "holdout_negative_pass_rate",
                "threshold_calibration_negative_pass_rate",
                "holdout_average_precision",
                "holdout_roc_auc",
                "model_path",
            ]:
                row[f"{prefix}_{column}"] = getattr(source, column, pd.NA)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["feature_set", "method"], ignore_index=True
    )


def evaluate_candidate_stack(
    config_path: str | Path,
    *,
    model_run_id: str,
    result_id: str,
    layer: ValidationLayer,
    layer_run_id: str,
    feature_set: str,
    method: str,
    budgets: Iterable[int] = DEFAULT_STACK_BUDGETS,
) -> dict[str, pd.DataFrame]:
    """Evaluate an OR-combined WN/WC compatibility pair on holdout rankings."""
    first = _first_stage_result(config_path, model_run_id=model_run_id, result_id=result_id)
    layer_results = compatible_layer_results(first, load_layer_results(layer, layer_run_id))
    selected = layer_results[
        layer_results["feature_set"].astype(str).eq(str(feature_set))
        & layer_results["method"].astype(str).eq(str(method))
        & layer_results["subtype"].astype(str).isin(REQUIRED_SUBTYPES)
    ].copy()
    if set(selected["subtype"].astype(str)) != set(REQUIRED_SUBTYPES):
        raise ValueError(
            f"Complete WN/WC validator pair not found for {feature_set} / {method}."
        )

    cases = case_review.load_case_predictions(
        config_path,
        run_id=model_run_id,
        result_id=result_id,
        split="holdout",
    )
    if cases.empty:
        return _empty_evaluation()
    cases = cases.sort_values("rank").reset_index(drop=True)
    prepared = _load_holdout_features(selected)
    overlapping_features = [
        column
        for column in prepared.columns
        if column != "source_id" and column in cases.columns
    ]
    scored = cases.drop(columns=overlapping_features).merge(
        prepared,
        on="source_id",
        how="left",
        validate="one_to_one",
    )
    scored = _score_validator_pair(scored, selected)
    scored["rank_before"] = np.arange(1, len(scored) + 1)
    pass_first = scored.sort_values(
        ["second_layer_pass", "score"],
        ascending=[False, False],
        kind="mergesort",
    ).copy()
    pass_first["rank_after"] = np.arange(1, len(pass_first) + 1)
    rank_after = pass_first.set_index("source_id")["rank_after"]
    scored["rank_after"] = scored["source_id"].map(rank_after).astype("int64")
    scored["rank_delta"] = scored["rank_before"] - scored["rank_after"]

    normalized_budgets = sorted(
        {int(value) for value in budgets if int(value) > 0 and int(value) <= len(scored)}
    )
    return {
        "recovery": _recovery_curve(scored, pass_first, normalized_budgets),
        "tradeoff": _tradeoff_curve(scored, normalized_budgets),
        "subtypes": _subtype_tradeoff(scored, normalized_budgets),
        "cases": _stack_case_view(scored),
        "lineage": _lineage_view(first, selected, layer_run_id),
    }


def _first_stage_result(
    config_path: str | Path,
    *,
    model_run_id: str,
    result_id: str,
) -> pd.Series:
    results = load_run_results(config_path, run_id=model_run_id)
    match = results[results["result_id"].astype(str).eq(str(result_id))]
    if match.empty:
        raise ValueError(f"First-stage result not found: {model_run_id} / {result_id}")
    return match.iloc[0]


def _load_holdout_features(selected: pd.DataFrame) -> pd.DataFrame:
    dataset_values = selected["dataset_path"].dropna().astype(str).unique().tolist()
    if len(dataset_values) != 1:
        raise ValueError("Compatible validator pair does not resolve to one dataset path.")
    dataset_path = resolve_path(dataset_values[0])
    if not dataset_path.exists():
        raise FileNotFoundError(f"Reduced modeling dataset not found: {dataset_path}")
    dataset = pd.read_parquet(dataset_path)
    dataset, _ = apply_color_locus_keep(dataset)
    if "modeling_split" not in dataset.columns:
        raise ValueError(f"modeling_split is absent from {dataset_path}")
    dataset = dataset[dataset["modeling_split"].eq("holdout")].copy()
    dataset = add_derived_features(dataset)

    features: set[str] = set()
    for path in selected["model_path"].dropna().astype(str):
        artifact = load(resolve_path(path))
        features.update(str(feature) for feature in artifact.get("features", []))
    missing = sorted(features - set(dataset.columns))
    if missing:
        raise ValueError(f"Second-layer artifact features are absent from the dataset: {missing}")
    return dataset[["source_id", *sorted(features)]].copy()


def _score_validator_pair(cases: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    scored = cases.copy()
    pass_columns: list[str] = []
    for subtype in REQUIRED_SUBTYPES:
        row = selected[selected["subtype"].astype(str).eq(subtype)].iloc[0]
        artifact_path = resolve_path(str(row["model_path"]))
        if not artifact_path.exists():
            raise FileNotFoundError(f"Second-layer model artifact not found: {artifact_path}")
        artifact = load(artifact_path)
        features = [str(feature) for feature in artifact["features"]]
        values = score_one_class_model(artifact["model"], scored[features].astype("float64"))
        score_column = f"{subtype.lower()}_compatibility_score"
        pass_column = f"{subtype.lower()}_pass"
        scored[score_column] = values
        scored[pass_column] = values >= float(artifact["threshold"])
        pass_columns.append(pass_column)
    scored["second_layer_pass"] = scored[pass_columns].any(axis=1)
    return scored


def _recovery_curve(
    first_stage: pd.DataFrame,
    pass_first: pd.DataFrame,
    budgets: list[int],
) -> pd.DataFrame:
    total_wr = int(first_stage["target"].eq(1).sum())
    rows = []
    for policy, frame in [("First stage", first_stage), ("Pass-first", pass_first)]:
        for budget in budgets:
            reviewed = frame.head(budget)
            wr = int(reviewed["target"].eq(1).sum())
            rows.append(
                {
                    "policy": policy,
                    "budget": budget,
                    "wr_recovered": wr,
                    "wr_recovered_pct": wr / total_wr if total_wr else np.nan,
                    "negatives_reviewed": int(reviewed["target"].eq(0).sum()),
                    "candidates_per_wr": budget / wr if wr else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _tradeoff_curve(scored: pd.DataFrame, budgets: list[int]) -> pd.DataFrame:
    rows = []
    for budget in budgets:
        top = scored.head(budget)
        positives = top["target"].eq(1)
        negatives = ~positives
        passed = top["second_layer_pass"].astype(bool)
        wr_input = int(positives.sum())
        negative_input = int(negatives.sum())
        wr_pass = int((positives & passed).sum())
        negative_pass = int((negatives & passed).sum())
        rows.append(
            {
                "input_budget": budget,
                "wr_input": wr_input,
                "wr_pass": wr_pass,
                "wr_retention": wr_pass / wr_input if wr_input else np.nan,
                "negatives_input": negative_input,
                "negatives_pass": negative_pass,
                "negative_pass_rate": negative_pass / negative_input if negative_input else np.nan,
                "negative_removal_rate": 1 - negative_pass / negative_input if negative_input else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _subtype_tradeoff(scored: pd.DataFrame, budgets: list[int]) -> pd.DataFrame:
    if "wr_subtype" not in scored.columns:
        return pd.DataFrame()
    rows = []
    for budget in budgets:
        top = scored.head(budget)
        for subtype, group in top[top["target"].eq(1)].groupby(
            top["wr_subtype"].fillna("unknown")
        ):
            retained = int(group["second_layer_pass"].sum())
            rows.append(
                {
                    "input_budget": budget,
                    "wr_subtype": str(subtype),
                    "wr_input": int(len(group)),
                    "wr_pass": retained,
                    "wr_retention": retained / len(group) if len(group) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _stack_case_view(scored: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "source_id",
        "object_name",
        "target",
        "wr_subtype",
        "spectral_type",
        "simbad_main_type",
        "score",
        "rank_before",
        "rank_after",
        "rank_delta",
        "wn_pass",
        "wc_pass",
        "second_layer_pass",
        "wn_compatibility_score",
        "wc_compatibility_score",
    ]
    return scored[[column for column in preferred if column in scored.columns]].copy()


def _lineage_view(first: pd.Series, selected: pd.DataFrame, layer_run_id: str) -> pd.DataFrame:
    layer = selected.iloc[0]
    return pd.DataFrame(
        [
            {
                "first_result_id": first.get("result_id"),
                "layer_run_id": layer_run_id,
                "dataset_variant": first.get("dataset_variant"),
                "dataset_sha256": first.get("dataset_sha256"),
                "models_config_sha256": first.get("models_config_sha256"),
                "holdout_fraction": layer.get("holdout_fraction"),
                "require_color_locus_keep": layer.get("require_color_locus_keep"),
                "color_locus_excluded_rows": layer.get("color_locus_excluded_rows"),
            }
        ]
    )


def _empty_evaluation() -> dict[str, pd.DataFrame]:
    return {
        "recovery": pd.DataFrame(),
        "tradeoff": pd.DataFrame(),
        "subtypes": pd.DataFrame(),
        "cases": pd.DataFrame(),
        "lineage": pd.DataFrame(),
    }
