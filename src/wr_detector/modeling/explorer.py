"""Read-only training-history queries and result enrichment for the Model Explorer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd

from wr_detector.config import load_yaml, resolve_path
from wr_detector.modeling.history import training_history_db_path


RANKING_METRICS = [
    "ranking_score",
    "holdout_recall_at_100",
    "holdout_average_precision",
    "holdout_precision_at_100",
    "holdout_recall_at_50",
    "holdout_precision_wr",
    "holdout_recall_wr",
    "holdout_fpr",
    "holdout_recall_at_fpr_0p005",
    "train_cv_gap_average_precision",
    "overfit_risk_score",
]

LOWER_IS_BETTER_METRICS = {
    "holdout_fpr",
    "cv_train_gap_f2",
    "train_cv_gap_average_precision",
    "overfit_risk_score",
    "holdout_cv_gap_f2",
}

RANKING_SCORE_WEIGHTS = {
    "holdout_recall_at_100": 0.35,
    "holdout_average_precision": 0.25,
    "holdout_precision_at_100": 0.15,
    "holdout_recall_at_50": 0.10,
    "holdout_precision_wr": 0.10,
    "holdout_recall_wr": 0.05,
}

MODEL_PROFILE_METRICS = {
    "Recall@100 specialist": ("holdout_recall_at_100", False),
    "AP specialist": ("holdout_average_precision", False),
    "Precision@100 specialist": ("holdout_precision_at_100", False),
    "Threshold specialist": ("threshold_operating_score", False),
    "Low-FPR specialist": ("holdout_fpr", True),
}

IMPORTANT_HYPERPARAMS = [
    "n_estimators",
    "max_depth",
    "min_samples_leaf",
    "max_features",
    "learning_rate",
    "subsample",
    "colsample_bytree",
    "reg_lambda",
    "min_child_weight",
]

DEFAULT_RESULT_COLUMNS = [
    "run_id",
    "result_id",
    "dataset_variant",
    "dataset_family",
    "astrometric_subset",
    "feature_set",
    "includes_parallax_error",
    "model",
    "sampler",
    "negative_ratio_label",
    "model_label",
    "selection_status",
    "n_train",
    "wr_train",
    "negative_train",
    "n_holdout",
    "wr_holdout",
    "negative_holdout",
    "holdout_wr_at_10",
    "holdout_wr_at_10_pct",
    "holdout_wr_at_50",
    "holdout_wr_at_50_pct",
    "holdout_precision_at_50",
    "holdout_recall_at_50",
    "holdout_wr_at_100",
    "holdout_wr_at_100_pct",
    "holdout_precision_at_100",
    "holdout_recall_at_100",
    "holdout_average_precision",
    "holdout_precision_wr",
    "holdout_recall_wr",
    "holdout_fp",
    "holdout_fpr",
    "threshold_operating_score",
    "ranking_score",
    "ranking_metric_coverage",
    "holdout_recall_at_fpr_0p005",
    "cv_train_gap_f2",
    "train_cv_gap_average_precision",
    "cv_holdout_drop_average_precision",
    "holdout_to_cv_average_precision_ratio",
    "overfit_warning_flag",
    "overfit_risk_score",
    "holdout_cv_gap_f2",
    "bayes_best_params",
    "model_path",
    "holdout_confusion_matrix_path",
    "holdout_roc_curve_path",
    "holdout_pr_curve_path",
    "feature_importance_figure_path",
]


def explorer_db_path(config_path: str | Path = "configs/models.yaml") -> Path:
    config = load_yaml(config_path)
    return training_history_db_path(config)


def list_explorer_runs(config_path: str | Path = "configs/models.yaml") -> pd.DataFrame:
    db_path = explorer_db_path(config_path)
    if not db_path.exists():
        return pd.DataFrame(columns=["run_id", "run_name", "imported_at", "notes", "row_count"])
    with _connect_read_only(db_path) as con:
        if not _table_exists(con, "training_runs"):
            return pd.DataFrame(columns=["run_id", "run_name", "imported_at", "notes", "row_count"])
        return con.execute("SELECT * FROM training_runs ORDER BY imported_at DESC").fetchdf()


def load_run_results(config_path: str | Path = "configs/models.yaml", *, run_id: str) -> pd.DataFrame:
    if not str(run_id).strip():
        return enrich_model_results(pd.DataFrame())
    db_path = explorer_db_path(config_path)
    if not db_path.exists():
        return enrich_model_results(pd.DataFrame())
    with _connect_read_only(db_path) as con:
        if not _table_exists(con, "model_results"):
            return enrich_model_results(pd.DataFrame())
        results = con.execute("SELECT * FROM model_results WHERE run_id = ?", [run_id]).fetchdf()
    if results.empty:
        return enrich_model_results(results)
    return enrich_model_results(results)


def load_run_artifacts(config_path: str | Path = "configs/models.yaml", *, run_id: str) -> pd.DataFrame:
    return _load_optional_run_table(config_path, "model_artifacts", run_id)


def load_feature_importance(config_path: str | Path = "configs/models.yaml", *, run_id: str, result_id: str | None = None) -> pd.DataFrame:
    df = _load_optional_run_table(config_path, "feature_importance", run_id)
    if result_id is not None and not df.empty and "result_id" in df.columns:
        df = df[df["result_id"].eq(result_id)].copy()
    return df


def load_prediction_summary(config_path: str | Path = "configs/models.yaml", *, run_id: str, result_id: str | None = None) -> pd.DataFrame:
    db_path = explorer_db_path(config_path)
    with _connect_read_only(db_path) as con:
        if not _table_exists(con, "model_predictions"):
            return pd.DataFrame(columns=["split", "target", "rows", "score_min", "score_mean", "score_median", "score_max"])
        clauses = ["run_id = ?"]
        params: list[object] = [run_id]
        if result_id is not None:
            clauses.append("result_id = ?")
            params.append(result_id)
        where = " AND ".join(clauses)
        return con.execute(
            f"""
            SELECT
                split,
                target,
                COUNT(*) AS rows,
                MIN(score) AS score_min,
                AVG(score) AS score_mean,
                MEDIAN(score) AS score_median,
                MAX(score) AS score_max
            FROM model_predictions
            WHERE {where}
            GROUP BY split, target
            ORDER BY split, target
            """,
            params,
        ).fetchdf()


def enrich_model_results(results: pd.DataFrame) -> pd.DataFrame:
    enriched = results.copy()
    if enriched.empty:
        for column in DEFAULT_RESULT_COLUMNS:
            if column not in enriched.columns:
                enriched[column] = pd.Series(dtype="object")
        return enriched

    enriched["dataset_family"] = enriched["dataset_variant"].map(dataset_family)
    enriched["astrometric_subset"] = enriched["dataset_variant"].map(astrometric_subset)
    enriched["includes_parallax_error"] = enriched["feature_set"].astype(str).str.contains("error", case=False, na=False)
    enriched["model_label"] = enriched.apply(model_label, axis=1)
    enriched = add_hyperparameter_columns(enriched)
    for split in ["train", "holdout"]:
        total_col = f"n_{split}"
        wr_col = f"wr_{split}"
        negative_col = f"negative_{split}"
        if total_col in enriched.columns and wr_col in enriched.columns:
            total = pd.to_numeric(enriched[total_col], errors="coerce")
            wr = pd.to_numeric(enriched[wr_col], errors="coerce")
            enriched[negative_col] = total - wr
    for k in [10, 50, 100, 500, 1000]:
        count_col = f"holdout_wr_at_{k}"
        pct_col = f"{count_col}_pct"
        if count_col in enriched.columns and "wr_holdout" in enriched.columns:
            derived_recall = _safe_divide(enriched[count_col], enriched["wr_holdout"])
            recall_col = f"holdout_recall_at_{k}"
            precision_col = f"holdout_precision_at_{k}"
            enriched[pct_col] = _prefer_existing(enriched, recall_col, derived_recall)
            enriched[recall_col] = enriched[pct_col]
            effective_budget = pd.concat(
                [
                    pd.to_numeric(enriched["n_holdout"], errors="coerce"),
                    pd.Series(float(k), index=enriched.index),
                ],
                axis=1,
            ).min(axis=1)
            derived_precision = _safe_divide(enriched[count_col], effective_budget)
            enriched[precision_col] = _prefer_existing(
                enriched,
                precision_col,
                derived_precision,
            )
    for column in [
        "holdout_recall_at_50",
        "holdout_recall_at_100",
        "holdout_precision_at_100",
        "holdout_average_precision",
        "holdout_precision_wr",
        "holdout_recall_wr",
        "holdout_fp",
        "holdout_recall_at_fpr_0p005",
        "train_cv_gap_average_precision",
        "overfit_risk_score",
    ]:
        if column not in enriched.columns:
            enriched[column] = pd.Series(float("nan"), index=enriched.index)
    if "holdout_fp" in enriched.columns and "negative_holdout" in enriched.columns:
        derived_fpr = _safe_divide(enriched["holdout_fp"], enriched["negative_holdout"])
        enriched["holdout_fpr"] = _prefer_existing(enriched, "holdout_fpr", derived_fpr)
    elif "holdout_fpr" not in enriched.columns:
        enriched["holdout_fpr"] = pd.Series(float("nan"), index=enriched.index)

    enriched["threshold_operating_score"] = _harmonic_mean(
        _numeric_column(enriched, "holdout_precision_wr"),
        _numeric_column(enriched, "holdout_recall_wr"),
    )
    enriched = add_ranking_score(enriched)
    return enriched


def add_ranking_score(results: pd.DataFrame) -> pd.DataFrame:
    """Add a relative, missing-aware model-selection score.

    With complete metrics this is the requested weighted sum. Older runs keep
    working through available-weight normalization and receive a small coverage
    discount so a one-metric row cannot outrank a fully audited result by accident.
    """
    scored = results.copy()
    if "threshold_operating_score" not in scored.columns:
        scored["threshold_operating_score"] = _harmonic_mean(
            _numeric_column(scored, "holdout_precision_wr"),
            _numeric_column(scored, "holdout_recall_wr"),
        )
    weighted = pd.Series(0.0, index=scored.index, dtype="float64")
    available_weight = pd.Series(0.0, index=scored.index, dtype="float64")
    total_weight = float(sum(RANKING_SCORE_WEIGHTS.values()))
    for column, weight in RANKING_SCORE_WEIGHTS.items():
        values = _numeric_column(scored, column).clip(lower=0.0, upper=1.0)
        available = values.notna()
        weighted = weighted.add(values.fillna(0.0) * weight, fill_value=0.0)
        available_weight = available_weight.add(available.astype("float64") * weight)

    normalized = weighted.div(available_weight.replace({0.0: pd.NA}))
    coverage = available_weight / total_weight
    coverage_discount = 0.85 + 0.15 * coverage
    fpr = _numeric_column(scored, "holdout_fpr").clip(lower=0.0, upper=1.0)
    soft_fpr_penalty = 0.05 * fpr.fillna(0.0).pow(0.5)
    scored["ranking_metric_coverage"] = coverage
    scored["ranking_score"] = (normalized * coverage_discount - soft_fpr_penalty).clip(
        lower=0.0,
        upper=1.0,
    )
    return scored


def assign_model_profiles(results: pd.DataFrame) -> pd.DataFrame:
    """Describe each model by its strongest relative operational characteristic."""
    profiled = results.copy()
    if profiled.empty:
        profiled["profile"] = pd.Series(dtype="object")
        return profiled

    strengths: dict[str, pd.Series] = {}
    for label, (column, lower_is_better) in MODEL_PROFILE_METRICS.items():
        values = _numeric_column(profiled, column)
        strengths[label] = values.rank(
            pct=True,
            ascending=not lower_is_better,
            na_option="keep",
        ).fillna(0.0)
    strength_frame = pd.DataFrame(strengths, index=profiled.index)
    profiled["profile"] = strength_frame.idxmax(axis=1)

    balance_strength = _numeric_column(profiled, "ranking_score").rank(
        pct=True,
        ascending=True,
        na_option="keep",
    ).fillna(0.0)
    standout_strength = strength_frame.max(axis=1)
    globally_balanced = balance_strength.ge(0.90) & standout_strength.lt(0.95)
    profiled.loc[globally_balanced, "profile"] = "Global balance"
    return profiled


def select_diverse_top_models(
    results: pd.DataFrame,
    *,
    top_n: int = 10,
) -> pd.DataFrame:
    """Select strong models across complementary profiles without near duplicates."""
    if results.empty or top_n <= 0:
        empty = results.head(0).copy()
        empty["profile"] = pd.Series(dtype="object")
        empty["rank"] = pd.Series(dtype="int64")
        return empty

    candidates = assign_model_profiles(add_ranking_score(results))
    profile_targets = [
        ("Recall@100 leader", "holdout_recall_at_100", False),
        ("AP leader", "holdout_average_precision", False),
        ("Precision@100 leader", "holdout_precision_at_100", False),
        ("Threshold balance", "threshold_operating_score", False),
        ("Lowest FPR", "holdout_fpr", True),
        ("Global balance", "ranking_score", False),
    ]
    selected_indices: list[object] = []
    selected_profiles: dict[object, str] = {}
    used_signatures: set[tuple[str, ...]] = set()

    for profile, metric, ascending in profile_targets:
        ordered = _profile_order(candidates, metric=metric, ascending=ascending)
        unused = [index for index in ordered.index if index not in selected_indices]
        if not unused:
            continue
        index = unused[0]
        selected_indices.append(index)
        selected_profiles[index] = profile
        used_signatures.add(_model_similarity_signature(candidates.loc[index]))
        if len(selected_indices) >= top_n:
            break

    balanced = _profile_order(candidates, metric="ranking_score", ascending=False)
    for require_new_signature in [True, False]:
        for index in balanced.index:
            if index in selected_indices:
                continue
            signature = _model_similarity_signature(candidates.loc[index])
            if require_new_signature and signature in used_signatures:
                continue
            selected_indices.append(index)
            selected_profiles[index] = str(candidates.at[index, "profile"])
            used_signatures.add(signature)
            if len(selected_indices) >= top_n:
                break
        if len(selected_indices) >= top_n:
            break

    selected = candidates.loc[selected_indices].copy()
    selected["profile"] = selected.index.map(selected_profiles)
    selected = selected.sort_values(
        ["ranking_score", "holdout_recall_at_100", "holdout_average_precision"],
        ascending=False,
        na_position="last",
    ).head(top_n)
    selected.insert(0, "rank", range(1, len(selected) + 1))
    return selected.reset_index(drop=True)


def add_hyperparameter_columns(results: pd.DataFrame) -> pd.DataFrame:
    enriched = results.copy()
    if "bayes_best_params" not in enriched.columns:
        enriched["hyperparams_compact"] = ""
        enriched["overfit_warning_flag"] = _overfit_warning_flag(enriched)
        return enriched

    parsed = enriched["bayes_best_params"].map(parse_best_params)
    numeric_params = [name for name in IMPORTANT_HYPERPARAMS if name != "max_features"]
    for name in IMPORTANT_HYPERPARAMS:
        enriched[name] = parsed.map(lambda params, param_name=name: params.get(param_name, pd.NA))
        if name in numeric_params:
            enriched[name] = pd.to_numeric(enriched[name], errors="coerce")
        else:
            # Mixed str/float values (e.g. max_features: "sqrt" or 0.5) break Arrow serialization.
            enriched[name] = enriched[name].map(lambda value: str(value) if pd.notna(value) else pd.NA)
    enriched["hyperparams_compact"] = enriched.apply(compact_hyperparams, axis=1)
    enriched["overfit_warning_flag"] = _overfit_warning_flag(enriched)
    return enriched


def _overfit_warning_flag(results: pd.DataFrame) -> pd.Series:
    if "overfit_warning_flag" in results.columns:
        return results["overfit_warning_flag"].fillna(False).astype(bool)
    if "selection_status" in results.columns:
        return results["selection_status"].eq("overfit_warning")
    return pd.Series(False, index=results.index)


def parse_best_params(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        params = value
    elif value is None or pd.isna(value) or value == "":
        params = {}
    else:
        try:
            params = json.loads(str(value))
        except (TypeError, json.JSONDecodeError):
            params = {}
    return {str(key).replace("estimator__", ""): param_value for key, param_value in params.items()}


def compact_hyperparams(row: pd.Series) -> str:
    pieces = []
    if pd.notna(row.get("n_estimators", pd.NA)):
        pieces.append(f"trees={int(row['n_estimators'])}")
    if pd.notna(row.get("max_depth", pd.NA)):
        pieces.append(f"depth={row['max_depth']}")
    if pd.notna(row.get("learning_rate", pd.NA)):
        pieces.append(f"eta={float(row['learning_rate']):.3f}")
    if pd.notna(row.get("min_samples_leaf", pd.NA)):
        pieces.append(f"leaf={row['min_samples_leaf']}")
    if pd.notna(row.get("min_child_weight", pd.NA)):
        pieces.append(f"child={row['min_child_weight']}")
    if pd.notna(row.get("subsample", pd.NA)):
        pieces.append(f"sub={float(row['subsample']):.2f}")
    if pd.notna(row.get("colsample_bytree", pd.NA)):
        pieces.append(f"col={float(row['colsample_bytree']):.2f}")
    if pd.notna(row.get("reg_lambda", pd.NA)):
        pieces.append(f"lambda={float(row['reg_lambda']):.2f}")
    return " | ".join(pieces)


def rank_models(results: pd.DataFrame, *, metric: str = "ranking_score", top_n: int | None = None) -> pd.DataFrame:
    if results.empty:
        return results.copy()
    ranked = results.copy()
    if metric == "ranking_score" and metric not in ranked.columns:
        ranked = add_ranking_score(ranked)
    if metric not in ranked.columns:
        raise ValueError(f"Metric not found in results: {metric}")
    ascending = metric in LOWER_IS_BETTER_METRICS
    tie_breakers = [
        column
        for column in [
            "ranking_score",
            "holdout_recall_at_100",
            "holdout_average_precision",
            "holdout_precision_at_100",
            "holdout_recall_at_fpr_0p005",
        ]
        if column in ranked.columns and column != metric
    ]
    sort_columns = [metric] + tie_breakers
    ascending_values = [ascending] + [False] * len(tie_breakers)
    ranked = ranked.sort_values(sort_columns, ascending=ascending_values, na_position="last")
    if top_n is not None:
        ranked = ranked.head(top_n)
    return ranked.reset_index(drop=True)


def best_models_by_dataset(results: pd.DataFrame, *, metric: str = "holdout_wr_at_100") -> pd.DataFrame:
    if results.empty:
        return results.copy()
    ranked = rank_models(results, metric=metric)
    return ranked.drop_duplicates("dataset_variant", keep="first").sort_values("dataset_variant").reset_index(drop=True)


def filter_results(
    results: pd.DataFrame,
    *,
    dataset_variants: Iterable[str] | None = None,
    dataset_families: Iterable[str] | None = None,
    astrometric_subsets: Iterable[str] | None = None,
    feature_sets: Iterable[str] | None = None,
    models: Iterable[str] | None = None,
    samplers: Iterable[str] | None = None,
    negative_ratio_labels: Iterable[str] | None = None,
    selection_statuses: Iterable[str] | None = None,
    include_warnings: bool = True,
) -> pd.DataFrame:
    filtered = results.copy()
    for column, values in [
        ("dataset_variant", dataset_variants),
        ("dataset_family", dataset_families),
        ("astrometric_subset", astrometric_subsets),
        ("feature_set", feature_sets),
        ("model", models),
        ("sampler", samplers),
        ("negative_ratio_label", negative_ratio_labels),
        ("selection_status", selection_statuses),
    ]:
        selected = _selected_values(values)
        if selected and column in filtered.columns:
            filtered = filtered[filtered[column].fillna("missing").astype(str).isin(selected)]
    if not include_warnings and "selection_status" in filtered.columns:
        filtered = filtered[filtered["selection_status"].eq("accepted")]
    return filtered.reset_index(drop=True)


def dataset_family(dataset_variant: object) -> str:
    value = str(dataset_variant)
    if value.startswith("strict_"):
        return "strict"
    if value.startswith("relaxed_"):
        return "relaxed"
    return "unknown"


def astrometric_subset(dataset_variant: object) -> str:
    value = str(dataset_variant)
    for prefix in ["strict_", "relaxed_"]:
        if value.startswith(prefix):
            return value.removeprefix(prefix)
    return value


def model_label(row: pd.Series) -> str:
    parts = [
        row.get("dataset_variant"),
        row.get("feature_set"),
        row.get("model"),
        row.get("sampler"),
        row.get("negative_ratio_label"),
    ]
    return " / ".join(str(part) for part in parts if pd.notna(part) and str(part).strip())


def resolve_artifact_path(path: object) -> Path | None:
    if path is None or pd.isna(path) or not str(path).strip():
        return None
    return resolve_path(str(path))


def _load_optional_run_table(config_path: str | Path, table: str, run_id: str) -> pd.DataFrame:
    db_path = explorer_db_path(config_path)
    with _connect_read_only(db_path) as con:
        if not _table_exists(con, table):
            return pd.DataFrame()
        return con.execute(f"SELECT * FROM {table} WHERE run_id = ?", [run_id]).fetchdf()


def _connect_read_only(db_path: Path) -> duckdb.DuckDBPyConnection:
    if not db_path.exists():
        raise FileNotFoundError(f"Training history DB not found: {db_path}")
    return duckdb.connect(str(db_path), read_only=True)


def _table_exists(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    return bool(
        con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
            [table],
        ).fetchone()[0]
    )


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numeric_numerator = pd.to_numeric(numerator, errors="coerce")
    numeric_denominator = pd.to_numeric(denominator, errors="coerce").replace({0: pd.NA})
    return numeric_numerator / numeric_denominator


def _prefer_existing(
    frame: pd.DataFrame,
    column: str,
    fallback: pd.Series,
) -> pd.Series:
    if column not in frame.columns:
        return fallback
    existing = pd.to_numeric(frame[column], errors="coerce")
    return existing.where(existing.notna(), fallback)


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(float("nan"), index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _harmonic_mean(left: pd.Series, right: pd.Series) -> pd.Series:
    denominator = left + right
    result = 2.0 * left * right / denominator.replace({0.0: pd.NA})
    both_zero = left.eq(0.0) & right.eq(0.0)
    return result.mask(both_zero, 0.0)


def _profile_order(
    candidates: pd.DataFrame,
    *,
    metric: str,
    ascending: bool,
) -> pd.DataFrame:
    if metric not in candidates.columns:
        return candidates.head(0)
    available = candidates[_numeric_column(candidates, metric).notna()].copy()
    if available.empty:
        return available
    sort_columns = [metric]
    ascending_values = [ascending]
    if metric != "ranking_score" and "ranking_score" in available.columns:
        sort_columns.append("ranking_score")
        ascending_values.append(False)
    return available.sort_values(
        sort_columns,
        ascending=ascending_values,
        na_position="last",
        kind="stable",
    )


def _model_similarity_signature(row: pd.Series) -> tuple[str, ...]:
    variant = str(row.get("dataset_variant", ""))
    family = dataset_family(variant)
    return tuple(
        str(value)
        for value in [
            row.get("model", ""),
            row.get("sampler", ""),
            family,
            row.get("feature_set", ""),
        ]
    )


def _selected_values(values: Iterable[str] | None) -> list[str]:
    if values is None:
        return []
    return [str(value) for value in values if value is not None]
