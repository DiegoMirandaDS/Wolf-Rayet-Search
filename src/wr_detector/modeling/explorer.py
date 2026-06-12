from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd

from wr_detector.config import load_yaml, resolve_path
from wr_detector.modeling.history import training_history_db_path


RANKING_METRICS = [
    "holdout_wr_at_10",
    "holdout_wr_at_50",
    "holdout_wr_at_100",
    "holdout_average_precision",
    "holdout_recall_at_fpr_0p005",
    "cv_train_gap_f2",
    "train_cv_gap_average_precision",
    "overfit_risk_score",
    "holdout_cv_gap_f2",
]

LOWER_IS_BETTER_METRICS = {
    "cv_train_gap_f2",
    "train_cv_gap_average_precision",
    "overfit_risk_score",
    "holdout_cv_gap_f2",
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
    "wr_holdout",
    "holdout_wr_at_10",
    "holdout_wr_at_10_pct",
    "holdout_wr_at_50",
    "holdout_wr_at_50_pct",
    "holdout_wr_at_100",
    "holdout_wr_at_100_pct",
    "holdout_average_precision",
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
    db_path = explorer_db_path(config_path)
    with _connect_read_only(db_path) as con:
        _require_table(con, "model_results", db_path)
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
    for k in [10, 50, 100, 500, 1000]:
        count_col = f"holdout_wr_at_{k}"
        pct_col = f"{count_col}_pct"
        if count_col in enriched.columns and "wr_holdout" in enriched.columns:
            enriched[pct_col] = _safe_divide(enriched[count_col], enriched["wr_holdout"])
    return enriched


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


def rank_models(results: pd.DataFrame, *, metric: str = "holdout_wr_at_100", top_n: int | None = None) -> pd.DataFrame:
    if results.empty:
        return results.copy()
    ranked = results.copy()
    if metric not in ranked.columns:
        raise ValueError(f"Metric not found in results: {metric}")
    ascending = metric in LOWER_IS_BETTER_METRICS
    tie_breakers = [column for column in ["holdout_wr_at_100", "holdout_average_precision", "holdout_recall_at_fpr_0p005"] if column in ranked.columns and column != metric]
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


def _require_table(con: duckdb.DuckDBPyConnection, table: str, db_path: Path) -> None:
    if not _table_exists(con, table):
        raise ValueError(f"{table} table not found in {db_path}")


def _table_exists(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    return bool(
        con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
            [table],
        ).fetchone()[0]
    )


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace({0: pd.NA})
    return numerator / denominator


def _selected_values(values: Iterable[str] | None) -> list[str]:
    if values is None:
        return []
    return [str(value) for value in values if value is not None]
