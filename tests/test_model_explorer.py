from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from wr_detector.modeling.explorer import (
    best_models_by_dataset,
    filter_results,
    list_explorer_runs,
    load_feature_importance,
    load_prediction_summary,
    load_run_results,
    rank_models,
)


def test_explorer_lists_runs_and_enriches_results(tmp_path):
    config = _write_history_db(tmp_path)

    runs = list_explorer_runs(config)
    results = load_run_results(config, run_id="run_a")

    assert runs["run_id"].tolist() == ["run_a"]
    assert len(results) == 3
    assert set(results["dataset_family"]) == {"strict", "relaxed"}
    assert set(results["astrometric_subset"]) == {"photometry", "poe_3"}
    assert results.set_index("feature_set").loc["colors_parallax_error", "includes_parallax_error"]
    assert "strict_photometry / colors_parallax / random_forest / smote / 10x" in set(results["model_label"])
    assert results.loc[results["dataset_variant"].eq("strict_photometry"), "holdout_wr_at_100_pct"].max() == 0.75


def test_explorer_filters_and_ranks_models(tmp_path):
    config = _write_history_db(tmp_path)
    results = load_run_results(config, run_id="run_a")

    filtered = filter_results(
        results,
        dataset_families=["strict"],
        astrometric_subsets=["photometry"],
        models=["xgboost"],
        include_warnings=True,
    )
    accepted_only = filter_results(results, include_warnings=False)
    ranked = rank_models(results, metric="holdout_wr_at_100")
    best = best_models_by_dataset(results, metric="holdout_wr_at_100")

    assert filtered["model"].tolist() == ["xgboost"]
    assert set(accepted_only["selection_status"]) == {"accepted"}
    assert ranked.iloc[0]["model"] == "xgboost"
    assert set(best["dataset_variant"]) == {"strict_photometry", "relaxed_poe_3"}
    assert best.loc[best["dataset_variant"].eq("strict_photometry"), "model"].item() == "xgboost"


def test_explorer_loads_optional_tables_and_handles_absent_predictions(tmp_path):
    config = _write_history_db(tmp_path, include_predictions=False)

    results = load_run_results(config, run_id="run_a")
    result_id = results.iloc[0]["result_id"]
    importance = load_feature_importance(config, run_id="run_a", result_id=result_id)
    prediction_summary = load_prediction_summary(config, run_id="run_a", result_id=result_id)

    assert importance["feature"].tolist() == ["BP_RP"]
    assert prediction_summary.empty


def _write_history_db(tmp_path: Path, *, include_predictions: bool = True) -> Path:
    db_path = tmp_path / "history.duckdb"
    config_path = tmp_path / "models.yaml"
    config_path.write_text(
        "\n".join(
            [
                "outputs:",
                f"  training_history_db: {db_path.as_posix()}",
            ]
        ),
        encoding="utf-8",
    )

    runs = pd.DataFrame(
        [
            {
                "run_id": "run_a",
                "run_name": "train_models",
                "imported_at": "2026-06-10T00:00:00+00:00",
                "source_csv": "reports/modeling/runs/run_a/model_training_results.csv",
                "config_path": "configs/models.yaml",
                "notes": None,
                "row_count": 3,
                "csv_sha256": "abc",
            }
        ]
    )
    results = pd.DataFrame(
        [
            _result_row("r1", "strict_photometry", "colors_parallax", "random_forest", "accepted", 12, 0.40),
            _result_row("r2", "strict_photometry", "colors_parallax_error", "xgboost", "overfit_warning", 15, 0.55),
            _result_row("r3", "relaxed_poe_3", "colors_parallax", "hist_gradient_boosting", "accepted", 10, 0.45),
        ]
    )
    importance = pd.DataFrame(
        [
            {
                "run_id": "run_a",
                "result_id": "r1",
                "dataset_variant": "strict_photometry",
                "feature_set": "colors_parallax",
                "model": "random_forest",
                "sampler": "smote",
                "feature": "BP_RP",
                "importance_mean": 0.3,
                "importance_std": 0.0,
                "importance_type": "model_feature_importance",
            }
        ]
    )
    predictions = pd.DataFrame(
        [
            {"run_id": "run_a", "result_id": "r1", "split": "holdout", "target": 1, "score": 0.9},
            {"run_id": "run_a", "result_id": "r1", "split": "holdout", "target": 0, "score": 0.2},
        ]
    )

    with duckdb.connect(str(db_path)) as con:
        con.register("runs", runs)
        con.execute("CREATE TABLE training_runs AS SELECT * FROM runs")
        con.unregister("runs")
        con.register("results", results)
        con.execute("CREATE TABLE model_results AS SELECT * FROM results")
        con.unregister("results")
        con.register("importance", importance)
        con.execute("CREATE TABLE feature_importance AS SELECT * FROM importance")
        con.unregister("importance")
        if include_predictions:
            con.register("predictions", predictions)
            con.execute("CREATE TABLE model_predictions AS SELECT * FROM predictions")
            con.unregister("predictions")
    return config_path


def _result_row(result_id: str, variant: str, feature_set: str, model: str, status: str, wr_at_100: int, ap: float) -> dict[str, object]:
    return {
        "run_id": "run_a",
        "result_id": result_id,
        "dataset_variant": variant,
        "feature_set": feature_set,
        "model": model,
        "sampler": "smote",
        "negative_ratio_label": "10x",
        "selection_status": status,
        "wr_holdout": 20,
        "holdout_wr_at_10": min(wr_at_100, 10),
        "holdout_wr_at_50": min(wr_at_100, 12),
        "holdout_wr_at_100": wr_at_100,
        "holdout_average_precision": ap,
        "holdout_recall_at_fpr_0p005": ap / 2,
        "cv_train_gap_f2": 0.1,
        "holdout_cv_gap_f2": 0.2,
        "bayes_best_params": "{}",
        "model_path": "reports/modeling/models/model.joblib",
    }
