from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from wr_detector.apps.explorer_ui.candidate_plots import (
    candidate_galactic_plane,
    candidate_mollweide,
    candidate_photometric,
    disposition_bar,
    jaccard_heatmap,
    model_rank_ladder,
    pool_tile_map,
    rank_agreement,
    scoring_coverage,
    support_distribution,
)
from wr_detector.apps.explorer_ui.charts import (
    color_magnitude,
    dataset_heatmap,
    galactic_plane_map,
    galactic_polar_chart,
    metric_bar,
)
from wr_detector.modeling.explorer import (
    add_ranking_score,
    best_models_by_dataset,
    enrich_model_results,
    filter_results,
    list_explorer_runs,
    load_feature_importance,
    load_prediction_summary,
    load_run_results,
    rank_models,
    select_diverse_top_models,
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
    assert set(results["negative_train"]) == {180}
    assert set(results["negative_holdout"]) == {80}


def test_altair_charts_omit_optional_none_formats():
    data = pd.DataFrame(
        [
            {
                "short_label": "RF/none | strict_photometry",
                "model": "random_forest",
                "selection_status": "accepted",
                "dataset_variant": "strict_photometry",
                "sampler": "none",
                "holdout_average_precision": 0.42,
            }
        ]
    )

    bar_spec = metric_bar(
        data,
        value_col="holdout_average_precision",
        value_title="Average precision",
    ).to_dict()
    vertical_bar_spec = metric_bar(
        data,
        value_col="holdout_average_precision",
        value_title="Average precision",
        orientation="vertical",
    ).to_dict()
    heatmap_spec = dataset_heatmap(
        data,
        value_col="holdout_average_precision",
        value_title="Average precision",
    ).to_dict()

    assert "format" not in bar_spec["encoding"]["x"]["axis"]
    assert vertical_bar_spec["layer"][0]["encoding"]["x"]["sort"] == "-y"
    assert "format" not in heatmap_spec["layer"][0]["encoding"]["color"]["legend"]


def test_prediction_pool_candidate_charts_compile():
    candidates = pd.DataFrame(
        {
            "source_id": [1, 2],
            "consensus_rank": [1, 2],
            "best_model_rank": [1, 5],
            "model_support": [5, 4],
            "review_disposition": [
                "no_exact_match",
                "emission_or_ambiguous",
            ],
            "simbad_main_id": [None, "Emission source"],
            "BP_RP": [1.5, 2.0],
            "G": [13.0, 14.0],
            "mollweide_x": [0.1, -0.2],
            "mollweide_y": [0.2, -0.1],
            "galactic_l": [10.0, 20.0],
            "galactic_b": [1.0, -2.0],
            "distance_plotted": [True, True],
            "distance_kpc": [2.0, 3.0],
            "parallax_over_error": [5.0, 4.0],
            "galactocentric_x_kpc": [7.0, 6.0],
            "galactocentric_y_kpc": [1.0, -1.0],
        }
    )
    tiles = pd.DataFrame(
        {
            "tile_id": ["tile"],
            "status": ["completed"],
            "ra_min": [0.0],
            "ra_max": [10.0],
            "dec_min": [-5.0],
            "dec_max": [5.0],
            "acquired_pre_locus": [100],
            "written": [80],
            "retention_fraction": [0.8],
        }
    )
    scoring = pd.DataFrame(
        {
            "model_label": ["broad"],
            "eligible_rows": [100],
            "scored_rows": [90],
            "missing_feature_rows": [10],
        }
    )
    jaccard = pd.DataFrame(
        {
            "left_role": ["a"],
            "right_role": ["a"],
            "jaccard": [1.0],
            "intersection": [100],
            "top_k": [100],
        }
    )
    evidence = pd.DataFrame(
        {
            "role": ["broad"],
            "model_rank": [2],
            "score": [0.9],
            "rrf_contribution": [1 / 62],
        }
    )

    charts = [
        pool_tile_map(tiles),
        scoring_coverage(scoring),
        disposition_bar(
            pd.DataFrame(
                {
                    "review_disposition": ["no_exact_match"],
                    "rows": [1],
                }
            )
        ),
        jaccard_heatmap(jaccard),
        support_distribution(candidates),
        rank_agreement(candidates),
        candidate_photometric(
            candidates,
            selected_source_id=1,
            x="BP_RP",
            y="G",
        ),
        candidate_mollweide(candidates, selected_source_id=1),
        candidate_galactic_plane(candidates, selected_source_id=1),
        model_rank_ladder(evidence),
    ]

    assert all(chart is not None for chart in charts)
    for chart in charts:
        chart.to_dict()


def test_case_visualizations_compile_with_diagnostic_layers():
    cases = pd.DataFrame(
        {
            "source_id": [1, 2, 3, 4],
            "target": [0, 0, 1, 1],
            "diagnostic_state": [
                "Background",
                "Contaminant @K",
                "WR outside @K",
                "WR recovered @K",
            ],
            "object_name": ["n1", "n2", "wr1", "wr2"],
            "rank": [4, 2, 3, 1],
            "score": [0.1, 0.8, 0.4, 0.9],
            "BP_RP": [1.0, 1.2, 1.4, 1.6],
            "G": [14.0, 13.0, 12.0, 11.0],
            "galactic_l": [0.0, 90.0, 180.0, 270.0],
            "galactic_b": [0.0, 5.0, -5.0, 10.0],
            "polar_x": [0.0, 85.0, 0.0, -80.0],
            "polar_y": [90.0, 0.0, -95.0, 0.0],
            "distance_kpc": [1.0, 2.0, 3.0, 4.0],
            "galactocentric_x_kpc": [7.0, 8.0, 10.0, 8.0],
            "galactocentric_y_kpc": [0.0, 2.0, 0.0, -4.0],
            "distance_plotted": [True] * 4,
        }
    )

    cmd = color_magnitude(cases, selected_source_id=4).to_dict()
    polar = galactic_polar_chart(cases, selected_source_id=4).to_dict()
    plane = galactic_plane_map(cases, selected_source_id=4).to_dict()

    assert len(cmd["layer"]) == 5
    assert len(polar["layer"]) >= 7
    assert len(plane["layer"]) >= 8


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


def test_relative_recall_uses_each_models_positive_holdout_denominator():
    results = enrich_model_results(
        pd.DataFrame(
            [
                _selection_row("large_holdout", wr_holdout=100, wr_at_100=50),
                _selection_row("small_holdout", wr_holdout=50, wr_at_100=40),
            ]
        )
    )

    ranked = rank_models(results, metric="holdout_recall_at_100")

    assert ranked["result_id"].tolist() == ["small_holdout", "large_holdout"]
    assert ranked.set_index("result_id").loc["small_holdout", "holdout_recall_at_100"] == 0.8
    assert ranked.set_index("result_id").loc["large_holdout", "holdout_recall_at_100"] == 0.5
    assert ranked.set_index("result_id").loc["small_holdout", "holdout_precision_at_100"] == 0.4


def test_ranking_score_matches_weighted_metrics_and_soft_fpr_penalty():
    scored = add_ranking_score(
        pd.DataFrame(
            [
                {
                    "holdout_recall_at_100": 0.8,
                    "holdout_average_precision": 0.6,
                    "holdout_precision_at_100": 0.4,
                    "holdout_recall_at_50": 0.5,
                    "holdout_precision_wr": 0.7,
                    "holdout_recall_wr": 0.6,
                    "holdout_fpr": 0.01,
                }
            ]
        )
    )

    assert scored.loc[0, "ranking_metric_coverage"] == 1.0
    assert scored.loc[0, "ranking_score"] == pytest.approx(0.635)


def test_diverse_top_models_covers_profiles_without_duplicate_results():
    rows = []
    models = ["random_forest", "hist_gradient_boosting", "xgboost"]
    samplers = ["none", "smote"]
    variants = ["strict_photometry", "relaxed_poe_2"]
    for index, (model, sampler, variant) in enumerate(
        (model, sampler, variant)
        for model in models
        for sampler in samplers
        for variant in variants
    ):
        strength = index / 20.0
        rows.append(
            {
                "result_id": f"r{index}",
                "model": model,
                "sampler": sampler,
                "dataset_variant": variant,
                "feature_set": "colors_parallax_error" if index % 2 else "colors_parallax",
                "holdout_recall_at_100": 0.5 + strength,
                "holdout_average_precision": 0.8 - strength / 2,
                "holdout_precision_at_100": 0.3 + (index % 4) / 10,
                "holdout_recall_at_50": 0.2 + (index % 5) / 10,
                "holdout_precision_wr": 0.4 + (index % 3) / 10,
                "holdout_recall_wr": 0.5 + (index % 2) / 10,
                "holdout_fpr": 0.001 + (11 - index) / 1000,
            }
        )

    selected = select_diverse_top_models(pd.DataFrame(rows), top_n=10)

    assert len(selected) == 10
    assert selected["result_id"].is_unique
    assert selected["rank"].tolist() == list(range(1, 11))
    assert {
        "Recall@100 leader",
        "AP leader",
        "Precision@100 leader",
        "Threshold balance",
        "Lowest FPR",
        "Global balance",
    }.issubset(set(selected["profile"]))


def test_old_run_with_missing_selection_metrics_remains_rankable():
    scored = add_ranking_score(
        pd.DataFrame(
            [
                {"result_id": "old_a", "holdout_average_precision": 0.6},
                {"result_id": "old_b", "holdout_average_precision": 0.5},
            ]
        )
    )

    assert scored["ranking_score"].notna().all()
    assert scored["ranking_metric_coverage"].tolist() == [0.25, 0.25]
    assert rank_models(scored, metric="ranking_score").iloc[0]["result_id"] == "old_a"


def test_explorer_loads_optional_tables_and_handles_absent_predictions(tmp_path):
    config = _write_history_db(tmp_path, include_predictions=False)

    results = load_run_results(config, run_id="run_a")
    result_id = results.iloc[0]["result_id"]
    importance = load_feature_importance(config, run_id="run_a", result_id=result_id)
    prediction_summary = load_prediction_summary(config, run_id="run_a", result_id=result_id)

    assert importance["feature"].tolist() == ["BP_RP"]
    assert prediction_summary.empty


def test_compact_active_model_picker_changes_only_after_apply():
    source = """
import pandas as pd
from wr_detector.apps.explorer_ui.ui import active_model_control

results = pd.DataFrame([
    {
        "result_id": "r_none",
        "dataset_variant": "relaxed_photometry",
        "feature_set": "colors_parallax_error",
        "includes_parallax_error": True,
        "model": "xgboost",
        "sampler": "none",
        "negative_ratio_label": "10x",
        "selection_status": "accepted",
        "holdout_wr_at_100": 48,
        "holdout_average_precision": 0.53,
        "holdout_recall_at_fpr_0p005": 0.63,
        "overfit_risk_score": 2.0,
    },
    {
        "result_id": "r_smote",
        "dataset_variant": "relaxed_photometry",
        "feature_set": "colors_parallax_error",
        "includes_parallax_error": True,
        "model": "xgboost",
        "sampler": "smote",
        "negative_ratio_label": "10x",
        "selection_status": "accepted",
        "holdout_wr_at_100": 43,
        "holdout_average_precision": 0.57,
        "holdout_recall_at_fpr_0p005": 0.60,
        "overfit_risk_score": 1.0,
    },
])
active_model_control(results, widget_key="test_picker")
"""
    app = AppTest.from_string(source, default_timeout=20)
    app.session_state["selected_model_result_id"] = "r_none"
    app.run()

    configuration = next(
        widget for widget in app.selectbox if widget.label == "Configuration"
    )
    configuration.select("r_smote").run()
    assert app.session_state["selected_model_result_id"] == "r_none"

    apply_button = next(button for button in app.button if button.label == "Use as active model")
    apply_button.click().run()
    assert app.session_state["selected_model_result_id"] == "r_smote"
    assert not app.exception


def test_compare_activation_preserves_checked_ranking_order():
    source = """
import pandas as pd
from wr_detector.apps.explorer_ui.pages.compare import _active_selection_controls

compared = pd.DataFrame([
    {"result_id": "ranked_first"},
    {"result_id": "ranked_second"},
    {"result_id": "ranked_third"},
])
_active_selection_controls(compared)
"""
    app = AppTest.from_string(source, default_timeout=20).run()

    activate = next(
        button
        for button in app.button
        if button.label == "Make selected models active"
    )
    activate.click().run()

    assert app.session_state["active_model_result_ids"] == [
        "ranked_first",
        "ranked_second",
        "ranked_third",
    ]
    assert app.session_state["selected_model_result_id"] == "ranked_first"
    assert not app.exception


def test_active_model_selection_navigation_picker_and_deactivation():
    source = """
import pandas as pd
from wr_detector.apps.explorer_ui.ui import active_model_control

results = pd.DataFrame([
    {
        "result_id": "best",
        "dataset_variant": "relaxed_photometry",
        "includes_parallax_error": False,
        "model": "xgboost",
        "sampler": "none",
        "selection_status": "accepted",
        "ranking_score": 0.90,
        "holdout_recall_at_100": 0.80,
        "holdout_average_precision": 0.70,
    },
    {
        "result_id": "selected_first",
        "dataset_variant": "strict_poe_3",
        "includes_parallax_error": True,
        "model": "random_forest",
        "sampler": "smote",
        "selection_status": "accepted",
        "ranking_score": 0.75,
        "holdout_recall_at_100": 0.70,
        "holdout_average_precision": 0.60,
    },
    {
        "result_id": "selected_second",
        "dataset_variant": "relaxed_poe_2",
        "includes_parallax_error": False,
        "model": "hist_gradient_boosting",
        "sampler": "smote_enn",
        "selection_status": "overfit_warning",
        "ranking_score": 0.65,
        "holdout_recall_at_100": 0.60,
        "holdout_average_precision": 0.50,
    },
])
active_model_control(results, widget_key="selection_picker")
"""
    app = AppTest.from_string(source, default_timeout=20)
    app.session_state["active_model_result_ids"] = [
        "selected_first",
        "selected_second",
    ]
    app.session_state["selected_model_result_id"] = "best"
    app.run()

    assert app.session_state["selected_model_result_id"] == "selected_first"
    selected_picker = next(
        widget for widget in app.selectbox if widget.label == "Selected model"
    )
    assert len(selected_picker.options) == 2
    assert selected_picker.options[0].startswith("RF/SMOTE | strict_poe_3")
    assert selected_picker.options[1].startswith("HGB/SMOTE-ENN | relaxed_poe_2")

    selected_picker.select("selected_second").run()
    assert app.session_state["selected_model_result_id"] == "selected_first"
    apply_selected = next(
        button for button in app.button if button.label == "Use as active model"
    )
    apply_selected.click().run()
    assert app.session_state["selected_model_result_id"] == "selected_second"

    next_button = next(button for button in app.button if button.label == "→")
    next_button.click().run()
    assert app.session_state["selected_model_result_id"] == "selected_first"

    deactivate = next(
        button for button in app.button if button.label == "Deactivate selection"
    )
    deactivate.click().run()
    with pytest.raises(KeyError):
        app.session_state["active_model_result_ids"]
    assert app.session_state["selected_model_result_id"] == "best"
    assert app.session_state["compare_selection_version"] == 1
    assert not app.exception


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


def _selection_row(
    result_id: str,
    *,
    wr_holdout: int,
    wr_at_100: int,
) -> dict[str, object]:
    return {
        "run_id": "run_relative",
        "result_id": result_id,
        "dataset_variant": "strict_photometry",
        "feature_set": "colors_parallax",
        "model": "xgboost",
        "sampler": "none",
        "negative_ratio_label": "10x",
        "selection_status": "accepted",
        "n_train": 500,
        "wr_train": 50,
        "n_holdout": 200,
        "wr_holdout": wr_holdout,
        "holdout_wr_at_50": min(wr_at_100, 30),
        "holdout_wr_at_100": wr_at_100,
        "holdout_average_precision": 0.5,
        "holdout_precision_wr": 0.6,
        "holdout_recall_wr": 0.5,
        "holdout_fp": 2,
        "bayes_best_params": "{}",
    }


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
        "n_train": 200,
        "wr_train": 20,
        "n_holdout": 100,
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
