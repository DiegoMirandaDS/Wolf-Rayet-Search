from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from wr_detector.modeling import (
    build_model_matrix,
    build_model_pipeline,
    cleanup_unreferenced_model_artifacts,
    compute_binary_metrics,
    compute_ranking_metrics,
    load_modeling_dataset,
    list_training_runs,
    make_global_holdout_mask,
    reduce_negative_variants,
    run_model_benchmark,
    select_threshold,
    sync_training_history,
    train_models,
)
from wr_detector.modeling.training import (
    apply_positive_cohort,
    compute_model_stability_diagnostics,
    make_threshold_selection_scores,
    resolve_positive_cohort,
)


def test_build_model_matrix_rejects_leakage_columns():
    df = pd.DataFrame({"source_id": [1, 2], "BP_RP": [1.0, 2.0], "target": [1, 0]})

    with pytest.raises(ValueError, match="Forbidden"):
        build_model_matrix(df, ["source_id", "BP_RP"])


def test_load_modeling_dataset_can_require_color_locus_keep(tmp_path):
    ref_dir = tmp_path / "ref"
    neg_dir = tmp_path / "neg"
    ref_dir.mkdir()
    neg_dir.mkdir()
    ref = pd.DataFrame([_row(1, target_like=1.0), _row(2, target_like=1.1)])
    ref["color_locus_keep"] = [True, False]
    neg = pd.DataFrame([_row(101, target_like=-1.0), _row(102, target_like=-1.1)])
    neg["color_locus_keep"] = [False, True]
    ref.to_parquet(ref_dir / "wr_reference_strict_photometry_color_locus.parquet", index=False)
    neg.to_parquet(neg_dir / "simbad_negative_strict_photometry_color_locus.parquet", index=False)
    paths = tmp_path / "paths.yaml"
    paths.write_text(
        f"processed_reference_dir: {ref_dir.as_posix()}\n"
        f"processed_simbad_negative_dir: {neg_dir.as_posix()}\n",
        encoding="utf-8",
    )
    filters = tmp_path / "filters.yaml"
    filters.write_text(
        "\n".join(
            [
                "color_locus:",
                '  reference_output_template: "wr_reference_{variant}_color_locus.parquet"',
                '  simbad_negative_output_template: "simbad_negative_{variant}_color_locus.parquet"',
            ]
        ),
        encoding="utf-8",
    )
    config = {
        "paths_config": paths,
        "filters_config": filters,
        "modeling_dataset": {"source": "color_locus", "require_color_locus_keep": True},
    }

    dataset = load_modeling_dataset(config, "strict_photometry")

    assert dataset["source_id"].tolist() == [1, 102]
    assert dataset["target"].tolist() == [1, 0]


def test_global_holdout_is_reproducible_by_source_id():
    source_ids = pd.Series([1, 2, 3, 4, 5, 1])

    first = make_global_holdout_mask(source_ids, holdout_fraction=0.2, random_state=42)
    second = make_global_holdout_mask(source_ids, holdout_fraction=0.2, random_state=42)

    assert first.tolist() == second.tolist()
    assert first.iloc[0] == first.iloc[5]


def test_global_holdout_can_be_class_stratified():
    source_ids = pd.Series(range(1000))
    target = pd.Series([1] * 100 + [0] * 900)

    holdout = make_global_holdout_mask(source_ids, holdout_fraction=0.2, random_state=42, target=target)

    wr_rate = holdout[target.eq(1)].mean()
    negative_rate = holdout[target.eq(0)].mean()
    assert 0.10 < wr_rate < 0.30
    assert 0.15 < negative_rate < 0.25


def test_sampler_is_inside_pipeline():
    pipeline = build_model_pipeline(
        {"estimator": "random_forest", "sampler": "smote"},
        random_state=42,
        positive_weight=10.0,
    )

    assert "sampler" in pipeline.named_steps
    assert "estimator" in pipeline.named_steps


def test_none_sampler_uses_estimator_weighting_without_synthetic_rows():
    pipeline = build_model_pipeline(
        {"estimator": "random_forest"},
        random_state=42,
        positive_weight=10.0,
        sampler_config={"type": "none"},
    )

    assert "sampler" not in pipeline.named_steps
    assert (
        pipeline.named_steps["estimator"].get_params()["class_weight"]
        == "balanced_subsample"
    )


def test_sampled_random_forest_does_not_apply_class_weight_twice():
    pipeline = build_model_pipeline(
        {"estimator": "random_forest"},
        random_state=42,
        positive_weight=10.0,
        sampler_config={"type": "smote", "k_neighbors": 3},
    )

    assert "sampler" in pipeline.named_steps
    assert pipeline.named_steps["estimator"].get_params()["class_weight"] is None


def test_none_sampler_uses_native_boosting_class_weights():
    hist = build_model_pipeline(
        {"estimator": "hist_gradient_boosting"},
        random_state=42,
        positive_weight=10.0,
        sampler_config={"type": "none"},
    )
    sampled_hist = build_model_pipeline(
        {"estimator": "hist_gradient_boosting"},
        random_state=42,
        positive_weight=10.0,
        sampler_config={"type": "smote", "k_neighbors": 3},
    )

    assert hist.named_steps["estimator"].get_params()["class_weight"] == {
        0: 1.0,
        1: 10.0,
    }
    assert (
        sampled_hist.named_steps["estimator"].get_params()["class_weight"]
        is None
    )


def test_none_sampler_uses_xgboost_scale_pos_weight():
    pytest.importorskip("xgboost")
    weighted = build_model_pipeline(
        {"estimator": "xgboost"},
        random_state=42,
        positive_weight=10.0,
        sampler_config={"type": "none"},
    )
    sampled = build_model_pipeline(
        {"estimator": "xgboost"},
        random_state=42,
        positive_weight=10.0,
        sampler_config={"type": "smote", "k_neighbors": 3},
    )

    assert weighted.named_steps["estimator"].get_params()[
        "scale_pos_weight"
    ] == pytest.approx(10.0)
    assert sampled.named_steps["estimator"].get_params()[
        "scale_pos_weight"
    ] == pytest.approx(1.0)


def test_native_positive_cohort_changes_only_wr_rows(tmp_path):
    reference_db = tmp_path / "wr_reference.duckdb"
    with duckdb.connect(str(reference_db)) as con:
        con.execute(
            "CREATE TABLE twomass_matches "
            "(source_id BIGINT, match_method VARCHAR)"
        )
        con.execute(
            "CREATE TABLE wise_matches "
            "(source_id BIGINT, match_method VARCHAR)"
        )
        con.executemany(
            "INSERT INTO twomass_matches VALUES (?, ?)",
            [(1, "gaia_xmatch"), (2, "vizier_cone"), (3, "gaia_xmatch")],
        )
        con.executemany(
            "INSERT INTO wise_matches VALUES (?, ?)",
            [(1, "gaia_xmatch"), (2, "gaia_xmatch"), (3, "vizier_cone")],
        )
    config = {
        "positive_cohorts": {
            "gaia_native_ir": {
                "type": "gaia_reference_match_method",
                "reference_db": str(reference_db),
                "twomass_match_method": "gaia_xmatch",
                "wise_match_method": "gaia_xmatch",
            }
        }
    }
    frame = pd.DataFrame(
        {
            "source_id": [1, 2, 3, 101, 102],
            "target": [1, 1, 1, 0, 0],
            "modeling_split": [
                "train",
                "train",
                "train",
                "train",
                "train",
            ],
            "BP_RP": [1.0, 1.1, 1.2, -1.0, -1.1],
        }
    )

    cohort = resolve_positive_cohort(config, "gaia_native_ir")
    filtered, lineage = apply_positive_cohort(
        frame,
        cohort,
        prefix="train_positive",
    )

    assert filtered["source_id"].tolist() == [1, 101, 102]
    pd.testing.assert_frame_equal(
        filtered.loc[filtered["target"].eq(0)].reset_index(drop=True),
        frame.loc[frame["target"].eq(0)].reset_index(drop=True),
    )
    assert lineage["train_positive_positives_before"] == 3
    assert lineage["train_positive_positives_after"] == 1
    assert lineage["train_positive_negatives"] == 2
    assert lineage["train_positive_cohort_source_ids_sha256"]
    assert lineage["train_positive_cohort_contract_sha256"]


def test_metrics_and_threshold_work_for_imbalanced_scores():
    y_true = [1, 0, 0, 0, 0, 0]
    y_score = [0.9, 0.8, 0.4, 0.3, 0.2, 0.1]

    threshold = select_threshold(y_true, y_score, min_precision=0.5)
    metrics = compute_binary_metrics(y_true, y_score, threshold=threshold["threshold"])

    assert metrics["recall_wr"] == 1.0
    assert metrics["tp"] == 1
    assert "average_precision" in metrics


def test_threshold_calibration_rejects_positive_rows():
    class Fitted:
        def predict_proba(self, x):
            return pd.DataFrame(
                {0: [0.8] * len(x), 1: [0.2] * len(x)}
            ).to_numpy()

    with pytest.raises(ValueError, match="negatives only"):
        make_threshold_selection_scores(
            fitted=Fitted(),
            y_train=pd.Series([1, 0]),
            oof_score=[0.8, 0.2],
            x_calibration=pd.DataFrame({"x": [1.0]}),
            y_calibration=pd.Series([1]),
        )


def test_ranking_metrics_report_top_k_and_fixed_fpr():
    metrics = compute_ranking_metrics(
        [1, 0, 1, 0, 0],
        [0.9, 0.8, 0.7, 0.2, 0.1],
        top_k=[1, 3],
        fpr_levels=[0.34],
    )

    assert metrics["precision_at_1"] == 1.0
    assert metrics["recall_at_3"] == 1.0
    assert metrics["wr_at_3"] == 2
    assert metrics["candidates_per_wr_at_3"] == pytest.approx(1.5)
    assert "recall_at_fpr_0p34" in metrics


def test_model_stability_diagnostics_use_average_precision_gap():
    diagnostics = compute_model_stability_diagnostics(
        {
            "selection": {
                "max_train_cv_f2_gap": 0.15,
                "max_train_cv_average_precision_gap": 0.20,
                "min_holdout_cv_average_precision_ratio": 0.25,
            }
        },
        train_metrics={"f2_wr": 0.60, "average_precision": 0.95},
        cv_metrics={"f2_wr": 0.52, "average_precision": 0.70},
        holdout_metrics={"f2_wr": 0.50, "average_precision": 0.30},
        holdout_ranking={"recall_at_100": 0.30, "recall_at_fpr_0p005": 0.20},
    )

    assert diagnostics["overfit_gap_f2_flag"] is False
    assert diagnostics["overfit_gap_average_precision_flag"] is True
    assert diagnostics["overfit_warning_flag"] is True
    assert diagnostics["train_cv_gap_average_precision"] == pytest.approx(0.25)


def test_run_model_benchmark_writes_summary_table(tmp_path):
    ref_dir = tmp_path / "ref"
    neg_dir = tmp_path / "neg"
    out_dir = tmp_path / "reports"
    ref_dir.mkdir()
    neg_dir.mkdir()
    feature_rows = []
    for source_id in range(1, 13):
        feature_rows.append(_row(source_id, target_like=1.0 + source_id / 100))
    negative_rows = []
    for source_id in range(101, 161):
        negative_rows.append(_row(source_id, target_like=-1.0 - source_id / 1000))
    pd.DataFrame(feature_rows).to_parquet(ref_dir / "wr.parquet", index=False)
    pd.DataFrame(negative_rows).to_parquet(neg_dir / "neg.parquet", index=False)

    paths = tmp_path / "paths.yaml"
    paths.write_text(
        f"processed_reference_dir: {ref_dir.as_posix()}\n"
        f"processed_simbad_negative_dir: {neg_dir.as_posix()}\n",
        encoding="utf-8",
    )
    filters = tmp_path / "filters.yaml"
    filters.write_text(
        "\n".join(
            [
                f"paths_config: {paths.as_posix()}",
                "output_files:",
                "  strict_photometry: wr.parquet",
                "simbad_negative_output_files:",
                "  strict_photometry: neg.parquet",
            ]
        ),
        encoding="utf-8",
    )
    config = tmp_path / "models.yaml"
    config.write_text(
        "\n".join(
            [
                f"paths_config: {paths.as_posix()}",
                f"filters_config: {filters.as_posix()}",
                "random_state: 42",
                "holdout_fraction: 0.2",
                "cv: {n_splits: 3, n_repeats: 2}",
                "selection: {metric: f2, min_precision: 0.5}",
                "outputs:",
                f"  reports_dir: {out_dir.as_posix()}",
                f"  benchmark_table: {(out_dir / 'benchmark.csv').as_posix()}",
                "dataset_variants: [strict_photometry]",
                "feature_sets:",
                "  colors_only: [BP_RP, G_BP, G_RP, J_H, J_K, H_K, W1_W2]",
                "models:",
                "  logistic_balanced:",
                "    estimator: logistic_regression",
                "    sampler: none",
                "  random_forest_smote:",
                "    estimator: random_forest",
                "    sampler: smote",
            ]
        ),
        encoding="utf-8",
    )

    results = run_model_benchmark(config)

    assert len(results) == 2
    assert (out_dir / "benchmark.csv").exists()
    assert {"holdout_f2_wr", "cv_train_gap_f2", "selected_threshold"}.issubset(results.columns)


def test_negative_reduction_keeps_wr_and_reduces_negatives_reproducibly(tmp_path):
    config = _write_modeling_config(tmp_path, n_wr=20, n_neg=300)

    first_summary, first_audit = reduce_negative_variants(config, variants=["strict_photometry"])
    first_reduced = pd.read_parquet(tmp_path / "modeling" / "strict_photometry_reduced.parquet")
    second_summary, _ = reduce_negative_variants(config, variants=["strict_photometry"])
    second_reduced = pd.read_parquet(tmp_path / "modeling" / "strict_photometry_reduced.parquet")

    train = first_reduced[first_reduced["modeling_split"] == "train"]
    calibration = first_reduced[first_reduced["modeling_split"] == "threshold_calibration"]
    train_wr = train[train["target"] == 1]
    train_neg = train[train["target"] == 0]
    assert len(train_neg) == 10 * len(train_wr)
    assert not calibration.empty
    assert calibration["target"].eq(0).all()
    assert int(first_summary.loc[0, "train_wr"]) == len(train_wr)
    assert first_audit["ks_statistic"].notna().all()
    assert first_reduced["source_id"].tolist() == second_reduced["source_id"].tolist()
    assert second_summary.loc[0, "train_negative_reduced"] == first_summary.loc[0, "train_negative_reduced"]


def test_train_models_minimal_bayes_search_outputs_metrics(tmp_path):
    config = _write_modeling_config(tmp_path, n_wr=20, n_neg=300)
    reduce_negative_variants(config, variants=["strict_photometry"])

    results = train_models(
        config,
        variants=["strict_photometry"],
        models=["logistic_regression"],
        samplers=["none"],
        feature_sets=["colors_only"],
        n_iter=2,
        run_id="test_train_run",
    )

    assert len(results) == 1
    assert {
        "run_id",
        "negative_ratio_label",
        "holdout_f2_wr",
        "holdout_accuracy",
        "holdout_balanced_accuracy",
        "selection_status",
        "model_path",
        "holdout_confusion_matrix_path",
        "holdout_roc_curve_path",
        "holdout_pr_curve_path",
        "feature_importance_path",
        "holdout_precision_at_50",
        "holdout_recall_at_fpr_0p001",
        "train_cv_gap_average_precision",
        "cv_holdout_drop_average_precision",
        "holdout_to_cv_average_precision_ratio",
        "overfit_warning_flag",
        "overfit_risk_score",
        "dataset_sha256",
        "models_config_sha256",
        "code_worktree_sha256",
        "feature_columns_json",
        "imbalance_strategy",
        "imbalance_parameter_json",
        "training_negative_to_positive_ratio",
        "positive_class_weight",
        "threshold_selection_method",
        "threshold_selection_negative_count",
        "threshold_calibration_negative_pass_rate",
        "threshold_calibration_score_p99",
        "model_sha256",
        "holdout_candidates_per_wr_at_50",
    }.issubset(results.columns)
    assert results.loc[0, "imbalance_strategy"] == (
        "estimator_native_class_weight"
    )
    assert results.loc[0, "positive_class_weight"] > 1
    assert "class_weight" in results.loc[0, "imbalance_parameter_json"]
    for column in ["model_path", "holdout_confusion_matrix_path", "holdout_roc_curve_path", "holdout_pr_curve_path", "feature_importance_path"]:
        assert Path(results.loc[0, column]).exists()
    predictions = pd.read_csv(results.loc[0, "predictions_path"])
    assert {"split", "row_id", "source_id", "target", "score", "predicted", "threshold"}.issubset(predictions.columns)

    assert (tmp_path / "reports" / "model_training_results.csv").exists()
    assert (tmp_path / "reports" / "artifacts" / "runs" / "test_train_run" / "model_training_results.csv").exists()

    second = train_models(
        config,
        variants=["strict_photometry"],
        models=["logistic_regression"],
        samplers=["none"],
        feature_sets=["colors_only"],
        n_iter=2,
        run_id="test_train_run",
        resume_run=True,
    )
    assert len(second) == 1


def test_merge_training_results_accepts_legacy_csv_without_run_columns():
    from wr_detector.modeling.training import merge_training_results

    previous = pd.DataFrame(
        [
            {
                "dataset_variant": "strict_photometry",
                "feature_set": "colors_only",
                "model": "random_forest",
                "sampler": "smote",
                "holdout_f2_wr": 0.1,
            }
        ]
    )
    current = pd.DataFrame(
        [
            {
                "run_id": "new_run",
                "dataset_variant": "strict_photometry",
                "negative_ratio_label": "10x",
                "feature_set": "colors_only",
                "model": "random_forest",
                "sampler": "smote",
                "holdout_f2_wr": 0.2,
            }
        ]
    )

    merged = merge_training_results(previous, current)

    assert {"run_id", "negative_ratio_label"}.issubset(merged.columns)
    assert set(merged["run_id"]) == {"legacy_csv", "new_run"}


def test_train_models_can_resume_multiple_negative_ratios(tmp_path):
    config = _write_modeling_config(tmp_path, n_wr=20, n_neg=300)
    reduce_negative_variants(config, variants=["strict_photometry"], negative_ratios=[10, 20])

    first = train_models(
        config,
        variants=["strict_photometry"],
        models=["logistic_regression"],
        samplers=["smote"],
        feature_sets=["colors_only"],
        negative_ratios=[10],
        n_iter=2,
        run_id="resume_test",
    )
    assert len(first) == 1
    assert first["negative_ratio_label"].tolist() == ["10x"]

    resumed = train_models(
        config,
        variants=["strict_photometry"],
        models=["logistic_regression"],
        samplers=["smote"],
        feature_sets=["colors_only"],
        negative_ratios=[10, 20],
        n_iter=2,
        run_id="resume_test",
        resume_run=True,
    )

    assert len(resumed) == 2
    assert set(resumed["negative_ratio_label"]) == {"10x", "20x"}
    run_csv = tmp_path / "reports" / "artifacts" / "runs" / "resume_test" / "model_training_results.csv"
    assert len(pd.read_csv(run_csv)) == 2


def test_training_history_syncs_results_and_cleanup_detects_orphans(tmp_path):
    reports_dir = tmp_path / "reports" / "modeling"
    models_dir = reports_dir / "models"
    figures_dir = reports_dir / "figures"
    tables_dir = tmp_path / "reports" / "tables"
    models_dir.mkdir(parents=True)
    figures_dir.mkdir(parents=True)
    tables_dir.mkdir(parents=True)

    model_path = models_dir / "model.joblib"
    metadata_path = models_dir / "model.json"
    cm_path = figures_dir / "cm.png"
    roc_path = figures_dir / "roc.png"
    pr_path = figures_dir / "pr.png"
    fi_fig_path = figures_dir / "fi.png"
    predictions_path = reports_dir / "run_predictions.csv"
    fi_path = reports_dir / "run_feature_importance.csv"
    orphan_path = models_dir / "old.joblib"
    run_orphan_dir = reports_dir / "runs" / "old_run" / "reports"
    run_orphan_dir.mkdir(parents=True)
    run_orphan_sidecar = run_orphan_dir / "old_predictions.csv"
    for path in [model_path, cm_path, roc_path, pr_path, fi_fig_path, orphan_path]:
        path.write_bytes(b"x")
    metadata_path.write_text('{"dataset_variant": "strict_photometry", "model": "xgboost"}', encoding="utf-8")
    predictions_path.write_text("split,target,score,predicted,threshold\nholdout,1,0.9,1,0.5\n", encoding="utf-8")
    fi_path.write_text("feature,importance_mean,importance_std,importance_type\nBP_RP,0.5,0.0,test\n", encoding="utf-8")
    run_orphan_sidecar.write_text("split,target,score,predicted,threshold\nholdout,0,0.1,0,0.5\n", encoding="utf-8")

    results_path = tables_dir / "model_training_results.csv"
    pd.DataFrame(
        [
            {
                "dataset_variant": "strict_photometry",
                "feature_set": "colors_parallax",
                "model": "xgboost",
                "sampler": "smote",
                "holdout_f2_wr": 0.5,
                "holdout_wr_at_100": 10,
                "holdout_average_precision": 0.4,
                "model_path": str(model_path),
                "holdout_confusion_matrix_path": str(cm_path),
                "holdout_roc_curve_path": str(roc_path),
                "holdout_pr_curve_path": str(pr_path),
                "predictions_path": str(predictions_path),
                "feature_importance_path": str(fi_path),
                "feature_importance_figure_path": str(fi_fig_path),
            }
        ]
    ).to_csv(results_path, index=False)

    config_path = tmp_path / "models.yaml"
    config_path.write_text(
        "\n".join(
            [
                "outputs:",
                f"  training_results: {results_path.as_posix()}",
                f"  training_history_db: {(reports_dir / 'history.duckdb').as_posix()}",
                f"  reports_dir: {reports_dir.as_posix()}",
                f"  models_dir: {models_dir.as_posix()}",
                f"  figures_dir: {figures_dir.as_posix()}",
            ]
        ),
        encoding="utf-8",
    )

    synced = sync_training_history(config_path, run_id="test_run", run_name="test", replace_run=True)
    assert synced["rows"] == 1
    assert synced["artifacts"] == 8
    assert synced["feature_importance_rows"] == 1
    assert synced["prediction_rows"] == 1
    assert synced["metadata_rows"] == 1

    con = duckdb.connect(str(reports_dir / "history.duckdb"), read_only=True)
    assert con.execute("SELECT COUNT(*) FROM training_runs").fetchone()[0] == 1
    assert con.execute("SELECT COUNT(*) FROM model_results").fetchone()[0] == 1
    assert con.execute("SELECT COUNT(*) FROM model_artifacts").fetchone()[0] == 8
    assert con.execute("SELECT COUNT(*) FROM model_predictions").fetchone()[0] == 1
    assert con.execute("SELECT COUNT(*) FROM model_metadata").fetchone()[0] == 1
    con.close()

    cleanup = cleanup_unreferenced_model_artifacts(config_path, apply=False)
    assert str(orphan_path) in set(cleanup.loc[~cleanup["keep"], "path"])
    assert str(run_orphan_sidecar) in set(cleanup.loc[~cleanup["keep"], "path"])
    cleanup_unreferenced_model_artifacts(config_path, apply=True)
    assert not orphan_path.exists()
    assert not run_orphan_sidecar.exists()
    assert model_path.exists()

    runs = list_training_runs(config_path)
    assert "test_run" in set(runs["run_id"])


def test_training_history_repairs_all_null_hash_column_type(tmp_path):
    results_path = tmp_path / "model_training_results.csv"
    history_path = tmp_path / "training_history.duckdb"
    config_path = tmp_path / "models.yaml"
    config_path.write_text(
        "\n".join(
            [
                "outputs:",
                f"  training_results: {results_path.as_posix()}",
                f"  training_history_db: {history_path.as_posix()}",
            ]
        ),
        encoding="utf-8",
    )
    base = {
        "dataset_variant": "relaxed_photometry",
        "feature_set": "colors_parallax_error",
        "model": "xgboost",
        "sampler": "none",
    }
    pd.DataFrame(
        [
            {
                **base,
                "train_positive_cohort_source_ids_sha256": None,
            }
        ]
    ).to_csv(results_path, index=False)
    sync_training_history(
        config_path,
        run_id="all_wr",
        replace_run=True,
    )

    expected_hash = "a" * 64
    pd.DataFrame(
        [
            {
                **base,
                "train_positive_cohort_source_ids_sha256": expected_hash,
            }
        ]
    ).to_csv(results_path, index=False)
    sync_training_history(
        config_path,
        run_id="native_wr",
        replace_run=True,
    )

    with duckdb.connect(str(history_path), read_only=True) as con:
        stored = con.execute(
            "SELECT train_positive_cohort_source_ids_sha256 "
            "FROM model_results WHERE run_id = 'native_wr'"
        ).fetchone()[0]
        column_type = con.execute(
            "SELECT data_type FROM information_schema.columns "
            "WHERE table_name = 'model_results' "
            "AND column_name = "
            "'train_positive_cohort_source_ids_sha256'"
        ).fetchone()[0]
    assert stored == expected_hash
    assert column_type == "VARCHAR"


def _row(source_id: int, *, target_like: float) -> dict[str, float | int | str]:
    return {
        "source_id": source_id,
        "sample_label": "wr" if source_id < 100 else "non_wr_simbad",
        "BP_RP": target_like + 0.1,
        "G_BP": target_like + 0.2,
        "G_RP": target_like + 0.3,
        "J_H": target_like + 0.4,
        "J_K": target_like + 0.5,
        "H_K": target_like + 0.6,
        "W1_W2": target_like + 0.7,
        "parallax": abs(target_like),
        "parallax_error": 0.1,
        "parallax_over_error": 5.0,
        "G": 10.0,
        "BP": 11.0,
        "RP": 9.0,
        "J": 8.0,
        "H": 7.5,
        "Ks": 7.0,
        "W1": 6.8,
        "W2": 6.6,
        "ruwe": 1.0,
        "pmra": 0.0,
        "pmdec": 0.0,
    }


def _write_modeling_config(tmp_path, *, n_wr: int, n_neg: int) -> Path:
    ref_dir = tmp_path / "ref"
    neg_dir = tmp_path / "neg"
    ref_dir.mkdir()
    neg_dir.mkdir()
    wr_rows = [_row(source_id, target_like=1.0 + source_id / 100) for source_id in range(1, n_wr + 1)]
    neg_rows = [_row(source_id, target_like=-1.0 - source_id / 1000) for source_id in range(101, 101 + n_neg)]
    pd.DataFrame(wr_rows).to_parquet(ref_dir / "wr.parquet", index=False)
    pd.DataFrame(neg_rows).to_parquet(neg_dir / "neg.parquet", index=False)
    paths = tmp_path / "paths.yaml"
    paths.write_text(
        f"processed_reference_dir: {ref_dir.as_posix()}\n"
        f"processed_simbad_negative_dir: {neg_dir.as_posix()}\n",
        encoding="utf-8",
    )
    filters = tmp_path / "filters.yaml"
    filters.write_text(
        "\n".join(
            [
                f"paths_config: {paths.as_posix()}",
                "output_files:",
                "  strict_photometry: wr.parquet",
                "simbad_negative_output_files:",
                "  strict_photometry: neg.parquet",
            ]
        ),
        encoding="utf-8",
    )
    config = tmp_path / "models.yaml"
    config.write_text(
        "\n".join(
            [
                f"paths_config: {paths.as_posix()}",
                f"filters_config: {filters.as_posix()}",
                "random_state: 42",
                "holdout_fraction: 0.2",
                "dataset_variants: [strict_photometry]",
                "feature_sets:",
                "  colors_only: [BP_RP, G_BP, G_RP, J_H, J_K, H_K, W1_W2]",
                "negative_reduction:",
                "  negative_to_wr_ratio: 10",
                "  quantile_bins: 2",
                "  columns: [BP_RP, G_BP, G_RP, J_H, J_K, H_K, W1_W2]",
                "cv: {n_splits: 2, n_repeats: 1}",
                "bayes_search: {n_iter: 2, n_jobs: 1, scoring: f2}",
                "selection: {metric: f2, min_precision: 0.5, min_balanced_accuracy: 0.5, max_train_cv_f2_gap: 1.0}",
                "outputs:",
                f"  reports_dir: {(tmp_path / 'reports' / 'artifacts').as_posix()}",
                f"  modeling_data_dir: {(tmp_path / 'modeling').as_posix()}",
                f"  reduction_summary: {(tmp_path / 'reports' / 'negative_reduction_summary.csv').as_posix()}",
                f"  reduction_audit: {(tmp_path / 'reports' / 'negative_reduction_audit.csv').as_posix()}",
                f"  training_results: {(tmp_path / 'reports' / 'model_training_results.csv').as_posix()}",
                f"  training_history_db: {(tmp_path / 'reports' / 'training_history.duckdb').as_posix()}",
                "  auto_sync_training_history: true",
                f"  models_dir: {(tmp_path / 'models').as_posix()}",
                f"  figures_dir: {(tmp_path / 'figures').as_posix()}",
                "  reduced_dataset_template: '{variant}_reduced.parquet'",
                "samplers:",
                "  none: {type: none}",
                "  smote: {type: smote, k_neighbors: 1}",
                "models:",
                "  logistic_regression:",
                "    estimator: logistic_regression",
                "    search_space:",
                "      estimator__C: {type: real, low: 0.1, high: 10.0, prior: log-uniform}",
            ]
        ),
        encoding="utf-8",
    )
    return config
