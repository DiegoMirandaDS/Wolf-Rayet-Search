from __future__ import annotations

import numpy as np
import pandas as pd
from types import SimpleNamespace

from wr_detector.pipelines.prediction_pool_delivery import (
    _contains_personal_path,
    _data_dictionary,
    _make_prediction_export,
    _model_contract_json,
)


def test_prediction_export_preserves_scores_and_explains_missing_ranks():
    consensus = pd.DataFrame(
        [
            {
                "eligibility_rank": 1,
                "consensus_rank": 2,
                "source_id": 10,
                "eligibility_normalized_rrf": 0.01,
                "rrf_score": 0.02,
                "eligible_model_count": 2,
                "ranked_model_count": 1,
                "eligible_support_fraction": 0.5,
                "model_support": 1,
                "variant_support": 1,
                "estimator_support": 1,
                "best_model_rank": 1,
                "mean_model_rank": 1.0,
            },
            {
                "eligibility_rank": 2,
                "consensus_rank": 1,
                "source_id": 20,
                "eligibility_normalized_rrf": 0.009,
                "rrf_score": 0.009,
                "eligible_model_count": 1,
                "ranked_model_count": 1,
                "eligible_support_fraction": 1.0,
                "model_support": 1,
                "variant_support": 1,
                "estimator_support": 1,
                "best_model_rank": 2,
                "mean_model_rank": 2.0,
            },
        ]
    )
    opportunities = pd.DataFrame(
        [
            {"source_id": 10, "result_id": "a", "eligible": True, "model_status": "ranked_top_10000", "eligibility_reason": "passed_all_variant_filters", "locus_keep": True, "photometry_keep": True, "astrometry_keep": True, "model_rank": 1},
            {"source_id": 10, "result_id": "b", "eligible": True, "model_status": "eligible_below_top_10000", "eligibility_reason": "passed_all_variant_filters", "locus_keep": True, "photometry_keep": True, "astrometry_keep": True, "model_rank": np.nan},
            {"source_id": 20, "result_id": "a", "eligible": False, "model_status": "ineligible_variant", "eligibility_reason": "failed_locus", "locus_keep": False, "photometry_keep": True, "astrometry_keep": True, "model_rank": np.nan},
            {"source_id": 20, "result_id": "b", "eligible": True, "model_status": "ranked_top_10000", "eligibility_reason": "passed_all_variant_filters", "locus_keep": True, "photometry_keep": True, "astrometry_keep": True, "model_rank": 2},
        ]
    )
    features = pd.DataFrame(
        {
            "source_id": [10, 20],
            "ra": [1.0, 2.0],
            "dec": [-1.0, -2.0],
            "gaia_designation": ["Gaia DR3 10", "Gaia DR3 20"],
        }
    )
    review = pd.DataFrame(
        {
            "source_id": [10],
            "review_disposition": ["no_exact_match"],
        }
    )
    models = pd.DataFrame(
        [
            {"result_id": "a", "delivery_key": "m1_a"},
            {"result_id": "b", "delivery_key": "m2_b"},
        ]
    )
    scores = pd.DataFrame(
        [
            {"source_id": 10, "result_id": "a", "score": 0.9, "predicted": True, "threshold": 0.5},
            {"source_id": 10, "result_id": "b", "score": 0.2, "predicted": False, "threshold": 0.5},
            {"source_id": 20, "result_id": "b", "score": 0.8, "predicted": True, "threshold": 0.5},
        ]
    )

    exported = _make_prediction_export(
        consensus,
        opportunities,
        features,
        review,
        models,
        scores,
    ).set_index("source_id")

    assert exported.loc[10, "m2_b_status"] == "eligible_below_top_10000"
    assert pd.isna(exported.loc[10, "m2_b_rank"])
    assert exported.loc[10, "m2_b_score"] == 0.2
    assert exported.loc[20, "m1_a_status"] == "ineligible_variant"
    assert exported.loc[20, "m1_a_eligibility_reason"] == "failed_locus"
    assert not bool(exported.loc[20, "m1_a_locus_keep"])
    assert pd.isna(exported.loc[20, "m1_a_rank"])
    assert pd.isna(exported.loc[20, "m1_a_score"])
    assert (
        exported.loc[20, "simbad_review_status"]
        == "not_queried_in_frozen_snapshot"
    )
    export = _make_prediction_export(
        consensus,
        opportunities,
        features,
        review,
        models,
        scores,
    )
    ranks = export["eligibility_rank"].tolist()
    assert ranks == sorted(ranks)
    assert export["source_id"].tolist() == [10, 20]


def test_personal_path_detection_rejects_windows_user_paths():
    assert _contains_personal_path(r'E:\Proyectos\Wolf-Rayet-Detector\model.joblib')
    assert _contains_personal_path(r'C:\Users\diego\artifact.csv')
    assert not _contains_personal_path('models/m1_broad_xgb.joblib')
    assert not _contains_personal_path('https://simbad.cds.unistra.fr/simbad/')


def test_model_contract_explains_filters_and_locus_in_spanish():
    row = {
        "delivery_key": "m3_test",
        "model_order": 3,
        "model": "xgboost",
        "sampler": "smote",
        "role": "accepted_stable",
        "result_id": "abc",
        "delivery_model_path": "models/m3_test.joblib",
        "model_sha256": "hash",
        "bayes_best_params": '{"estimator__max_depth": 2}',
        "feature_columns_json": '["BP_RP","parallax_over_error"]',
        "dataset_variant": "strict_poe_2",
        "n_train": 110,
        "wr_train": 10,
        "holdout_fraction": 0.2,
        "split_policy": "stable",
        "negative_reduction_method": "stratified",
        "lineage_negative_ratio": "10x",
        "imbalance_strategy": "resampling_only",
        "positive_class_weight": 1.0,
        "imbalance_parameter_json": '{}',
        "selection_status": "accepted",
        "delivery_assessment": "preferred_stable_component",
        "holdout_average_precision": 0.5,
        "holdout_roc_auc": 0.9,
        "holdout_recall_at_100": 0.7,
        "holdout_precision_at_100": 0.3,
        "overfit_warning_flag": False,
        "overfit_risk_score": 0.7,
        "run_id": "run",
        "dataset_sha256": "dataset",
        "reference_dataset_sha256": "reference",
        "models_config_sha256": "config",
        "locus_run_id": "locus",
        "require_color_locus_keep": True,
    }
    plane = SimpleNamespace(
        name="G_BP__BP_RP", x="G_BP", y="BP_RP", slope=1.0,
        intercept=0.0, threshold=0.2, transform="signed_log1p",
    )
    locus = SimpleNamespace(
        planes=[plane], aggregate_min_outlier_planes=2, source_sha256="locus_hash"
    )
    filters = {
        "photometry": {
            "families": {
                "strict": {
                    "twomass_allowed_qualities": ["A"],
                    "wise_allowed_qualities": ["A"],
                    "description": "unused",
                }
            }
        }
    }

    payload = _model_contract_json(row, exact_locus=locus, filters_config=filters)

    assert payload["hiperparametros"]["estimator__max_depth"] == 2
    assert payload["columnas_entrenamiento"] == ["BP_RP", "parallax_over_error"]
    assert payload["elegibilidad_prediction_pool"]["astrometria"]["parallax_over_error_minimo"] == 2.0
    assert payload["elegibilidad_prediction_pool"]["fotometria"]["calidad_2mass_permitida"] == ["A"]
    assert payload["locus_color"]["w1_w2_define_plano_locus"] is False
    assert payload["validacion"]["estado_seleccion"] == "aceptado"


def test_dictionary_is_single_spanish_catalog_for_both_csvs():
    models = pd.DataFrame([{"delivery_key": "m1_test"}])
    dictionary = _data_dictionary(
        ["eligibility_rank", "m1_test_score", "simbad_url"],
        ["orden", "average_precision_holdout"],
        models,
    )

    assert set(dictionary["archivo"]) == {
        "predictions_all_eligibility_ranked.csv / predictions_top_100.csv",
        "metricas_modelos.csv",
    }
    assert dictionary["descripcion"].str.len().gt(10).all()
    assert dictionary.loc[dictionary["columna"].eq("m1_test_score"), "descripcion"].iloc[0].startswith("Score bruto")
