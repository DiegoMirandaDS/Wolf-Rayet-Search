"""Build a compact, auditable handoff for prediction-pool candidates and models."""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import struct
from typing import Any, Iterable, Mapping
import zipfile

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from wr_detector.config import PROJECT_ROOT, load_yaml, resolve_path
from wr_detector.modeling.explorer import add_ranking_score
from wr_detector.pipelines.prediction_pool_candidates import (
    file_sha256,
    load_candidate_review_config,
)
from wr_detector.pipelines.exact_variant_union import load_exact_loci_from_exports
from wr_detector.pipelines.prediction_pool_scoring import (
    load_prediction_scoring_config,
)


COMPACT_FEATURE_COLUMNS = (
    "gaia_designation",
    "ra",
    "dec",
    "galactic_l",
    "galactic_b",
    "G",
    "BP_RP",
    "J_K",
    "W1_W2",
    "parallax",
    "parallax_error",
    "parallax_over_error",
    "pmra",
    "pmdec",
    "ruwe",
    "astrometric_excess_noise",
    "phot_bp_rp_excess_factor",
    "phot_bp_n_blended_transits",
    "phot_rp_n_blended_transits",
    "duplicated_source",
    "phot_variable_flag",
    "tmass_quality",
    "tmass_angular_distance",
    "wise_quality",
    "wise_angular_distance",
    "wise_cc_flags",
    "wise_ext_flag",
    "ag_gspphot",
    "halpha_ew",
    "halpha_ew_error",
    "halpha_ew_flag",
    "espels_class",
    "espels_wc_probability",
    "espels_wn_probability",
    "espels_be_probability",
    "espels_pne_probability",
    "compatible_variant_count",
    "compatible_variant_mask",
)

SIMBAD_COLUMNS = (
    "simbad_main_id",
    "simbad_main_type",
    "simbad_other_types",
    "simbad_sp_type",
    "match_method",
    "match_count",
    "separation_arcsec",
    "review_disposition",
    "review_comment",
)


def build_prediction_pool_delivery(
    config_path: str | Path,
) -> dict[str, Any]:
    """Export candidate predictions, model artifacts and pool inventory."""
    config_path = Path(config_path)
    config = load_candidate_review_config(config_path)
    delivery = dict(config.get("delivery", {}))
    if not delivery:
        raise KeyError("Candidate config is missing the delivery section.")

    candidate_db = resolve_path(config["outputs"]["database"])
    scoring_config_path = resolve_path(config["scoring_config"])
    scoring_config = load_prediction_scoring_config(scoring_config_path)
    scoring_db = resolve_path(scoring_config["output_db"])
    pool_config_path = resolve_path(scoring_config["pool_config"])
    pool_config = load_yaml(pool_config_path)
    pool_db = resolve_path(pool_config["output_db"])
    training_db = resolve_path(config["training_history_db"])
    for path in [candidate_db, scoring_db, pool_db, training_db]:
        if not path.exists():
            raise FileNotFoundError(path)

    output_dir = resolve_path(delivery["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    _remove_obsolete_delivery_artifacts(output_dir)
    model_dir = output_dir / "models"
    graphics_dir = output_dir / "graficos_modelos"
    config_dir = output_dir / "configs"
    for path in [model_dir, graphics_dir, config_dir]:
        path.mkdir(parents=True, exist_ok=True)

    review = config["review"]
    scoring_run_id = str(review["scoring_run_id"])
    review_run_id = str(review["review_run_id"])
    top_limit = int(delivery.get("top_limit", 100))
    if top_limit <= 0:
        raise ValueError("delivery.top_limit must be positive.")

    consensus, opportunities, review_frame = _load_candidate_tables(candidate_db)
    if not {
        "eligibility_rank",
        "eligibility_normalized_rrf",
        "eligible_model_count",
        "eligible_support_fraction",
    }.issubset(consensus.columns):
        raise ValueError(
            "Candidate review predates eligibility-aware ranking; rebuild it first."
        )
    model_metadata = _load_delivery_model_metadata(
        scoring_db,
        training_db,
        scoring_run_id=scoring_run_id,
        model_run_id=str(review["model_run_id"]),
        selected_models=[dict(item) for item in review["models"]],
    )
    candidate_scores = _load_candidate_scores(
        scoring_db,
        scoring_run_id=scoring_run_id,
        source_ids=consensus["source_id"].astype("int64").tolist(),
    )
    pool_features = _load_pool_features(
        pool_db,
        consensus["source_id"].astype("int64").tolist(),
    )
    predictions = _make_prediction_export(
        consensus,
        opportunities,
        pool_features,
        review_frame,
        model_metadata,
        candidate_scores,
    )
    top_predictions = predictions.head(top_limit).copy()

    all_path = output_dir / "predictions_all_eligibility_ranked.csv"
    top_path = output_dir / f"predictions_top_{top_limit}.csv"
    dictionary_path = output_dir / "diccionario_columnas.csv"
    model_metrics_path = output_dir / "metricas_modelos.csv"
    _atomic_csv(predictions, all_path)
    _atomic_csv(top_predictions, top_path)
    filters_config = load_yaml(resolve_path(pool_config["filters_config"]))
    paths_config = load_yaml(resolve_path(filters_config["paths_config"]))
    selected_variants = sorted(set(model_metadata["dataset_variant"].astype(str)))
    exact_loci = load_exact_loci_from_exports(
        reference_dir=resolve_path(paths_config["processed_reference_dir"]),
        reference_output_template=filters_config["color_locus"][
            "reference_output_template"
        ],
        variants=selected_variants,
        aggregate_min_outlier_planes=int(
            filters_config["color_locus"]["aggregate_min_outlier_planes"]
        ),
    )
    delivered_models = _copy_model_artifacts(
        model_metadata,
        model_dir=model_dir,
        graphics_dir=graphics_dir,
        training_db=training_db,
        exact_loci=exact_loci,
        filters_config=filters_config,
    )
    model_metrics = _make_model_metrics(delivered_models)
    _atomic_csv(model_metrics, model_metrics_path)
    dictionary = _data_dictionary(
        predictions.columns,
        model_metrics.columns,
        delivered_models,
    )
    _atomic_csv(dictionary, dictionary_path)
    _copy_lineage_files(
        [
            config_path,
            scoring_config_path,
            pool_config_path,
            resolve_path(scoring_config["models_config"]),
            resolve_path(pool_config["filters_config"]),
            resolve_path(filters_config["paths_config"]),
        ],
        config_dir,
    )
    pool_bytes = _prediction_pool_bytes(pool_db)

    export_files = [
        all_path,
        top_path,
        dictionary_path,
        model_metrics_path,
        *model_dir.glob("*"),
        *graphics_dir.glob("*.png"),
        *config_dir.glob("*"),
    ]
    workbook_path = output_dir / "wr_candidates_top_100.xlsx"
    if workbook_path.exists():
        export_files.append(workbook_path)
    archive_path = output_dir / "wr_candidates_and_top5_models.zip"
    _write_zip(archive_path, output_dir, export_files)
    audit = audit_prediction_pool_delivery(config_path)
    if not audit["ok"]:
        raise RuntimeError("Prediction delivery audit failed: " + "; ".join(audit["errors"]))
    return {
        "delivery_directory": str(output_dir),
        "archive_path": str(archive_path),
        "prediction_rows": len(predictions),
        "top_rows": len(top_predictions),
        "pool_bytes": pool_bytes,
        "audit": audit,
    }


def audit_prediction_pool_delivery(config_path: str | Path) -> dict[str, Any]:
    config = load_candidate_review_config(config_path)
    delivery = config.get("delivery", {})
    output_dir = resolve_path(delivery["directory"])
    top_limit = int(delivery.get("top_limit", 100))
    all_path = output_dir / "predictions_all_eligibility_ranked.csv"
    top_path = output_dir / f"predictions_top_{top_limit}.csv"
    model_metrics_path = output_dir / "metricas_modelos.csv"
    dictionary_path = output_dir / "diccionario_columnas.csv"
    archive_path = output_dir / "wr_candidates_and_top5_models.zip"
    errors: list[str] = []
    for path in [
        all_path,
        top_path,
        model_metrics_path,
        dictionary_path,
        archive_path,
    ]:
        if not path.exists():
            errors.append(f"missing={path.name}")
    if errors:
        return {"ok": False, "errors": errors, "directory": str(output_dir)}

    predictions = pd.read_csv(
        all_path,
        dtype={"source_id": "string"},
        low_memory=False,
    )
    top = pd.read_csv(
        top_path,
        dtype={"source_id": "string"},
        low_memory=False,
    )
    model_metrics = pd.read_csv(model_metrics_path)
    if predictions["source_id"].duplicated().any():
        errors.append("duplicate_source_ids")
    if predictions["eligibility_rank"].tolist() != list(
        range(1, len(predictions) + 1)
    ):
        errors.append("non_contiguous_eligibility_rank")
    if not top.equals(predictions.head(top_limit)):
        errors.append("top_export_is_not_exact_truncation")
    if len(model_metrics) != len(config["review"]["models"]):
        errors.append("model_count_mismatch")
    if model_metrics.get("clave_modelo", pd.Series(dtype="object")).duplicated().any():
        errors.append("duplicate_model_metrics_delivery_key")
    if "grafico_modelo" in model_metrics:
        for relative in model_metrics["grafico_modelo"].astype(str):
            graphic = output_dir / relative
            if not graphic.is_file():
                errors.append(f"missing_model_graphic={relative}")
            else:
                width, height = _png_dimensions(graphic)
                if width < 1600 or height < 1000:
                    errors.append(
                        f"incomplete_model_graphic={relative}:{width}x{height}"
                    )
    else:
        errors.append("missing_model_graphic_column")
    status_columns = [
        f"{item['delivery_key']}_status" for item in config["review"]["models"]
    ]
    allowed = {
        "ranked_top_10000",
        "eligible_below_top_10000",
        "ineligible_variant",
    }
    for column in status_columns:
        if not set(predictions[column].dropna().unique()).issubset(allowed):
            errors.append(f"invalid_status_values={column}")
        prefix = column.removesuffix("_status")
        ineligible = predictions[column].eq("ineligible_variant")
        component_columns = [
            f"{prefix}_locus_keep",
            f"{prefix}_photometry_keep",
            f"{prefix}_astrometry_keep",
        ]
        missing_components = [
            item for item in component_columns if item not in predictions.columns
        ]
        if missing_components:
            errors.append(f"missing_component_columns={prefix}")
            continue
        passed_all = predictions[component_columns].fillna(False).all(axis=1)
        if not passed_all.eq(~ineligible).all():
            errors.append(f"component_status_mismatch={prefix}")
        if predictions.loc[ineligible, f"{prefix}_rank"].notna().any():
            errors.append(f"ineligible_rank_not_null={prefix}")
        if predictions.loc[ineligible, f"{prefix}_score"].notna().any():
            errors.append(f"ineligible_score_not_null={prefix}")
        if predictions.loc[~ineligible, f"{prefix}_score"].isna().any():
            errors.append(f"eligible_score_is_null={prefix}")
    model_jsons = sorted((output_dir / "models").glob("*.json"))
    if len(model_jsons) != len(config["review"]["models"]):
        errors.append("model_json_count_mismatch")
    required_json_sections = {
        "modelo",
        "hiperparametros",
        "columnas_entrenamiento",
        "elegibilidad_prediction_pool",
        "locus_color",
        "entrenamiento",
        "validacion",
        "linaje",
    }
    for path in model_jsons:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not required_json_sections.issubset(payload):
            errors.append(f"incomplete_model_json={path.name}")
        artifact = output_dir / str(payload.get("modelo", {}).get("archivo", ""))
        if not artifact.is_file():
            errors.append(f"missing_model_artifact={path.name}")
        elif file_sha256(artifact) != payload.get("modelo", {}).get("sha256"):
            errors.append(f"model_hash_mismatch={path.name}")
    expected_configs = {
        "prediction_pool_candidates.yaml",
        "prediction_pool_scoring.yaml",
        "prediction_pool_exact_union.yaml",
        "models.yaml",
        "filters.yaml",
        "paths.yaml",
    }
    delivered_configs = {path.name for path in (output_dir / "configs").glob("*.yaml")}
    if delivered_configs != expected_configs:
        errors.append("unexpected_config_set")
    forbidden = {
        "README.md",
        "delivery_manifest.json",
        "data_dictionary.csv",
        "model_evaluation.csv",
        "models_metrics.csv",
        "prediction_pool_inventory.csv",
    }
    for name in forbidden:
        if (output_dir / name).exists():
            errors.append(f"obsolete_delivery_file={name}")
    for obsolete_dir in ["model_diagnostics", "model_details"]:
        if (output_dir / obsolete_dir).exists():
            errors.append(f"obsolete_delivery_directory={obsolete_dir}")
    portable_files = [*output_dir.glob("*.csv"), *model_jsons]
    for path in portable_files:
        if _contains_personal_path(path.read_text(encoding="utf-8")):
            errors.append(f"personal_path_leak={path.name}")
    if archive_path.exists():
        with zipfile.ZipFile(archive_path) as bundle:
            archived = set(bundle.namelist())
            expected = {
                path.relative_to(output_dir).as_posix()
                for path in output_dir.rglob("*")
                if path.is_file()
                and path != archive_path
                and not path.name.startswith("~$")
            }
            if archived != expected:
                errors.append("archive_content_mismatch")
            if bundle.testzip() is not None:
                errors.append("archive_crc_failure")
    return {
        "ok": not errors,
        "errors": errors,
        "directory": str(output_dir),
        "prediction_rows": int(len(predictions)),
        "top_rows": int(len(top)),
        "models": int(len(model_metrics)),
    }


def _load_candidate_tables(
    candidate_db: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with duckdb.connect(str(candidate_db), read_only=True) as con:
        tables = {str(row[0]) for row in con.execute("SHOW TABLES").fetchall()}
        required = {
            "candidate_consensus",
            "candidate_model_eligibility",
            "candidate_review",
        }
        missing = sorted(required - tables)
        if missing:
            raise ValueError(f"Candidate database is missing tables: {missing}")
        consensus = con.execute(
            "SELECT * FROM candidate_consensus ORDER BY eligibility_rank"
        ).fetchdf()
        opportunities = con.execute(
            "SELECT * FROM candidate_model_eligibility"
        ).fetchdf()
        review = con.execute("SELECT * FROM candidate_review").fetchdf()
    return consensus, opportunities, review


def _load_delivery_model_metadata(
    scoring_db: Path,
    training_db: Path,
    *,
    scoring_run_id: str,
    model_run_id: str,
    selected_models: list[dict[str, Any]],
) -> pd.DataFrame:
    result_ids = [str(item["result_id"]) for item in selected_models]
    placeholders = ",".join("?" for _ in result_ids)
    with duckdb.connect(str(scoring_db), read_only=True) as con:
        scoring = con.execute(
            f"""
            SELECT result_id, dataset_variant, variant_bit
            FROM prediction_scoring_models
            WHERE scoring_run_id=? AND result_id IN ({placeholders})
            """,
            [scoring_run_id, *result_ids],
        ).fetchdf()
    with duckdb.connect(str(training_db), read_only=True) as con:
        metrics = con.execute(
            f"""
            SELECT result_id, run_id, feature_set, model, sampler,
                   selection_status, selected_threshold,
                   threshold_precision_floor_met,
                   wr_holdout, n_holdout,
                   holdout_average_precision, holdout_roc_auc,
                   train_average_precision, cv_average_precision,
                   train_roc_auc, cv_roc_auc,
                   train_precision_wr, cv_precision_wr,
                   train_recall_wr, cv_recall_wr,
                   train_f2_wr, cv_f2_wr,
                   holdout_precision_wr, holdout_recall_wr, holdout_f2_wr,
                   holdout_balanced_accuracy, holdout_accuracy,
                   holdout_tn, holdout_fp, holdout_fn, holdout_tp,
                   holdout_precision_at_10, holdout_recall_at_10,
                   holdout_wr_at_10,
                   holdout_precision_at_50, holdout_recall_at_50,
                   holdout_wr_at_50,
                   holdout_precision_at_100, holdout_recall_at_100,
                   holdout_wr_at_100,
                   holdout_precision_at_500, holdout_recall_at_500,
                   holdout_wr_at_500,
                   holdout_recall_at_fpr_0p001,
                   holdout_recall_at_fpr_0p005,
                   threshold_calibration_negative_pass_rate,
                   cv_train_gap_f2, holdout_cv_gap_f2,
                   train_cv_gap_average_precision,
                   cv_holdout_drop_average_precision,
                   overfit_warning_flag, overfit_risk_score,
                   bayes_best_params, feature_columns_json,
                   n_train, wr_train, sampler_type, imbalance_strategy,
                   positive_class_weight, imbalance_parameter_json,
                   dataset_sha256, models_config_sha256,
                   reference_dataset_sha256, locus_run_id,
                   require_color_locus_keep, holdout_fraction, split_policy,
                   negative_reduction_method, lineage_negative_ratio,
                   model_path, model_sha256,
                   holdout_pr_curve_path, dataset_path
            FROM model_results
            WHERE result_id IN ({placeholders})
            """,
            result_ids,
        ).fetchdf()
    configured = pd.DataFrame(selected_models)
    configured["model_order"] = np.arange(1, len(configured) + 1)
    out = configured.merge(scoring, on="result_id", validate="one_to_one")
    out = out.merge(
        metrics,
        on="result_id",
        validate="one_to_one",
        suffixes=("_configured", ""),
    )
    if set(out["run_id"].astype(str)) != {model_run_id}:
        raise ValueError("Delivery models do not belong to the configured run.")
    if "delivery_key" not in out:
        out["delivery_key"] = out.apply(
            lambda row: f"m{int(row['model_order'])}_{_slug(row['role'])}",
            axis=1,
        )
    if out["delivery_key"].duplicated().any():
        raise ValueError("Model delivery keys must be unique.")
    return out.sort_values("model_order").reset_index(drop=True)


def _load_candidate_scores(
    scoring_db: Path,
    *,
    scoring_run_id: str,
    source_ids: Iterable[int],
) -> pd.DataFrame:
    ids = pd.DataFrame({"source_id": pd.Series(list(source_ids), dtype="int64")})
    with duckdb.connect(str(scoring_db), read_only=True) as con:
        con.register("candidate_ids", ids)
        frame = con.execute(
            """
            SELECT score.source_id, score.result_id, score.score,
                   score.predicted, score.threshold
            FROM prediction_pool_scores score
            INNER JOIN candidate_ids ids USING (source_id)
            WHERE score.scoring_run_id=?
            """,
            [scoring_run_id],
        ).fetchdf()
    if frame.duplicated(["source_id", "result_id"]).any():
        raise ValueError("Full candidate scores contain duplicate source/model rows.")
    return frame


def _scoring_pool_build_id(scoring_db: Path, scoring_run_id: str) -> str:
    with duckdb.connect(str(scoring_db), read_only=True) as con:
        row = con.execute(
            """
            SELECT pool_build_id
            FROM prediction_scoring_runs
            WHERE scoring_run_id=?
            """,
            [scoring_run_id],
        ).fetchone()
    if row is None:
        raise ValueError(f"Unknown scoring run: {scoring_run_id}")
    return str(row[0])


def _load_pool_features(pool_db: Path, source_ids: Iterable[int]) -> pd.DataFrame:
    ids = pd.DataFrame({"source_id": pd.Series(list(source_ids), dtype="int64")})
    columns = ", ".join(f"source.{column}" for column in COMPACT_FEATURE_COLUMNS)
    with duckdb.connect() as con:
        con.execute(f"ATTACH '{pool_db.as_posix()}' AS pool (READ_ONLY)")
        con.register("candidate_ids", ids)
        frame = con.execute(
            f"""
            SELECT source.source_id, {columns}
            FROM pool.prediction_pool_sources source
            INNER JOIN candidate_ids ids USING (source_id)
            ORDER BY source.source_id
            """
        ).fetchdf()
    if len(frame) != len(ids) or frame["source_id"].nunique() != len(ids):
        raise ValueError("Pool feature export did not preserve source grain.")
    return frame


def _make_prediction_export(
    consensus: pd.DataFrame,
    opportunities: pd.DataFrame,
    features: pd.DataFrame,
    review: pd.DataFrame,
    models: pd.DataFrame,
    candidate_scores: pd.DataFrame,
) -> pd.DataFrame:
    ranking_columns = [
        "eligibility_rank",
        "consensus_rank",
        "source_id",
        "eligibility_normalized_rrf",
        "rrf_score",
        "eligible_model_count",
        "ranked_model_count",
        "eligible_support_fraction",
        "model_support",
        "variant_support",
        "estimator_support",
        "best_model_rank",
        "mean_model_rank",
    ]
    output = consensus[ranking_columns].merge(
        features,
        on="source_id",
        how="left",
        validate="one_to_one",
    )
    output.insert(
        2,
        "gaia_dr3_id",
        output["source_id"].map(lambda value: f"Gaia DR3 {int(value)}"),
    )
    opportunities = opportunities.drop(
        columns=["score", "predicted", "threshold"],
        errors="ignore",
    ).merge(
        candidate_scores,
        on=["source_id", "result_id"],
        how="left",
        validate="one_to_one",
    )
    if opportunities.loc[opportunities["eligible"], "score"].isna().any():
        raise ValueError("At least one eligible candidate lacks its model score.")
    if opportunities.loc[~opportunities["eligible"], "score"].notna().any():
        raise ValueError("An ineligible candidate unexpectedly has a model score.")

    for row in models.to_dict("records"):
        key = str(row["delivery_key"])
        subset = opportunities.loc[
            opportunities["result_id"].astype(str).eq(str(row["result_id"])),
            [
                "source_id",
                "model_status",
                "eligibility_reason",
                "locus_keep",
                "photometry_keep",
                "astrometry_keep",
                "model_rank",
                "score",
            ],
        ].rename(
            columns={
                "model_status": f"{key}_status",
                "eligibility_reason": f"{key}_eligibility_reason",
                "locus_keep": f"{key}_locus_keep",
                "photometry_keep": f"{key}_photometry_keep",
                "astrometry_keep": f"{key}_astrometry_keep",
                "model_rank": f"{key}_rank",
                "score": f"{key}_score",
            }
        )
        output = output.merge(subset, on="source_id", validate="one_to_one")

    available_review_columns = [
        "source_id",
        *[column for column in SIMBAD_COLUMNS if column in review.columns],
    ]
    output = output.merge(
        review[available_review_columns],
        on="source_id",
        how="left",
        validate="one_to_one",
    )
    output["simbad_review_status"] = np.where(
        output["review_disposition"].notna(),
        output["review_disposition"],
        "not_queried_in_frozen_snapshot",
    )
    output["simbad_url"] = output["source_id"].map(
        lambda value: (
            "https://simbad.cds.unistra.fr/simbad/sim-id?"
            f"Ident=Gaia+DR3+{int(value)}"
        )
    )
    output["gaia_archive_url"] = "https://gea.esac.esa.int/archive/"
    output["aladin_url"] = output.apply(
        lambda row: (
            "https://aladin.cds.unistra.fr/AladinLite/?"
            f"target={row['ra']}%20{row['dec']}&fov=0.05&survey=P%2FDSS2%2Fcolor"
        ),
        axis=1,
    )
    output["esasky_url"] = output.apply(
        lambda row: (
            "https://sky.esa.int/esasky/?"
            f"target={row['ra']}%20{row['dec']}&fov=0.1"
        ),
        axis=1,
    )
    preferred = [
        "eligibility_rank",
        "consensus_rank",
        "gaia_dr3_id",
        "source_id",
        "eligibility_normalized_rrf",
        "rrf_score",
        "eligible_model_count",
        "ranked_model_count",
        "eligible_support_fraction",
        "best_model_rank",
    ]
    model_columns = [
        column
        for row in models.to_dict("records")
        for column in [
            f"{row['delivery_key']}_status",
            f"{row['delivery_key']}_eligibility_reason",
            f"{row['delivery_key']}_locus_keep",
            f"{row['delivery_key']}_photometry_keep",
            f"{row['delivery_key']}_astrometry_keep",
            f"{row['delivery_key']}_rank",
            f"{row['delivery_key']}_score",
        ]
    ]
    remaining = [
        column for column in output.columns if column not in preferred + model_columns
    ]
    output = output[preferred + model_columns + remaining]
    return output.sort_values("eligibility_rank", kind="mergesort").reset_index(drop=True)


def _copy_model_artifacts(
    models: pd.DataFrame,
    *,
    model_dir: Path,
    graphics_dir: Path,
    training_db: Path,
    exact_loci: Mapping[str, Any],
    filters_config: Mapping[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in models.to_dict("records"):
        key = str(row["delivery_key"])
        source_model = resolve_path(row["model_path"])
        if not source_model.exists() or file_sha256(source_model) != row["model_sha256"]:
            raise ValueError(f"Model artifact failed hash validation: {row['result_id']}")
        target_model = model_dir / f"{key}.joblib"
        shutil.copy2(source_model, target_model)
        assessment = (
            "preferred_stable_component"
            if str(row["selection_status"]) == "accepted"
            and not bool(row["overfit_warning_flag"])
            else "ranking_use_with_caveats"
        )
        delivered = {
                **row,
                "model_path": _project_relative_path(row["model_path"]),
                "delivery_model_path": f"models/{target_model.name}",
                "delivery_model_bytes": target_model.stat().st_size,
                "model_graphic_path": f"graficos_modelos/{key}_graficos.png",
                "delivery_assessment": assessment,
        }
        json_path = model_dir / f"{key}.json"
        delivered["delivery_model_json_path"] = f"models/{json_path.name}"
        _atomic_json(
            json_path,
            _model_contract_json(
                delivered,
                exact_locus=exact_loci[str(row["dataset_variant"])],
                filters_config=filters_config,
            ),
        )
        _render_model_graphic(
            delivered,
            training_db=training_db,
            output_path=graphics_dir / f"{key}_graficos.png",
        )
        rows.append(delivered)
    return pd.DataFrame(rows)


def _render_model_graphic(
    row: Mapping[str, Any],
    *,
    training_db: Path,
    output_path: Path,
) -> None:
    """Render the four essential model diagnostics without application chrome."""
    with duckdb.connect(str(training_db), read_only=True) as con:
        holdout = con.execute(
            """
            SELECT source_id, target, score
            FROM model_predictions
            WHERE result_id=? AND split='holdout'
            ORDER BY row_id
            """,
            [str(row["result_id"])],
        ).fetchdf()
        importance = con.execute(
            """
            SELECT feature, importance_mean, importance_std, importance_type
            FROM feature_importance
            WHERE result_id=?
            ORDER BY importance_mean DESC, feature
            """,
            [str(row["result_id"])],
        ).fetchdf()
    if holdout.empty or holdout["target"].nunique() < 2:
        raise ValueError(f"Missing two-class holdout predictions: {row['result_id']}")
    if importance.empty:
        raise ValueError(f"Missing synchronized feature importance: {row['result_id']}")

    y_true = holdout["target"].to_numpy(dtype=int)
    y_score = holdout["score"].to_numpy(dtype=float)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
    threshold = float(row["selected_threshold"])
    predicted = y_score >= threshold
    tp = int(np.sum((predicted == 1) & (y_true == 1)))
    fp = int(np.sum((predicted == 1) & (y_true == 0)))
    fn = int(np.sum((predicted == 0) & (y_true == 1)))
    tn = int(np.sum((predicted == 0) & (y_true == 0)))
    point_precision = tp / (tp + fp) if tp + fp else 0.0
    point_recall = tp / (tp + fn) if tp + fn else 0.0
    point_fpr = fp / (fp + tn) if fp + tn else 0.0

    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=120)
    fig.patch.set_facecolor("#ffffff")
    for ax in axes.flat:
        ax.set_facecolor("#ffffff")
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.8, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

    metrics = ["AP", "ROC AUC", "Precisión", "Recall", "F2"]
    stages = ["train", "cv", "holdout"]
    colors = ["#4e79a7", "#edc948", "#e15759"]
    values = {
        stage: [float(row[f"{stage}_{suffix}"]) for suffix in (
            "average_precision", "roc_auc", "precision_wr", "recall_wr", "f2_wr"
        )]
        for stage in stages
    }
    x = np.arange(len(metrics))
    width = 0.24
    for index, (stage, color) in enumerate(zip(stages, colors)):
        axes[0, 0].bar(x + (index - 1) * width, values[stage], width,
                       label=stage, color=color, zorder=2)
    axes[0, 0].set_title("Estabilidad train / CV / holdout", loc="left", weight="bold")
    axes[0, 0].set_xticks(x, metrics)
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].legend(frameon=False, ncols=3, loc="lower left")

    importance = importance.sort_values("importance_mean", ascending=True)
    errors = importance["importance_std"].fillna(0.0).to_numpy(dtype=float)
    axes[0, 1].barh(
        importance["feature"], importance["importance_mean"],
        xerr=errors if np.any(errors) else None,
        color="#4e79a7", ecolor="#6b7280", capsize=2,
    )
    axes[0, 1].set_title("Importancia de variables", loc="left", weight="bold")
    importance_type = str(importance["importance_type"].iloc[0])
    axes[0, 1].set_xlabel(f"importancia · {importance_type}")
    axes[0, 1].grid(axis="x", color="#e5e7eb", linewidth=0.8)
    axes[0, 1].grid(axis="y", visible=False)

    axes[1, 0].step(recall, precision, where="post", color="#f28e2b", linewidth=2)
    axes[1, 0].scatter([point_recall], [point_precision], color="#e15759", s=45, zorder=3)
    axes[1, 0].set_title(
        f"Precision–recall | AP {float(row['holdout_average_precision']):.3f}",
        loc="left", weight="bold",
    )
    axes[1, 0].set(xlabel="recall", ylabel="precisión", xlim=(0, 1), ylim=(0, 1.03))

    axes[1, 1].plot(fpr, tpr, color="#4e79a7", linewidth=2)
    axes[1, 1].plot([0, 1], [0, 1], color="#6b7280", linestyle="--", linewidth=1)
    axes[1, 1].scatter([point_fpr], [point_recall], color="#e15759", s=45, zorder=3)
    axes[1, 1].set_title(
        f"ROC | AUC {float(row['holdout_roc_auc']):.3f}",
        loc="left", weight="bold",
    )
    axes[1, 1].set(xlabel="tasa de falsos positivos", ylabel="tasa de verdaderos positivos",
                   xlim=(0, 1), ylim=(0, 1.03))

    label = f"M{int(row['model_order'])} · {row['model']} / {row['sampler']} · {row['dataset_variant']}"
    fig.suptitle(label, x=0.06, y=0.985, ha="left", fontsize=16, weight="bold")
    fig.text(
        0.06, 0.952,
        f"Holdout: {int(row['n_holdout']):,} objetos, {int(row['wr_holdout'])} WR · "
        f"punto rojo: umbral operativo {threshold:.4f}",
        ha="left", color="#4b5563", fontsize=10,
    )
    fig.tight_layout(rect=(0.04, 0.03, 0.99, 0.93), h_pad=3.0, w_pad=3.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _model_metric_descriptions() -> dict[str, str]:
    return {
        "orden": "Orden M1–M5 usado en el entregable.",
        "clave_modelo": "Nombre corto estable que vincula CSV, joblib, JSON y captura.",
        "nombre_modelo": "Etiqueta compacta con estimador, muestreo, variante y variables.",
        "rol_en_consenso": "Motivo científico por el que el modelo fue incluido en el conjunto.",
        "estado_seleccion": "Resultado de los chequeos de selección del entrenamiento.",
        "evaluacion_cientifica": "Lectura breve de utilidad y cautelas del modelo.",
        "variante_datos": "Contrato de locus, fotometría y astrometría usado por el modelo.",
        "conjunto_variables": "Nombre del conjunto de columnas de entrada.",
        "algoritmo": "Familia del estimador entrenado.",
        "muestreo": "Estrategia de remuestreo aplicada al entrenamiento.",
        "puntuacion_seleccion": "Score compuesto utilizado para comparar modelos en el Explorer.",
        "filas_entrenamiento": "Cantidad total de filas usadas para ajustar el modelo.",
        "wr_entrenamiento": "Cantidad de estrellas WR en entrenamiento.",
        "filas_holdout": "Cantidad total de filas en el holdout independiente.",
        "wr_holdout": "Cantidad de estrellas WR en el holdout.",
        "average_precision_holdout": "Área bajo la curva precision–recall en holdout.",
        "roc_auc_holdout": "Área bajo la curva ROC en holdout.",
        "wr_top_10": "WR recuperadas entre los diez primeros objetos del holdout.",
        "recall_top_10": "Fracción de WR del holdout recuperada en el top 10.",
        "wr_top_50": "WR recuperadas entre los cincuenta primeros objetos del holdout.",
        "recall_top_50": "Fracción de WR del holdout recuperada en el top 50.",
        "wr_top_100": "WR recuperadas entre los cien primeros objetos del holdout.",
        "precision_top_100": "Fracción de WR dentro del top 100 del holdout.",
        "recall_top_100": "Fracción de WR del holdout recuperada en el top 100.",
        "recall_fpr_0_5_pct": "Recall alcanzable con una tasa de falsos positivos de 0,5%.",
        "tasa_paso_negativos_calibracion": "Fracción del pool negativo de calibración que supera el umbral.",
        "average_precision_train": "Average precision medida sobre entrenamiento.",
        "average_precision_cv": "Average precision media de validación cruzada.",
        "alerta_sobreajuste": "Indicador compuesto de posible sobreajuste o inestabilidad.",
        "riesgo_sobreajuste": "Puntuación continua del diagnóstico de sobreajuste; menor es mejor.",
        "archivo_modelo": "Ruta relativa al archivo joblib dentro del paquete.",
        "sha256_modelo": "Hash SHA-256 del joblib entregado.",
        "json_modelo": "Ruta al JSON autocontenido con hiperparámetros y contrato de entrada.",
        "grafico_modelo": "Ruta al PNG limpio con estabilidad, importancia, precision-recall y ROC.",
    }


def _dictionary_type(column: str) -> str:
    if column in {"source_id", "gaia_dr3_id", "gaia_designation"} or column.endswith("_id"):
        return "texto"
    if column.endswith("_keep") or column in {
        "duplicated_source", "alerta_sobreajuste"
    }:
        return "booleano"
    if column.endswith("_url") or column.startswith("archivo_") or column in {
        "json_modelo", "grafico_modelo", "sha256_modelo"
    }:
        return "texto"
    integer_tokens = ("rank", "count", "transits", "orden", "filas_", "wr_")
    if any(token in column for token in integer_tokens) and not column.startswith("mean_"):
        return "entero"
    numeric_tokens = (
        "score", "rrf", "fraction", "probability", "parallax", "pmra", "pmdec",
        "ruwe", "noise", "factor", "distance", "separation", "halpha",
        "average_precision", "roc_auc", "recall", "precision", "tasa_",
        "riesgo_", "puntuacion_", "ra", "dec", "galactic_", "G", "BP_RP",
        "J_K", "W1_W2", "ag_gspphot",
    )
    return "decimal" if any(token in column for token in numeric_tokens) else "texto"


def _dictionary_unit(column: str) -> str:
    if column in {"ra", "dec", "galactic_l", "galactic_b"}:
        return "grados"
    if column in {"parallax", "parallax_error", "astrometric_excess_noise"}:
        return "mas"
    if column in {"pmra", "pmdec"}:
        return "mas/año"
    if column in {"tmass_angular_distance", "wise_angular_distance", "separation_arcsec"}:
        return "arcsec"
    if column in {"G", "BP_RP", "J_K", "W1_W2", "ag_gspphot"}:
        return "mag"
    if column in {"halpha_ew", "halpha_ew_error"}:
        return "nm (Gaia DR3)"
    if any(token in column for token in ("fraction", "probability", "precision", "recall", "tasa_", "roc_auc", "average_precision")):
        return "fracción 0–1"
    return ""


def _json_value(value: Any, default: Any) -> Any:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _nullable_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _prediction_pool_bytes(pool_db: Path) -> int:
    with duckdb.connect(str(pool_db), read_only=True) as con:
        value = con.execute(
            "SELECT COALESCE(SUM(parquet_bytes), 0) FROM prediction_pool_effective_tiles"
        ).fetchone()[0]
    return int(value)


def _png_dimensions(path: Path) -> tuple[int, int]:
    header = path.read_bytes()[:24]
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not a valid PNG: {path}")
    return struct.unpack(">II", header[16:24])


def _remove_obsolete_delivery_artifacts(output_dir: Path) -> None:
    obsolete_files = [
        "README.md",
        "delivery_manifest.json",
        "data_dictionary.csv",
        "model_evaluation.csv",
        "models_metrics.csv",
        "prediction_pool_inventory.csv",
    ]
    for name in obsolete_files:
        (output_dir / name).unlink(missing_ok=True)
    for path in output_dir.glob("*.inspect.ndjson"):
        path.unlink()
    for obsolete_dir in ["model_diagnostics", "model_details"]:
        path = output_dir / obsolete_dir
        if path.exists():
            shutil.rmtree(path)
    for directory in [output_dir / "models", output_dir / "configs"]:
        if directory.exists():
            for path in directory.iterdir():
                if path.is_file():
                    path.unlink()


def _model_contract_json(
    row: Mapping[str, Any],
    *,
    exact_locus: Any,
    filters_config: Mapping[str, Any],
) -> dict[str, Any]:
    variant = str(row["dataset_variant"])
    family = "strict" if variant.startswith("strict_") else "relaxed"
    photometry = filters_config["photometry"]["families"][family]
    role_labels = {
        "broad_shortlist": "lista amplia",
        "average_precision_leader": "líder de average precision",
        "accepted_stable": "componente estable aceptado",
        "strict_photometry_counterpart": "contraparte fotométrica estricta",
        "estimator_diversity": "diversidad de estimador",
    }
    selection_labels = {
        "accepted": "aceptado",
        "overfit_warning": "alerta de sobreajuste",
        "holdout_precision_floor_not_met": "precisión holdout bajo el mínimo",
    }
    if variant.endswith("_photometry"):
        astrometry = {
            "requiere_paralaje_positiva": False,
            "parallax_over_error_minimo": None,
            "explicacion": "La variante fotométrica no impone corte astrométrico.",
        }
    else:
        suffix = variant.rsplit("_", 1)[-1]
        threshold = None if suffix == "soft" else float(suffix)
        astrometry = {
            "requiere_paralaje_positiva": True,
            "parallax_over_error_minimo": threshold,
            "explicacion": (
                "Exige parallax > 0."
                if threshold is None
                else f"Exige parallax > 0 y parallax_over_error >= {threshold:g}."
            ),
        }
    planes = [
        {
            "plano": plane.name,
            "eje_x": plane.x,
            "eje_y": plane.y,
            "pendiente": plane.slope,
            "intercepto": plane.intercept,
            "umbral_residuo_absoluto": plane.threshold,
            "transformacion": plane.transform,
        }
        for plane in exact_locus.planes
    ]
    return {
        "modelo": {
            "clave": str(row["delivery_key"]),
            "orden": int(row["model_order"]),
            "algoritmo": str(row["model"]),
            "muestreo": str(row["sampler"]),
            "rol_en_consenso": role_labels.get(str(row["role"]), str(row["role"])),
            "rol_en_consenso_codigo": str(row["role"]),
            "resultado_id": str(row["result_id"]),
            "archivo": str(row["delivery_model_path"]),
            "sha256": str(row["model_sha256"]),
            "scores_son_probabilidades_calibradas": False,
        },
        "hiperparametros": _json_value(row.get("bayes_best_params"), {}),
        "columnas_entrenamiento": _json_value(row.get("feature_columns_json"), []),
        "elegibilidad_prediction_pool": {
            "variante": variant,
            "requiere_features_finitas": True,
            "requiere_pasar_locus": True,
            "requiere_pasar_calidad_fotometrica": True,
            "requiere_pasar_astrometria": not variant.endswith("_photometry"),
            "fotometria": {
                "magnitudes_requeridas": ["G", "BP", "RP", "J", "H", "Ks", "W1", "W2"],
                "calidad_2mass_permitida": list(photometry["twomass_allowed_qualities"]),
                "calidad_wise_permitida": list(photometry["wise_allowed_qualities"]),
                "explicacion": (
                    "Exige Gaia G/BP/RP, 2MASS J/H/Ks y WISE W1/W2; "
                    + (
                        "todas las bandas infrarrojas requeridas deben tener calidad A."
                        if family == "strict"
                        else "las bandas infrarrojas requeridas pueden tener calidad A o B."
                    )
                ),
            },
            "astrometria": astrometry,
        },
        "locus_color": {
            "colores_en_planos": sorted(
                {value for plane in exact_locus.planes for value in (plane.x, plane.y)}
            ),
            "w1_w2_define_plano_locus": False,
            "regla_aceptacion": (
                f"Todos los colores de los planos deben ser finitos y la fuente debe "
                f"quedar fuera de menos de {exact_locus.aggregate_min_outlier_planes} planos."
            ),
            "minimo_planos_outlier_para_rechazar": int(
                exact_locus.aggregate_min_outlier_planes
            ),
            "planos": planes,
        },
        "entrenamiento": {
            "filas": int(row["n_train"]),
            "wr": int(row["wr_train"]),
            "fraccion_holdout": float(row["holdout_fraction"]),
            "politica_split": str(row["split_policy"]),
            "reduccion_negativos": str(row["negative_reduction_method"]),
            "razon_negativos": str(row["lineage_negative_ratio"]),
            "estrategia_desbalance": str(row["imbalance_strategy"]),
            "peso_clase_positiva": _nullable_float(row.get("positive_class_weight")),
            "parametros_desbalance": _json_value(
                row.get("imbalance_parameter_json"), {}
            ),
        },
        "validacion": {
            "estado_seleccion": selection_labels.get(
                str(row["selection_status"]), str(row["selection_status"])
            ),
            "estado_seleccion_codigo": str(row["selection_status"]),
            "evaluacion_entrega": str(row["delivery_assessment"]),
            "average_precision_holdout": float(row["holdout_average_precision"]),
            "roc_auc_holdout": float(row["holdout_roc_auc"]),
            "recall_top_100": float(row["holdout_recall_at_100"]),
            "precision_top_100": float(row["holdout_precision_at_100"]),
            "alerta_sobreajuste": bool(row["overfit_warning_flag"]),
            "riesgo_sobreajuste": float(row["overfit_risk_score"]),
        },
        "linaje": {
            "run_id": str(row["run_id"]),
            "dataset_sha256": str(row["dataset_sha256"]),
            "dataset_referencia_sha256": str(row["reference_dataset_sha256"]),
            "config_modelos_sha256": str(row["models_config_sha256"]),
            "locus_run_id": str(row["locus_run_id"]),
            "locus_sha256": str(exact_locus.source_sha256),
            "color_locus_keep_requerido": bool(row["require_color_locus_keep"]),
        },
    }


def _pool_inventory(pool_db: Path) -> pd.DataFrame:
    with duckdb.connect(str(pool_db), read_only=True) as con:
        frame = con.execute(
            """
            SELECT tile_id, status, written AS operational_rows,
                   parquet_path, parquet_bytes, acquisition_sha256,
                   manifest_path, manifest_sha256
            FROM prediction_pool_effective_tiles
            ORDER BY tile_id
            """
        ).fetchdf()
    frame["parquet_path"] = frame["parquet_path"].map(
        lambda value: f"prediction_pool/acquisition/{Path(str(value)).name}"
    )
    frame["manifest_path"] = frame["manifest_path"].map(
        lambda value: f"prediction_pool/manifests/{Path(str(value)).name}"
    )
    return frame


def _copy_lineage_files(paths: list[Path], output_dir: Path) -> list[str]:
    copied: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        target = output_dir / path.name
        shutil.copy2(path, target)
        copied.append(f"configs/{target.name}")
    return copied


def _make_model_metrics(models: pd.DataFrame) -> pd.DataFrame:
    """Create a compact Spanish comparison of the five delivered models."""
    out = models.copy()
    out["holdout_negative"] = out["n_holdout"] - out["wr_holdout"]
    out["holdout_wr_prevalence"] = out["wr_holdout"] / out["n_holdout"]
    out["holdout_fpr"] = out["holdout_fp"] / (
        out["holdout_fp"] + out["holdout_tn"]
    )
    out = add_ranking_score(out)
    estimator_labels = {
        "xgboost": "XGB",
        "hist_gradient_boosting": "HGB",
    }
    feature_labels = {
        "colors_parallax_error": "+err",
        "colors_parallax": "colors",
    }
    out["model_label"] = out.apply(
        lambda row: (
            f"M{int(row['model_order'])} | "
            f"{estimator_labels.get(str(row['model']), row['model'])}/"
            f"{row['sampler']} | {row['dataset_variant']} "
            f"{feature_labels.get(str(row['feature_set']), row['feature_set'])}"
        ),
        axis=1,
    )
    out["scientific_assessment"] = np.where(
        out["delivery_assessment"].eq("preferred_stable_component"),
        "Componente estable preferido: aceptado y sin alerta de sobreajuste.",
        "Componente útil del ranking, con cautelas de selección o estabilidad.",
    )
    out["role"] = out["role"].replace(
        {
            "broad_shortlist": "lista amplia",
            "average_precision_leader": "líder de average precision",
            "accepted_stable": "componente estable aceptado",
            "strict_photometry_counterpart": "contraparte fotométrica estricta",
            "estimator_diversity": "diversidad de estimador",
        }
    )
    out["selection_status"] = out["selection_status"].replace(
        {
            "accepted": "aceptado",
            "overfit_warning": "alerta de sobreajuste",
            "holdout_precision_floor_not_met": "precisión holdout bajo el mínimo",
        }
    )
    compact = out[
        [
            "model_order", "delivery_key", "model_label", "role",
            "selection_status", "scientific_assessment", "dataset_variant",
            "feature_set", "model", "sampler", "ranking_score", "n_train",
            "wr_train", "n_holdout", "wr_holdout", "holdout_average_precision",
            "holdout_roc_auc", "holdout_wr_at_10", "holdout_recall_at_10",
            "holdout_wr_at_50", "holdout_recall_at_50", "holdout_wr_at_100",
            "holdout_precision_at_100", "holdout_recall_at_100",
            "holdout_recall_at_fpr_0p005",
            "threshold_calibration_negative_pass_rate",
            "train_average_precision", "cv_average_precision",
            "overfit_warning_flag", "overfit_risk_score", "delivery_model_path",
            "model_sha256", "delivery_model_json_path",
            "model_graphic_path",
        ]
    ].rename(
        columns={
            "model_order": "orden",
            "delivery_key": "clave_modelo",
            "model_label": "nombre_modelo",
            "role": "rol_en_consenso",
            "selection_status": "estado_seleccion",
            "scientific_assessment": "evaluacion_cientifica",
            "dataset_variant": "variante_datos",
            "feature_set": "conjunto_variables",
            "model": "algoritmo",
            "sampler": "muestreo",
            "ranking_score": "puntuacion_seleccion",
            "n_train": "filas_entrenamiento",
            "wr_train": "wr_entrenamiento",
            "n_holdout": "filas_holdout",
            "wr_holdout": "wr_holdout",
            "holdout_average_precision": "average_precision_holdout",
            "holdout_roc_auc": "roc_auc_holdout",
            "holdout_wr_at_10": "wr_top_10",
            "holdout_recall_at_10": "recall_top_10",
            "holdout_wr_at_50": "wr_top_50",
            "holdout_recall_at_50": "recall_top_50",
            "holdout_wr_at_100": "wr_top_100",
            "holdout_precision_at_100": "precision_top_100",
            "holdout_recall_at_100": "recall_top_100",
            "holdout_recall_at_fpr_0p005": "recall_fpr_0_5_pct",
            "threshold_calibration_negative_pass_rate": "tasa_paso_negativos_calibracion",
            "train_average_precision": "average_precision_train",
            "cv_average_precision": "average_precision_cv",
            "overfit_warning_flag": "alerta_sobreajuste",
            "overfit_risk_score": "riesgo_sobreajuste",
            "delivery_model_path": "archivo_modelo",
            "model_sha256": "sha256_modelo",
            "delivery_model_json_path": "json_modelo",
            "model_graphic_path": "grafico_modelo",
        }
    )
    return compact.sort_values("orden").reset_index(drop=True)


def _project_relative_path(value: Any) -> str:
    path = resolve_path(str(value)).resolve()
    try:
        return path.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def _contains_personal_path(text: str) -> bool:
    return bool(
        re.search(r"(?i)(?:^|[\"',\s])[a-z]:[\\/]|[\\/]Users[\\/]", text)
    )


def _data_dictionary(
    prediction_columns: Iterable[str],
    model_metric_columns: Iterable[str],
    models: pd.DataFrame,
) -> pd.DataFrame:
    descriptions = {
        "eligibility_rank": "Ranking principal corregido por elegibilidad; 1 es el candidato prioritario.",
        "consensus_rank": "Ranking RRF original con pesos iguales, conservado sin modificaciones.",
        "gaia_dr3_id": "Identificador Gaia DR3 escrito como texto para evitar redondeo en Excel.",
        "source_id": "Identificador numérico único de Gaia DR3; debe tratarse como texto fuera de Python/DuckDB.",
        "eligibility_normalized_rrf": "RRF dividido por la cantidad de modelos que podían evaluar la fuente.",
        "rrf_score": "Suma RRF original: 1/(60 + rango del modelo) para cada lista donde apareció.",
        "eligible_model_count": "Cantidad de los cinco modelos cuyos filtros completos aceptaron la fuente.",
        "ranked_model_count": "Cantidad de modelos elegibles donde la fuente entró en el top 10.000.",
        "eligible_support_fraction": "ranked_model_count / eligible_model_count.",
        "best_model_rank": "Mejor ranking individual alcanzado entre los modelos.",
        "model_support": "Cantidad de modelos que aportaron la fuente al consenso.",
        "variant_support": "Cantidad de variantes de datos distintas que apoyan la fuente.",
        "estimator_support": "Cantidad de familias de estimador distintas que apoyan la fuente.",
        "mean_model_rank": "Promedio de los rankings individuales disponibles.",
        "gaia_designation": "Designación textual oficial de Gaia DR3.",
        "ra": "Ascensión recta ICRS de Gaia DR3.",
        "dec": "Declinación ICRS de Gaia DR3.",
        "galactic_l": "Longitud galáctica.",
        "galactic_b": "Latitud galáctica.",
        "G": "Magnitud media Gaia G.",
        "BP_RP": "Color intra-Gaia BP−RP.",
        "J_K": "Color intra-2MASS J−Ks.",
        "W1_W2": "Color intra-WISE W1−W2.",
        "parallax": "Paralaje Gaia DR3.",
        "parallax_error": "Incertidumbre formal de la paralaje Gaia DR3.",
        "parallax_over_error": "Relación paralaje / error de paralaje.",
        "pmra": "Movimiento propio en ascensión recta.",
        "pmdec": "Movimiento propio en declinación.",
        "ruwe": "Renormalised Unit Weight Error de la solución astrométrica Gaia.",
        "astrometric_excess_noise": "Ruido astrométrico excedente informado por Gaia.",
        "phot_bp_rp_excess_factor": "Factor de exceso de flujo BP/RP de Gaia.",
        "phot_bp_n_blended_transits": "Tránsitos BP marcados como blended.",
        "phot_rp_n_blended_transits": "Tránsitos RP marcados como blended.",
        "duplicated_source": "Indicador Gaia de fuente duplicada.",
        "phot_variable_flag": "Indicador de variabilidad fotométrica de Gaia.",
        "tmass_quality": "Cadena ph_qual de 2MASS para J/H/Ks.",
        "tmass_angular_distance": "Separación angular de la contraparte 2MASS.",
        "wise_quality": "Cadena ph_qual de AllWISE; para elegibilidad se usan W1/W2.",
        "wise_angular_distance": "Separación angular de la contraparte AllWISE.",
        "wise_cc_flags": "Banderas AllWISE de contaminación y confusión.",
        "wise_ext_flag": "Bandera AllWISE de fuente extendida.",
        "ag_gspphot": "Extinción en banda G estimada por Gaia GSP-Phot.",
        "halpha_ew": "Ancho equivalente H-alpha de Gaia ESP-ELS; evidencia auxiliar.",
        "halpha_ew_error": "Incertidumbre del ancho equivalente H-alpha.",
        "halpha_ew_flag": "Bandera de calidad del H-alpha ESP-ELS.",
        "espels_class": "Clase espectral propuesta por Gaia ESP-ELS; no altera el ranking.",
        "espels_wc_probability": "Probabilidad ESP-ELS para clase WC; evidencia auxiliar.",
        "espels_wn_probability": "Probabilidad ESP-ELS para clase WN; evidencia auxiliar.",
        "espels_be_probability": "Probabilidad ESP-ELS para estrella Be; evidencia auxiliar.",
        "espels_pne_probability": "Probabilidad ESP-ELS para nebulosa planetaria; evidencia auxiliar.",
        "compatible_variant_count": "Cantidad de variantes exactas del pool compatibles con la fuente.",
        "compatible_variant_mask": "Bitmask estable que codifica las variantes exactas compatibles.",
        "simbad_main_id": "Identificador principal devuelto por el snapshot de SIMBAD.",
        "simbad_main_type": "Tipo principal de objeto en SIMBAD.",
        "simbad_other_types": "Otros tipos de objeto registrados en SIMBAD.",
        "simbad_sp_type": "Tipo espectral disponible en SIMBAD.",
        "match_method": "Método del crossmatch SIMBAD: identificador exacto o posición.",
        "match_count": "Cantidad de coincidencias encontradas por el método registrado.",
        "separation_arcsec": "Separación angular del match posicional aceptado.",
        "review_disposition": "Clasificación científica asignada durante la revisión SIMBAD congelada.",
        "review_comment": "Comentario de revisión científica.",
        "simbad_review_status": "Estado SIMBAD congelado; distingue revisado de todavía no consultado.",
        "simbad_url": "Enlace a la consulta exacta Gaia DR3 en SIMBAD.",
        "gaia_archive_url": "Enlace general al archivo Gaia.",
        "aladin_url": "Enlace de Aladin centrado en la posición de la fuente.",
        "esasky_url": "Enlace de ESASky centrado en la posición de la fuente.",
    }
    model_map = {
        str(row["delivery_key"]): row for row in models.to_dict("records")
    }
    rows: list[dict[str, Any]] = []
    for column in prediction_columns:
        description = descriptions.get(column)
        notes = ""
        for key, model in model_map.items():
            if column == f"{key}_status":
                description = (
                    f"Estado de elegibilidad/ranking para {key}: ranked_top_10000, "
                    "eligible_below_top_10000 o ineligible_variant."
                )
                notes = "Un modelo inelegible nunca evaluó la fuente."
            elif column == f"{key}_eligibility_reason":
                description = (
                    f"Filtro exacto aprobado o componente que impidió evaluar la fuente con {key}."
                )
            elif column == f"{key}_locus_keep":
                description = f"Indica si la fuente pasó el locus de color exacto de {key}."
            elif column == f"{key}_photometry_keep":
                description = (
                    f"Indica si pasó magnitudes y calidad 2MASS/WISE exigidas por {key}."
                )
            elif column == f"{key}_astrometry_keep":
                description = (
                    f"Indica si pasó la restricción de paralaje de {key}."
                )
            elif column == f"{key}_rank":
                description = f"Ranking original de {key}; queda vacío si no entró en su top 10.000."
            elif column == f"{key}_score":
                description = (
                    f"Score bruto de ranking de {key}; no es una probabilidad calibrada."
                )
                notes = "Vacío únicamente cuando la fuente era inelegible para el modelo."
        if description is None:
            raise KeyError(f"Missing Spanish dictionary description for {column}")
        rows.append(
            {
                "archivo": "predictions_all_eligibility_ranked.csv / predictions_top_100.csv",
                "columna": column,
                "tipo": _dictionary_type(column),
                "unidad": _dictionary_unit(column),
                "descripcion": description,
                "observaciones": notes,
            }
        )
    model_descriptions = _model_metric_descriptions()
    for column in model_metric_columns:
        if column not in model_descriptions:
            raise KeyError(f"Missing Spanish model-metric description for {column}")
        rows.append(
            {
                "archivo": "metricas_modelos.csv",
                "columna": column,
                "tipo": _dictionary_type(column),
                "unidad": _dictionary_unit(column),
                "descripcion": model_descriptions[column],
                "observaciones": "",
            }
        )
    return pd.DataFrame(rows)


def _write_zip(archive: Path, root: Path, files: Iterable[Path]) -> None:
    temporary = archive.with_suffix(archive.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    with zipfile.ZipFile(
        temporary,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as bundle:
        for path in sorted({Path(item) for item in files}):
            if path.is_file():
                bundle.write(path, path.relative_to(root).as_posix())
    temporary.replace(archive)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _slug(value: object) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
    return text or "model"
