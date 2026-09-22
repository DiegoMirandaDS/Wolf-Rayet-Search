from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from wr_detector.modeling.prediction_pool_review import (
    load_candidate_detail,
    load_candidate_jaccard,
    load_candidate_model_evidence,
    load_candidate_models,
    load_candidate_plot_frame,
    load_candidate_rankings,
    load_candidate_review_runs,
    load_candidate_summary,
    load_disposition_counts,
    load_pool_status,
    load_scoring_status,
    prediction_pool_review_revision,
    review_availability,
)


def test_prediction_pool_review_queries_are_traceable_and_deterministic(
    tmp_path: Path,
):
    config = _review_fixture(tmp_path)

    availability = review_availability(config)
    runs = load_candidate_review_runs(config)
    pool, tiles = load_pool_status(config)
    scoring, scored_models = load_scoring_status(config)
    models = load_candidate_models(config)
    summary = load_candidate_summary(config)
    dispositions = load_disposition_counts(config)
    consensus = load_candidate_rankings(
        config,
        min_model_support=2,
        followup_only=True,
    )
    specialist = load_candidate_rankings(
        config,
        view="specialist",
        result_id="a",
        limit=2,
    )
    detail = load_candidate_detail(config, source_id=2)
    evidence = load_candidate_model_evidence(config, source_id=2)
    jaccard = load_candidate_jaccard(config, top_k=2)
    plot = load_candidate_plot_frame(config)

    assert availability["ready"]
    assert len(prediction_pool_review_revision(config)) == 64
    assert runs["review_run_id"].tolist() == ["review"]
    assert pool.loc[0, "completed_tiles"] == 1
    assert tiles.loc[0, "retention_fraction"] == 0.8
    assert scoring.loc[0, "status"] == "completed"
    assert scored_models["scored_fraction"].tolist() == [0.9, 0.9]
    assert set(models["result_id"]) == {"a", "b"}
    assert summary["review_rows"] == 3
    assert set(dispositions["review_disposition"]) == {
        "no_exact_match",
        "catalogued_non_wr",
        "emission_or_ambiguous",
    }
    assert consensus["source_id"].tolist() == [2]
    assert specialist["source_id"].tolist() == [1, 2]
    assert detail.loc[0, "simbad_main_id"] == "Emission source"
    assert evidence["model_rank"].tolist() == [1, 2]
    diagonal = jaccard[
        jaccard["left_result_id"].eq(jaccard["right_result_id"])
    ]
    assert diagonal["jaccard"].eq(1.0).all()
    assert {"mollweide_x", "distance_plotted"}.issubset(plot.columns)


def test_prediction_pool_review_degrades_when_databases_are_missing(
    tmp_path: Path,
):
    candidate_config = tmp_path / "candidate.yaml"
    scoring_config = tmp_path / "scoring.yaml"
    pool_config = tmp_path / "pool.yaml"
    pool_config.write_text(
        f"output_db: {(tmp_path / 'missing_pool.duckdb').as_posix()}\n",
        encoding="utf-8",
    )
    scoring_config.write_text(
        "\n".join(
            [
                f"pool_config: {pool_config.as_posix()}",
                f"output_db: {(tmp_path / 'missing_scores.duckdb').as_posix()}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    candidate_config.write_text(
        "\n".join(
            [
                f"scoring_config: {scoring_config.as_posix()}",
                "outputs:",
                f"  database: {(tmp_path / 'missing_candidates.duckdb').as_posix()}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert not review_availability(candidate_config)["ready"]
    assert load_candidate_review_runs(candidate_config).empty
    assert load_candidate_rankings(candidate_config).empty
    pool, tiles = load_pool_status(candidate_config)
    assert pool.empty
    assert tiles.empty


def _review_fixture(tmp_path: Path) -> Path:
    pool_db = tmp_path / "pool.duckdb"
    with duckdb.connect(str(pool_db)) as con:
        con.execute(
            """
            CREATE TABLE exact_union_builds (
                pool_build_id VARCHAR, status VARCHAR, envelope_sha256 VARCHAR,
                bitmask_schema_version VARCHAR, bitmask_schema_sha256 VARCHAR,
                updated_at TIMESTAMP
            )
            """
        )
        con.execute(
            "INSERT INTO exact_union_builds VALUES "
            "('pool', 'completed', 'env', 'v1', 'bits', TIMESTAMP '2026-07-27')"
        )
        con.execute(
            """
            CREATE TABLE exact_union_tiles (
                pool_build_id VARCHAR, tile_id VARCHAR, ra_min DOUBLE,
                ra_max DOUBLE, dec_min DOUBLE, dec_max DOUBLE, status VARCHAR,
                acquired_pre_locus BIGINT, accepted_union BIGINT,
                known_excluded BIGINT, written BIGINT, parquet_bytes BIGINT,
                updated_at TIMESTAMP
            )
            """
        )
        con.execute(
            "INSERT INTO exact_union_tiles VALUES "
            "('pool', 'tile', 0, 10, -5, 5, 'completed', "
            "100, 82, 2, 80, 1000, TIMESTAMP '2026-07-27')"
        )

    scoring_db = tmp_path / "scoring.duckdb"
    with duckdb.connect(str(scoring_db)) as con:
        con.execute(
            """
            CREATE TABLE prediction_scoring_runs (
                scoring_run_id VARCHAR, model_run_id VARCHAR,
                pool_build_id VARCHAR, status VARCHAR,
                model_selection_sha256 VARCHAR, config_sha256 VARCHAR,
                created_at TIMESTAMP, updated_at TIMESTAMP
            )
            """
        )
        con.execute(
            "INSERT INTO prediction_scoring_runs VALUES "
            "('score', 'models', 'pool', 'completed', 'models-sha', "
            "'config-sha', TIMESTAMP '2026-07-27', TIMESTAMP '2026-07-27')"
        )
        con.execute(
            """
            CREATE TABLE prediction_scoring_tiles (
                scoring_run_id VARCHAR, result_id VARCHAR,
                dataset_variant VARCHAR, status VARCHAR,
                rows_variant_compatible BIGINT, rows_missing_features BIGINT,
                rows_scored BIGINT, rows_predicted_positive BIGINT,
                output_bytes BIGINT
            )
            """
        )
        con.execute(
            "INSERT INTO prediction_scoring_tiles VALUES "
            "('score','a','relaxed','completed',100,10,90,3,1000),"
            "('score','b','strict','completed',100,10,90,2,900)"
        )

    candidate_db = tmp_path / "candidate.duckdb"
    models = pd.DataFrame(
        [
            {
                "result_id": "a",
                "dataset_variant": "relaxed",
                "model": "xgboost",
                "role": "broad",
                "run_id": "models",
                "feature_set": "colors",
                "sampler": "none",
                "holdout_average_precision": 0.6,
            },
            {
                "result_id": "b",
                "dataset_variant": "strict",
                "model": "hist_gradient_boosting",
                "role": "diverse",
                "run_id": "models",
                "feature_set": "colors",
                "sampler": "none",
                "holdout_average_precision": 0.5,
            },
        ]
    )
    ranks = pd.DataFrame(
        [
            {"model_rank": 1, "source_id": 1, "score": 0.9, "predicted": True, "threshold": 0.5, "result_id": "a", "dataset_variant": "relaxed", "model": "xgboost", "role": "broad"},
            {"model_rank": 2, "source_id": 2, "score": 0.8, "predicted": True, "threshold": 0.5, "result_id": "a", "dataset_variant": "relaxed", "model": "xgboost", "role": "broad"},
            {"model_rank": 1, "source_id": 2, "score": 0.7, "predicted": True, "threshold": 0.4, "result_id": "b", "dataset_variant": "strict", "model": "hist_gradient_boosting", "role": "diverse"},
            {"model_rank": 2, "source_id": 3, "score": 0.6, "predicted": True, "threshold": 0.4, "result_id": "b", "dataset_variant": "strict", "model": "hist_gradient_boosting", "role": "diverse"},
        ]
    )
    review = pd.DataFrame(
        [
            _candidate_row(1, 1, 1, "catalogued_non_wr", "Galaxy"),
            _candidate_row(2, 2, 2, "emission_or_ambiguous", "Emission source"),
            _candidate_row(3, 3, 1, "no_exact_match", None),
        ]
    )
    consensus = review[
        [
            "consensus_rank",
            "source_id",
            "rrf_score",
            "model_support",
            "variant_support",
            "estimator_support",
            "best_model_rank",
            "mean_model_rank",
        ]
    ]
    recommendations = review.iloc[[1]].copy()
    recommendations.insert(0, "followup_rank", [1])
    run = pd.DataFrame(
        [
            {
                "review_run_id": "review",
                "scoring_run_id": "score",
                "model_run_id": "models",
                "pool_build_id": "pool",
                "created_at": pd.Timestamp("2026-07-27"),
            }
        ]
    )
    with duckdb.connect(str(candidate_db)) as con:
        for name, frame in {
            "candidate_review_runs": run,
            "candidate_models": models,
            "candidate_model_ranks": ranks,
            "candidate_consensus": consensus,
            "candidate_review": review,
            "candidate_recommendations": recommendations,
        }.items():
            con.register("incoming", frame)
            con.execute(f"CREATE TABLE {name} AS SELECT * FROM incoming")
            con.unregister("incoming")

    pool_config = tmp_path / "pool.yaml"
    pool_config.write_text(
        f"output_db: {pool_db.as_posix()}\n",
        encoding="utf-8",
    )
    scoring_config = tmp_path / "scoring.yaml"
    scoring_config.write_text(
        "\n".join(
            [
                f"pool_config: {pool_config.as_posix()}",
                f"output_db: {scoring_db.as_posix()}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    candidate_config = tmp_path / "candidate.yaml"
    candidate_config.write_text(
        "\n".join(
            [
                f"scoring_config: {scoring_config.as_posix()}",
                "outputs:",
                f"  database: {candidate_db.as_posix()}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return candidate_config


def _candidate_row(
    source_id: int,
    rank: int,
    support: int,
    disposition: str,
    simbad_id: str | None,
) -> dict[str, object]:
    return {
        "consensus_rank": rank,
        "source_id": source_id,
        "rrf_score": 0.05 / rank,
        "model_support": support,
        "variant_support": support,
        "estimator_support": support,
        "best_model_rank": rank,
        "mean_model_rank": float(rank),
        "gaia_designation": f"Gaia DR3 {source_id}",
        "ra": 10.0 * source_id,
        "dec": -2.0 * source_id,
        "G": 12.0 + source_id,
        "BP_RP": 1.0 + source_id / 10,
        "J_H": 0.5,
        "H_K": 0.3,
        "J_K": 0.8,
        "W1_W2": 0.4,
        "parallax": 0.5,
        "parallax_over_error": 5.0,
        "ruwe": 1.0,
        "halpha_ew": -1.0 if source_id != 3 else None,
        "espels_class": "beStar" if source_id == 2 else None,
        "match_found": simbad_id is not None,
        "match_method": "exact_gaia_dr3_id" if simbad_id else "none",
        "simbad_main_id": simbad_id,
        "simbad_main_type": "Em*" if source_id == 2 else None,
        "review_disposition": disposition,
        "review_comment": "Spectroscopy required.",
    }
