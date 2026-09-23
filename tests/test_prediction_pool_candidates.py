from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from wr_detector.pipelines.prediction_pool_candidates import (
    POOL_FEATURE_COLUMNS,
    audit_prediction_pool_candidates,
    build_prediction_pool_candidates,
    classify_simbad_match,
    eligibility_aware_consensus,
    _normalize_exact_simbad,
    _write_review_database,
    reciprocal_rank_consensus,
)


def test_reciprocal_rank_consensus_is_deterministic_and_preserves_support():
    ranks = pd.DataFrame(
        [
            {"source_id": 2, "result_id": "a", "dataset_variant": "relaxed", "model": "xgb", "model_rank": 1},
            {"source_id": 1, "result_id": "a", "dataset_variant": "relaxed", "model": "xgb", "model_rank": 2},
            {"source_id": 1, "result_id": "b", "dataset_variant": "strict", "model": "hgb", "model_rank": 1},
            {"source_id": 3, "result_id": "b", "dataset_variant": "strict", "model": "hgb", "model_rank": 2},
        ]
    )

    consensus = reciprocal_rank_consensus(ranks, rrf_k=60)

    assert consensus["source_id"].tolist() == [1, 2, 3]
    assert consensus["consensus_rank"].tolist() == [1, 2, 3]
    assert consensus.loc[0, "model_support"] == 2
    assert consensus.loc[0, "variant_support"] == 2


def test_eligibility_aware_consensus_distinguishes_locus_and_rank_budget():
    ranks = pd.DataFrame(
        [
            {"source_id": 1, "result_id": "a", "dataset_variant": "relaxed", "model": "xgb", "model_rank": 1, "score": 0.9},
            {"source_id": 2, "result_id": "b", "dataset_variant": "strict", "model": "hgb", "model_rank": 1, "score": 0.8},
        ]
    )
    consensus = reciprocal_rank_consensus(ranks, rrf_k=60)
    models = pd.DataFrame(
        [
            {"result_id": "a", "variant_bit": 0, "dataset_variant": "relaxed", "model": "xgb"},
            {"result_id": "b", "variant_bit": 1, "dataset_variant": "strict", "model": "hgb"},
        ]
    )
    sources = pd.DataFrame(
        {
            "source_id": pd.Series([1, 2], dtype="int64"),
            "exact_locus_variant_mask": pd.Series([3, 2], dtype="uint64"),
            "photometry_variant_mask": pd.Series([3, 3], dtype="uint64"),
            "astrometry_variant_mask": pd.Series([3, 3], dtype="uint64"),
            "compatible_variant_mask": pd.Series([3, 2], dtype="uint64"),
        }
    )

    enriched, opportunities = eligibility_aware_consensus(
        consensus,
        ranks,
        models,
        sources,
        rrf_k=60,
    )

    status = opportunities.set_index(["source_id", "result_id"])[
        "model_status"
    ].to_dict()
    assert status[(1, "a")] == "ranked_top_10000"
    assert status[(1, "b")] == "eligible_below_top_10000"
    assert status[(2, "a")] == "ineligible_variant"
    assert status[(2, "b")] == "ranked_top_10000"
    failed = opportunities.set_index(["source_id", "result_id"]).loc[(2, "a")]
    assert not bool(failed["locus_keep"])
    assert bool(failed["photometry_keep"])
    assert bool(failed["astrometry_keep"])
    assert failed["eligibility_reason"] == "failed_locus"
    source_2 = enriched.loc[enriched["source_id"].eq(2)].iloc[0]
    assert int(source_2["eligible_model_count"]) == 1
    assert float(source_2["eligible_support_fraction"]) == 1.0
    assert int(source_2["eligibility_rank"]) == 1


def test_simbad_dispositions_are_conservative():
    assert classify_simbad_match(
        {
            "match_found": True,
            "match_method": "exact_gaia_dr3_id",
            "simbad_main_type": "WR*",
        }
    ) == "known_wr"
    assert classify_simbad_match(
        {
            "match_found": True,
            "match_method": "exact_gaia_dr3_id",
            "simbad_main_type": "Galaxy",
        }
    ) == "catalogued_non_wr"
    assert classify_simbad_match(
        {
            "match_found": True,
            "match_method": "exact_gaia_dr3_id",
            "simbad_main_type": "Be*",
        }
    ) == "emission_or_ambiguous"
    assert classify_simbad_match(
        {
            "match_found": True,
            "match_method": "position_ambiguous",
            "simbad_main_type": "Star",
        }
    ) == "ambiguous_positional_match"
    assert classify_simbad_match({"match_found": False}) == "no_exact_match"


def test_exact_simbad_normalizer_treats_blank_upload_rows_as_unmatched():
    normalized = _normalize_exact_simbad(
        pd.DataFrame(
            {
                "user_specified_id": [
                    "Gaia DR3 123",
                    "Gaia DR3 456",
                ],
                "main_id": ["", "WR 1"],
                "otype": ["", "WR*"],
            }
        )
    )

    unmatched = normalized.loc[normalized["source_id"] == 123].iloc[0]
    matched = normalized.loc[normalized["source_id"] == 456].iloc[0]
    assert not bool(unmatched["match_found"])
    assert unmatched["match_method"] == "none"
    assert int(unmatched["match_count"]) == 0
    assert bool(matched["match_found"])
    assert matched["match_method"] == "exact_gaia_dr3_id"


def test_candidate_review_builds_auditable_outputs(tmp_path: Path):
    setup = _candidate_fixture(tmp_path)

    def fake_simbad(
        candidates: pd.DataFrame, _config: dict[str, object]
    ) -> pd.DataFrame:
        types = {1: "WR*", 2: "Galaxy", 3: "Be*", 4: None}
        return pd.DataFrame(
            {
                "source_id": candidates["source_id"].astype("int64"),
                "match_found": candidates["source_id"].map(
                    lambda value: types[int(value)] is not None
                ),
                "match_method": candidates["source_id"].map(
                    lambda value: (
                        "exact_gaia_dr3_id"
                        if types[int(value)] is not None
                        else "none"
                    )
                ),
                "match_count": candidates["source_id"].map(
                    lambda value: int(types[int(value)] is not None)
                ),
                "separation_arcsec": 0.0,
                "simbad_main_id": candidates["source_id"].map(
                    lambda value: (
                        f"Object {value}"
                        if types[int(value)] is not None
                        else None
                    )
                ),
                "simbad_main_type": candidates["source_id"].map(
                    lambda value: types[int(value)]
                ),
                "simbad_other_types": None,
                "simbad_sp_type": None,
                "queried_at": pd.Timestamp("2026-07-26", tz="UTC"),
            }
        )

    result = build_prediction_pool_candidates(
        setup["candidate_config"],
        refresh_simbad=True,
        simbad_query=fake_simbad,
    )

    assert result["audit"]["ok"]
    audit = audit_prediction_pool_candidates(setup["candidate_config"])
    assert audit["ok"]
    with duckdb.connect(str(setup["review_db"]), read_only=True) as con:
        dispositions = dict(
            con.execute(
                """
                SELECT source_id, review_disposition
                FROM candidate_review
                ORDER BY source_id
                """
            ).fetchall()
        )
        recommended = con.execute(
            """
            SELECT source_id, followup_rank, consensus_rank
            FROM candidate_recommendations
            ORDER BY followup_rank
            """
        ).fetchall()
    assert dispositions[1] == "known_wr"
    assert dispositions[2] == "catalogued_non_wr"
    assert dispositions[3] == "emission_or_ambiguous"
    assert dispositions[4] == "no_exact_match"
    assert {row[0] for row in recommended}.issubset({3, 4})
    assert recommended == sorted(recommended, key=lambda row: row[1])
    assert [row[2] for row in recommended] == sorted(row[2] for row in recommended)


def test_candidate_review_database_preserves_previous_runs(tmp_path: Path):
    db_path = tmp_path / "current.duckdb"
    history_dir = tmp_path / "history"
    first = _write_review_database(
        db_path,
        {"candidate_review_runs": pd.DataFrame({"review_run_id": ["first"]})},
        history_dir=history_dir,
    )
    second = _write_review_database(
        db_path,
        {"candidate_review_runs": pd.DataFrame({"review_run_id": ["second"]})},
        history_dir=history_dir,
    )

    assert first != second
    assert sorted(history_dir.glob("*.duckdb")) == sorted([first, second])
    with duckdb.connect(str(first), read_only=True) as con:
        assert con.execute("SELECT review_run_id FROM candidate_review_runs").fetchone() == ("first",)
    with duckdb.connect(str(db_path), read_only=True) as con:
        assert con.execute("SELECT review_run_id FROM candidate_review_runs").fetchone() == ("second",)


def _candidate_fixture(tmp_path: Path) -> dict[str, Path]:
    pool_db = tmp_path / "pool.duckdb"
    pool = pd.DataFrame(index=range(4))
    for column in POOL_FEATURE_COLUMNS:
        pool[column] = np.nan
    pool["source_id"] = pd.Series([1, 2, 3, 4], dtype="int64")
    pool["gaia_designation"] = [f"Gaia DR3 {value}" for value in [1, 2, 3, 4]]
    pool["source_hash_v1"] = ["h1", "h2", "h3", "h4"]
    pool["ra"] = [1.0, 2.0, 3.0, 4.0]
    pool["dec"] = [-1.0, -2.0, -3.0, -4.0]
    pool["exact_locus_variant_mask"] = pd.Series([3, 3, 3, 3], dtype="uint64")
    pool["photometry_variant_mask"] = pd.Series([3, 3, 3, 3], dtype="uint64")
    pool["astrometry_variant_mask"] = pd.Series([3, 3, 3, 3], dtype="uint64")
    pool["compatible_variant_mask"] = pd.Series([3, 3, 3, 3], dtype="uint64")
    pool["compatible_variant_count"] = pd.Series([2, 2, 2, 2], dtype="int16")
    with duckdb.connect(str(pool_db)) as con:
        con.register("pool_frame", pool)
        con.execute(
            "CREATE TABLE prediction_pool_sources AS SELECT * FROM pool_frame"
        )

    scoring_db = tmp_path / "scoring.duckdb"
    scores = pd.DataFrame(
        [
            {"source_id": 1, "score": 0.9, "predicted": True, "threshold": 0.5, "result_id": "a", "dataset_variant": "relaxed", "scoring_run_id": "score"},
            {"source_id": 3, "score": 0.8, "predicted": True, "threshold": 0.5, "result_id": "a", "dataset_variant": "relaxed", "scoring_run_id": "score"},
            {"source_id": 4, "score": 0.7, "predicted": True, "threshold": 0.5, "result_id": "a", "dataset_variant": "relaxed", "scoring_run_id": "score"},
            {"source_id": 2, "score": 0.95, "predicted": True, "threshold": 0.5, "result_id": "b", "dataset_variant": "strict", "scoring_run_id": "score"},
            {"source_id": 3, "score": 0.85, "predicted": True, "threshold": 0.5, "result_id": "b", "dataset_variant": "strict", "scoring_run_id": "score"},
            {"source_id": 4, "score": 0.75, "predicted": True, "threshold": 0.5, "result_id": "b", "dataset_variant": "strict", "scoring_run_id": "score"},
        ]
    )
    with duckdb.connect(str(scoring_db)) as con:
        con.execute(
            """
            CREATE TABLE prediction_scoring_runs (
                scoring_run_id VARCHAR, model_run_id VARCHAR,
                pool_build_id VARCHAR, status VARCHAR
            )
            """
        )
        con.execute(
            "INSERT INTO prediction_scoring_runs VALUES "
            "('score', 'models', 'pool', 'completed')"
        )
        con.execute(
            """
            CREATE TABLE prediction_scoring_models (
                scoring_run_id VARCHAR, result_id VARCHAR,
                dataset_variant VARCHAR, variant_bit INTEGER
            )
            """
        )
        con.execute(
            "INSERT INTO prediction_scoring_models VALUES "
            "('score', 'a', 'relaxed', 0), ('score', 'b', 'strict', 1)"
        )
        con.execute(
            """
            CREATE TABLE prediction_scoring_tiles (
                scoring_run_id VARCHAR, result_id VARCHAR,
                tile_id VARCHAR, status VARCHAR
            )
            """
        )
        con.execute(
            "INSERT INTO prediction_scoring_tiles VALUES "
            "('score', 'a', 'tile', 'completed'), "
            "('score', 'b', 'tile', 'completed')"
        )
        con.register("score_frame", scores)
        con.execute(
            "CREATE TABLE prediction_pool_scores AS SELECT * FROM score_frame"
        )

    pool_config = tmp_path / "pool.yaml"
    pool_config.write_text(f"output_db: {pool_db.as_posix()}\n", encoding="utf-8")
    scoring_config = tmp_path / "scoring.yaml"
    scoring_config.write_text(
        "\n".join(
            [
                f"pool_config: {pool_config.as_posix()}",
                "models_config: configs/models.yaml",
                f"output_db: {scoring_db.as_posix()}",
                f"output_dir: {(tmp_path / 'scores').as_posix()}",
                f"manifest_dir: {(tmp_path / 'score_manifests').as_posix()}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    review_db = tmp_path / "candidates.duckdb"
    candidate_config = tmp_path / "candidates.yaml"
    candidate_config.write_text(
        "\n".join(
            [
                f"scoring_config: {scoring_config.as_posix()}",
                "review:",
                "  review_run_id: review",
                "  scoring_run_id: score",
                "  model_run_id: models",
                "  models:",
                "    - result_id: a",
                "      model: xgboost",
                "      role: broad",
                "    - result_id: b",
                "      model: hist_gradient_boosting",
                "      role: diverse",
                "ranking:",
                "  rrf_k: 60",
                "  per_model_limit: 3",
                "  review_limit: 4",
                "  deep_review_limit: 2",
                "  headline_limit: 1",
                "simbad:",
                f"  snapshot_path: {(tmp_path / 'simbad.parquet').as_posix()}",
                "outputs:",
                f"  database: {review_db.as_posix()}",
                f"  directory: {(tmp_path / 'reports').as_posix()}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "candidate_config": candidate_config,
        "review_db": review_db,
    }
