from __future__ import annotations

import json
from pathlib import Path

import duckdb
from joblib import dump
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from wr_detector.pipelines.exact_variant_union import (
    build_variant_mask_schema,
)
from wr_detector.pipelines.prediction_pool_scoring import (
    _validate_path_component,
    audit_prediction_pool_scoring,
    file_sha256,
    list_scoreable_models,
    score_prediction_pool,
)


def test_real_sky_tile_ids_are_safe_path_components():
    _validate_path_component(
        "ra000p00_010p00__dec+00p00_+10p00",
        label="tile_id",
    )


def test_scoring_filters_variant_bit_tracks_missing_and_resumes(tmp_path):
    setup = _write_scoring_fixture(tmp_path)

    planned = score_prediction_pool(
        setup["scoring_config"],
        scoring_run_id="score_test",
        model_run_id="model_run",
        result_ids=["result_strict"],
        dry_run=True,
    )
    assert planned["work_units"] == 1
    assert planned["models"][0]["bit"] == 0

    result = score_prediction_pool(
        setup["scoring_config"],
        scoring_run_id="score_test",
        model_run_id="model_run",
        result_ids=["result_strict"],
    )
    assert result["completed_this_run"] == 1
    output_path = (
        tmp_path
        / "scores"
        / "score_test"
        / "result_strict"
        / "tile.parquet"
    )
    scored = pd.read_parquet(output_path)
    assert scored["source_id"].tolist() == [1]
    assert scored["model_run_id"].unique().tolist() == ["model_run"]
    assert scored["scoring_run_id"].unique().tolist() == ["score_test"]

    with duckdb.connect(
        str(tmp_path / "scoring.duckdb"), read_only=True
    ) as con:
        counts = con.execute(
            """
            SELECT rows_variant_compatible, rows_missing_features,
                   rows_scored
            FROM prediction_scoring_tiles
            """
        ).fetchone()
    assert counts == (2, 1, 1)

    resumed = score_prediction_pool(
        setup["scoring_config"],
        scoring_run_id="score_test",
        model_run_id="model_run",
        result_ids=["result_strict"],
    )
    assert resumed["completed_this_run"] == 0
    assert resumed["skipped_valid"] == 1

    audit = audit_prediction_pool_scoring(
        setup["scoring_config"],
        scoring_run_id="score_test",
    )
    assert audit["ok"]
    assert audit["errors"] == []


def test_scoring_rejects_changed_model_artifact(tmp_path):
    setup = _write_scoring_fixture(tmp_path)
    setup["model_path"].write_bytes(b"changed")

    with pytest.raises(ValueError, match="Model hash mismatch"):
        score_prediction_pool(
            setup["scoring_config"],
            scoring_run_id="score_test",
            model_run_id="model_run",
            result_ids=["result_strict"],
            dry_run=True,
        )


def test_scoring_requires_explicit_model_selection(tmp_path):
    setup = _write_scoring_fixture(tmp_path)

    available = list_scoreable_models(
        setup["scoring_config"],
        model_run_id="model_run",
    )
    assert available["result_id"].tolist() == ["result_strict"]
    with pytest.raises(ValueError, match="explicit model result_id"):
        score_prediction_pool(
            setup["scoring_config"],
            scoring_run_id="score_test",
            model_run_id="model_run",
            result_ids=[],
            dry_run=True,
        )


def test_scoring_rejects_pool_without_completed_build_status(tmp_path):
    setup = _write_scoring_fixture(tmp_path)
    with duckdb.connect(str(setup["pool_db"])) as con:
        con.execute(
            "UPDATE exact_union_builds SET status='partial'"
        )

    with pytest.raises(ValueError, match="is not completed"):
        score_prediction_pool(
            setup["scoring_config"],
            scoring_run_id="score_test",
            model_run_id="model_run",
            result_ids=["result_strict"],
            dry_run=True,
        )


def _write_scoring_fixture(tmp_path: Path) -> dict[str, Path]:
    schema = build_variant_mask_schema(
        ["strict_photometry", "relaxed_photometry"]
    )
    strict_bit = 1 << schema.bit_for("strict_photometry")
    relaxed_bit = 1 << schema.bit_for("relaxed_photometry")
    acquisition = pd.DataFrame(
        {
            "source_id": [1, 2, 3, 4],
            "source_hash_v1": ["h1", "h2", "h3", "h4"],
            "f1": [0.0, 0.2, 0.4, 0.6],
            "f2": [0.1, 0.3, None, 0.7],
            "compatible_variant_mask": [
                strict_bit,
                relaxed_bit,
                strict_bit,
                strict_bit,
            ],
            "passes_any_exact_variant": [True, True, True, True],
            "known_source_exclusion": [None, None, None, "known_wr"],
        }
    )
    acquisition_path = tmp_path / "acquisition" / "tile.parquet"
    acquisition_path.parent.mkdir()
    acquisition.to_parquet(acquisition_path, index=False)
    acquisition_sha256 = file_sha256(acquisition_path)
    tile_manifest = tmp_path / "pool_manifests" / "tile.json"
    tile_manifest.parent.mkdir()
    tile_manifest.write_text('{"tile_id": "tile"}', encoding="utf-8")

    locus_sha256 = "b" * 64
    pool_db = tmp_path / "pool.duckdb"
    with duckdb.connect(str(pool_db)) as con:
        con.execute(
            """
            CREATE TABLE exact_union_builds (
                pool_build_id VARCHAR,
                status VARCHAR,
                envelope_sha256 VARCHAR,
                bitmask_schema_version VARCHAR,
                bitmask_schema_sha256 VARCHAR
            )
            """
        )
        con.execute(
            "INSERT INTO exact_union_builds VALUES (?, ?, ?, ?, ?)",
            ["pool", "completed", "envelope", schema.version, schema.sha256],
        )
        con.execute(
            """
            CREATE TABLE exact_union_contract (
                bitmask_schema JSON,
                loci JSON
            )
            """
        )
        loci = {
            "strict_photometry": {
                "locus_run_id": "locus_strict",
                "source_sha256": locus_sha256,
            },
            "relaxed_photometry": {
                "locus_run_id": "locus_relaxed",
                "source_sha256": "c" * 64,
            },
        }
        con.execute(
            "INSERT INTO exact_union_contract VALUES (?::JSON, ?::JSON)",
            [json.dumps(schema.as_manifest()), json.dumps(loci)],
        )
        con.execute(
            """
            CREATE TABLE exact_union_tiles (
                pool_build_id VARCHAR,
                tile_id VARCHAR,
                status VARCHAR,
                parquet_path VARCHAR,
                acquisition_sha256 VARCHAR,
                manifest_path VARCHAR,
                manifest_sha256 VARCHAR,
                written BIGINT
            )
            """
        )
        con.execute(
            "INSERT INTO exact_union_tiles VALUES "
            "('pool', 'tile', 'completed', ?, ?, ?, ?, 4)",
            [
                str(acquisition_path),
                acquisition_sha256,
                str(tile_manifest),
                file_sha256(tile_manifest),
            ],
        )

    paths_config = tmp_path / "paths.yaml"
    filters_config = tmp_path / "filters.yaml"
    paths_config.write_text("{}\n", encoding="utf-8")
    filters_config.write_text("{}\n", encoding="utf-8")
    pool_config = tmp_path / "pool.yaml"
    pool_config.write_text(
        "\n".join(
            [
                f"paths_config: {paths_config.as_posix()}",
                f"filters_config: {filters_config.as_posix()}",
                "build:",
                "  pool_build_id: pool",
                f"output_db: {pool_db.as_posix()}",
                "tiling:",
                "  explicit_tiles:",
                "    - tile_id: tile",
                "      ra_min: 0",
                "      ra_max: 1",
                "      dec_min: 0",
                "      dec_max: 1",
            ]
        ),
        encoding="utf-8",
    )

    model = LogisticRegression().fit(
        pd.DataFrame(
            {"f1": [0.0, 0.2, 0.8, 1.0], "f2": [0.1, 0.3, 0.9, 1.1]}
        ),
        [0, 0, 1, 1],
    )
    model_path = tmp_path / "model.joblib"
    dump(model, model_path)
    history_db = tmp_path / "history.duckdb"
    model_results = pd.DataFrame(
        [
            {
                "run_id": "model_run",
                "result_id": "result_strict",
                "dataset_variant": "strict_photometry",
                "feature_set": "test",
                "model": "logistic_regression",
                "sampler": "none",
                "holdout_average_precision": 0.8,
                "holdout_wr_at_50": 2,
                "holdout_wr_at_100": 3,
                "selection_status": "eligible",
                "locus_run_id": "locus_strict",
                "reference_dataset_sha256": locus_sha256,
                "model_path": str(model_path),
                "model_sha256": file_sha256(model_path),
                "feature_columns_json": json.dumps(["f1", "f2"]),
                "selected_threshold": 0.5,
            }
        ]
    )
    with duckdb.connect(str(history_db)) as con:
        con.register("results", model_results)
        con.execute("CREATE TABLE model_results AS SELECT * FROM results")
    models_config = tmp_path / "models.yaml"
    models_config.write_text(
        "\n".join(
            [
                "outputs:",
                f"  training_history_db: {history_db.as_posix()}",
            ]
        ),
        encoding="utf-8",
    )
    scoring_config = tmp_path / "scoring.yaml"
    scoring_config.write_text(
        "\n".join(
            [
                f"pool_config: {pool_config.as_posix()}",
                f"models_config: {models_config.as_posix()}",
                f"output_db: {(tmp_path / 'scoring.duckdb').as_posix()}",
                f"output_dir: {(tmp_path / 'scores').as_posix()}",
                f"manifest_dir: {(tmp_path / 'score_manifests').as_posix()}",
                "batch_rows: 2",
                "compression: zstd",
                "safety:",
                "  require_complete_pool: true",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "scoring_config": scoring_config,
        "model_path": model_path,
        "pool_db": pool_db,
    }
