from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from wr_detector.db import write_reference_database, write_simbad_negative_database
from wr_detector.features import add_color_features, annotate_color_locus_planes, evaluate_color_locus_planes
from wr_detector.pipelines.prediction_pool import (
    SkyTile,
    audit_prediction_pool,
    build_prediction_pool_adql,
    build_quality_predicate,
    derive_color_envelope,
    ingest_prediction_tile,
    initialize_prediction_pool_database,
    local_storage_bytes,
    mark_tile_completed,
    refresh_known_sources,
    refresh_prediction_pool_metadata_views,
)


def _write_prediction_config(tmp_path: Path, variants: list[str] | None = None) -> dict:
    processed_reference = tmp_path / "processed" / "reference"
    processed_simbad = tmp_path / "processed" / "simbad"
    database_dir = tmp_path / "databases"
    raw_dir = tmp_path / "raw"
    pool_dir = tmp_path / "processed" / "prediction_pool" / "tiles"
    for path in [processed_reference, processed_simbad, database_dir, raw_dir, pool_dir]:
        path.mkdir(parents=True, exist_ok=True)

    paths_yaml = tmp_path / "paths.yaml"
    paths_yaml.write_text(
        "\n".join(
            [
                f"database_dir: {database_dir.as_posix()}",
                f"processed_reference_dir: {processed_reference.as_posix()}",
                f"processed_simbad_negative_dir: {processed_simbad.as_posix()}",
                f"wr_reference_db: {(database_dir / 'wr_reference.duckdb').as_posix()}",
                f"simbad_negative_db: {(database_dir / 'simbad_negative.duckdb').as_posix()}",
            ]
        ),
        encoding="utf-8",
    )
    filters_yaml = tmp_path / "filters.yaml"
    filters_yaml.write_text(
        "\n".join(
            [
                f"paths_config: {paths_yaml.as_posix()}",
                "color_locus:",
                "  dataset_variants:",
                "    - relaxed_photometry",
                '  reference_output_template: "wr_reference_{variant}_color_locus.parquet"',
            ]
        ),
        encoding="utf-8",
    )
    return {
        "paths_config": str(paths_yaml),
        "filters_config": str(filters_yaml),
        "output_db": str(database_dir / "prediction_pool.duckdb"),
        "staging_dir": str(raw_dir),
        "output_dir": str(pool_dir),
        "parquet": {"compression": "zstd"},
        "paths": {
            "processed_reference_dir": str(processed_reference),
            "processed_simbad_negative_dir": str(processed_simbad),
            "wr_reference_db": str(database_dir / "wr_reference.duckdb"),
            "simbad_negative_db": str(database_dir / "simbad_negative.duckdb"),
        },
        "filters": {
            "color_locus": {
                "dataset_variants": ["relaxed_photometry"],
                "reference_output_template": "wr_reference_{variant}_color_locus.parquet",
            }
        },
        "photometry": {
            "twomass_allowed_qualities": ["A", "B"],
            "wise_allowed_qualities": ["A", "B"],
        },
        "astrometry": {"min_parallax": None, "min_parallax_over_error": None},
        "tiling": {"ra_step_deg": 10.0, "dec_step_deg": 10.0},
        "safety": {"max_local_bytes": 10_000_000_000, "max_workers": 1},
        "color_envelope": {
            "variants": variants or ["relaxed_photometry"],
            "use_kept_rows": True,
        },
    }


def _write_color_locus_parquet(config: dict, variant: str = "relaxed_photometry") -> None:
    df = add_color_features(
        pd.DataFrame(
            {
                "source_id": range(1, 8),
                "G": [10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0],
                "BP": [11.0, 11.4, 11.9, 12.3, 12.8, 13.2, 13.7],
                "RP": [9.0, 9.2, 9.5, 9.7, 10.0, 10.2, 10.5],
                "J": [8.0, 8.2, 8.4, 8.6, 8.8, 9.0, 9.2],
                "H": [7.5, 7.6, 7.8, 7.9, 8.1, 8.2, 8.4],
                "Ks": [7.1, 7.2, 7.3, 7.5, 7.6, 7.8, 7.9],
                "W1": [7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6],
                "W2": [6.9, 7.0, 7.1, 7.2, 7.3, 7.4, 7.5],
            }
        )
    )
    fits = evaluate_color_locus_planes(
        df,
        [{"x": "G_BP", "y": "G_RP"}, {"x": "J_H", "y": "J_K"}],
        min_positive_fraction=1.0,
        estimator="linear",
        transform="signed_log1p",
        residual_quantile=1.0,
    )
    annotated = annotate_color_locus_planes(
        df,
        fits,
        residual_sigma_threshold=2.0,
        threshold_method="empirical_quantile",
        aggregate_min_outlier_planes=2,
    )
    out = Path(config["paths"]["processed_reference_dir"]) / f"wr_reference_{variant}_color_locus.parquet"
    annotated.to_parquet(out, index=False)


def test_adql_builder_uses_required_gaia_xmatches_and_intra_mission_colors(tmp_path):
    config = _write_prediction_config(tmp_path)
    _write_color_locus_parquet(config)
    envelope = derive_color_envelope(config)
    tile = SkyTile("t", 0.0, 1.0, -1.0, 0.0)

    adql = build_prediction_pool_adql(tile, envelope, config, row_limit=100)

    assert "TOP 100" in adql
    assert "gaiadr3.tmass_psc_xsc_best_neighbour" in adql
    assert "gaiadr3.tmass_psc_xsc_join" in adql
    assert "gaiadr1.tmass_original_valid" in adql
    assert "gaiadr3.allwise_best_neighbour" in adql
    assert "gaiadr1.allwise_original_valid" in adql
    assert "gaia.phot_g_mean_mag - gaia.phot_bp_mean_mag" in adql
    assert "gaia.phot_g_mean_flux_error AS G_flux_error" in adql
    assert "tmass.j_msigcom AS J_error" in adql
    assert "wise.w4mpro_error AS W4_error" in adql
    assert "gaia.ruwe" not in adql
    assert "tmass.j_m - tmass.h_m" in adql
    assert "wise.w1mpro - wise.w2mpro" in adql
    assert "gaia.phot_g_mean_mag - tmass" not in adql
    assert "tmass.ks_m - wise" not in adql
    assert "gaia.parallax >" not in adql
    assert "gaia.parallax_over_error >=" not in adql
    assert "LOG(1 + ABS" not in adql


def test_quality_policy_accepts_ab_for_required_bands(tmp_path):
    config = _write_prediction_config(tmp_path)

    predicate = build_quality_predicate(config)

    assert "tmass.ph_qual LIKE 'AAA%'" in predicate
    assert "tmass.ph_qual LIKE 'BBB%'" in predicate
    assert "wise.ph_qual LIKE 'AA%'" in predicate
    assert "wise.ph_qual LIKE 'BB%'" in predicate
    assert "SUBSTRING" not in predicate


def test_color_envelope_derives_from_relaxed_color_locus_outputs(tmp_path):
    config = _write_prediction_config(tmp_path)
    _write_color_locus_parquet(config)

    envelope = derive_color_envelope(config)

    assert envelope["variants"] == ["relaxed_photometry"]
    assert {"G_BP", "G_RP", "BP_RP", "J_H", "J_K", "H_K", "W1_W2"}.issubset(envelope["color_bounds"])
    assert {tuple((fit["x"], fit["y"])) for fit in envelope["fits"]} == {("G_BP", "G_RP"), ("J_H", "J_K")}


def test_known_sources_are_excluded_during_tile_ingest(tmp_path):
    config = _write_prediction_config(tmp_path)
    _write_color_locus_parquet(config)
    envelope = derive_color_envelope(config)
    db_path = Path(config["output_db"])

    write_reference_database(
        config["paths"]["wr_reference_db"],
        snapshot={"snapshot_id": "test"},
        wr_catalog_raw=pd.DataFrame(),
        wr_reference=pd.DataFrame({"source_id": [10]}),
        gaia_sources=pd.DataFrame(),
        twomass_matches=pd.DataFrame(),
        wise_matches=pd.DataFrame(),
        crossmatch_log=pd.DataFrame(),
    )
    write_simbad_negative_database(
        config["paths"]["simbad_negative_db"],
        simbad_negative_raw=pd.DataFrame(),
        simbad_negative_sources=pd.DataFrame({"source_id": [20]}),
        gaia_sources=pd.DataFrame(),
        twomass_matches=pd.DataFrame(),
        wise_matches=pd.DataFrame(),
        crossmatch_log=pd.DataFrame(),
    )
    con = duckdb.connect(str(db_path))
    initialize_prediction_pool_database(con)
    refresh_known_sources(con, config)
    con.close()
    csv_path = tmp_path / "tile.csv"
    pd.DataFrame(
        {
            "source_id": [10, 20, 30],
            "gaia_designation": ["a", "b", "c"],
            "ra": [0.1, 0.2, 0.3],
            "dec": [-0.1, -0.2, -0.3],
            "G": [10.0, 10.5, 11.0],
            "BP": [11.0, 11.4, 11.9],
            "RP": [9.0, 9.2, 9.5],
            "G_flux": [100.0, 101.0, 102.0],
            "G_flux_error": [1.0, 1.0, 1.0],
            "BP_flux": [90.0, 91.0, 92.0],
            "BP_flux_error": [1.0, 1.0, 1.0],
            "RP_flux": [110.0, 111.0, 112.0],
            "RP_flux_error": [1.0, 1.0, 1.0],
            "J": [8.0, 8.2, 8.4],
            "H": [7.5, 7.6, 7.8],
            "Ks": [7.1, 7.2, 7.3],
            "J_error": [0.02, 0.02, 0.02],
            "H_error": [0.02, 0.02, 0.02],
            "Ks_error": [0.02, 0.02, 0.02],
            "W1": [7.0, 7.1, 7.2],
            "W2": [6.9, 7.0, 7.1],
            "W3": [None, None, None],
            "W4": [None, None, None],
            "W1_error": [0.03, 0.03, 0.03],
            "W2_error": [0.03, 0.03, 0.03],
            "W3_error": [None, None, None],
            "W4_error": [None, None, None],
            "parallax": [None, None, None],
            "parallax_error": [None, None, None],
            "parallax_over_error": [None, None, None],
            "pmra": [None, None, None],
            "pmdec": [None, None, None],
            "tmass_id": ["t1", "t2", "t3"],
            "tmass_quality": ["AAA", "AAA", "AAA"],
            "wise_id": ["w1", "w2", "w3"],
            "wise_quality": ["AA", "AA", "AA"],
        }
    ).to_csv(csv_path, index=False)

    tile = SkyTile("tile", 0, 1, -1, 0)
    parquet_path, inserted, excluded = ingest_prediction_tile(db_path, csv_path, tile, envelope, config)
    mark_tile_completed(db_path, tile, parquet_path, inserted=inserted, excluded=excluded, job_id="test_job")

    con = duckdb.connect(str(db_path))
    try:
        sources = con.execute("SELECT source_id FROM prediction_pool_sources ORDER BY source_id").fetchall()
        source_columns = [row[1] for row in con.execute("PRAGMA table_info('prediction_pool_sources')").fetchall()]
        manifest = con.execute("SELECT tile_id, row_count FROM prediction_pool_parquet_files").fetchall()
        effective = con.execute("SELECT tile_id, row_count FROM prediction_pool_effective_tiles").fetchall()
        exclusions = con.execute(
            "SELECT source_id, exclusion_label FROM prediction_pool_exclusions ORDER BY source_id"
        ).fetchall()
    finally:
        con.close()
    assert inserted == 1
    assert excluded == 2
    assert parquet_path.exists()
    assert sources == [(30,)]
    assert manifest == [("tile", 1)]
    assert effective == [("tile", 1)]
    assert "W4_error" in source_columns
    assert "G_flux_error" in source_columns
    assert "ruwe" not in source_columns
    assert exclusions == [(10, "known_wr"), (20, "known_non_wr_simbad")]


def test_prediction_pool_effective_tiles_exclude_subdivided_parents(tmp_path):
    db_path = tmp_path / "prediction_pool.duckdb"
    con = duckdb.connect(str(db_path))
    try:
        initialize_prediction_pool_database(con)
        now = pd.Timestamp("2026-06-09")
        con.execute(
            """
            INSERT INTO prediction_pool_tiles VALUES
            ('parent', 0, 10, -10, 0, 'skipped', 0, 0, NULL, NULL, 'subdivided_into_subtiles', ?),
            ('child_a', 0, 5, -10, -5, 'completed', 7, 0, 'child_a.parquet', NULL, NULL, ?),
            ('child_b', 5, 10, -5, 0, 'completed', 11, 0, 'child_b.parquet', NULL, NULL, ?)
            """,
            [now, now, now],
        )
        con.execute(
            """
            INSERT INTO prediction_pool_parquet_files VALUES
            ('child_a', 'child_a.parquet', 7, 100, ?),
            ('child_b', 'child_b.parquet', 11, 120, ?)
            """,
            [now, now],
        )
        refresh_prediction_pool_metadata_views(con)
        effective = con.execute("SELECT tile_id, row_count FROM prediction_pool_effective_tiles ORDER BY tile_id").fetchall()
        coverage = con.execute("SELECT tile_id, coverage_terminal, coverage_role FROM prediction_pool_tile_coverage ORDER BY tile_id").fetchall()
    finally:
        con.close()

    assert effective == [("child_a", 7), ("child_b", 11)]
    assert coverage == [
        ("child_a", True, "effective_tile"),
        ("child_b", True, "effective_tile"),
        ("parent", True, "subdivided_parent"),
    ]

    audit = audit_prediction_pool(db_path)
    assert audit["counts"]["prediction_pool_effective_tiles"] == 2
    assert audit["counts"]["prediction_pool_tile_coverage"] == 3


def test_local_storage_bytes_counts_database_and_staging(tmp_path):
    db_path = tmp_path / "prediction_pool.duckdb"
    staging = tmp_path / "staging"
    staging.mkdir()
    db_path.write_bytes(b"12345")
    (staging / "tile.csv").write_bytes(b"123")

    assert local_storage_bytes(db_path, staging) == 8
