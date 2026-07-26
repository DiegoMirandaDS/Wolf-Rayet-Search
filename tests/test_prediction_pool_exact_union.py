from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from wr_detector.pipelines.exact_variant_union import (
    ExactLocus,
    LocusPlane,
    build_variant_mask_schema,
)
from wr_detector.pipelines.prediction_pool import BASE_COLUMNS, SkyTile
from wr_detector.pipelines.prediction_pool_exact_union import (
    completed_tile_is_valid,
    deduplicate_tmass_crossmatches,
    initialize_exact_union_database,
    persist_acquisition_tile,
    prepare_acquisition_frame,
    refresh_exact_union_views,
    register_completed_tile,
)


def _raw_acquisition() -> pd.DataFrame:
    rows = 3
    values = {
        "source_id": [1, 2, 3],
        "gaia_designation": ["Gaia 1", "Gaia 2", "Gaia 3"],
        "ra": [1.0, 1.1, 1.2],
        "dec": [-1.0, -1.1, -1.2],
        "G": [10.0, 10.0, 10.0],
        "BP": [11.0, 11.0, 11.0],
        "RP": [10.0, 10.0, 8.0],
        "G_flux": [100.0] * rows,
        "G_flux_error": [1.0] * rows,
        "BP_flux": [90.0] * rows,
        "BP_flux_error": [1.0] * rows,
        "RP_flux": [110.0] * rows,
        "RP_flux_error": [1.0] * rows,
        "J": [8.0] * rows,
        "H": [7.5] * rows,
        "Ks": [7.2] * rows,
        "J_error": [0.02] * rows,
        "H_error": [0.02] * rows,
        "Ks_error": [0.02] * rows,
        "W1": [7.0] * rows,
        "W2": [6.9] * rows,
        "W3": [None] * rows,
        "W4": [None] * rows,
        "W1_error": [0.03] * rows,
        "W2_error": [0.03] * rows,
        "W3_error": [None] * rows,
        "W4_error": [None] * rows,
        "parallax": [1.0] * rows,
        "parallax_error": [0.2] * rows,
        "parallax_over_error": [5.0] * rows,
        "pmra": [None] * rows,
        "pmdec": [None] * rows,
        "tmass_id": ["t1", "t2", "t3"],
        "tmass_quality": ["AAA", "BBB", "CCC"],
        "wise_id": ["w1", "w2", "w3"],
        "wise_quality": ["AA", "BB", "CC"],
    }
    frame = pd.DataFrame(values)
    assert set(BASE_COLUMNS) == set(frame.columns)
    return frame


def _loci() -> dict[str, ExactLocus]:
    plane = LocusPlane(
        x="G_BP",
        y="G_RP",
        slope=0.0,
        intercept=0.0,
        threshold=0.01,
    )
    return {
        variant: ExactLocus(
            variant=variant,
            planes=(plane,),
            aggregate_min_outlier_planes=1,
            source_path=f"{variant}.parquet",
            source_sha256=variant * 2,
            locus_run_id=f"run_{variant}",
        )
        for variant in ["strict_photometry", "relaxed_photometry"]
    }


def test_acquisition_keeps_ineligible_and_known_rows_but_masks_compatibility():
    loci = _loci()
    schema = build_variant_mask_schema(list(loci))
    frame = prepare_acquisition_frame(
        _raw_acquisition(),
        tile=SkyTile("tile", 0, 5, -5, 0),
        loci=loci,
        schema=schema,
        known_source_labels={1: "known_wr"},
    )

    assert len(frame) == 3
    assert frame["passes_any_exact_variant"].tolist() == [True, True, False]
    assert frame["compatible_variant_count"].tolist() == [2, 1, 0]
    assert frame["known_source_exclusion"].fillna("").tolist() == [
        "known_wr",
        "",
        "",
    ]
    assert frame["source_hash_v1"].str.len().eq(16).all()


def test_tmass_duplicate_resolution_is_deterministic_and_preserves_alternatives():
    raw = _raw_acquisition().iloc[[0, 1]].copy()
    duplicate = raw.iloc[[0]].copy()
    duplicate["source_id"] = 1
    duplicate["tmass_id"] = "t1_better"
    duplicate["tmass_quality"] = "AAA"
    duplicate["J_error"] = 0.01
    raw.loc[raw.index[0], "tmass_quality"] = "BAA"
    raw.loc[raw.index[0], "tmass_id"] = "t1_other"
    raw = pd.concat([raw, duplicate], ignore_index=True)

    resolved = deduplicate_tmass_crossmatches(raw)
    selected = resolved[resolved["source_id"].eq(1)].iloc[0]

    assert len(resolved) == 2
    assert selected["tmass_id"] == "t1_better"
    assert selected["tmass_candidate_count"] == 2
    assert "t1_better" in selected["tmass_alternatives_json"]
    assert "t1_other" in selected["tmass_alternatives_json"]


def test_validated_tile_resume_and_logical_eligible_view(tmp_path: Path):
    loci = _loci()
    schema = build_variant_mask_schema(list(loci))
    tile = SkyTile("tile", 0, 5, -5, 0)
    frame = prepare_acquisition_frame(
        _raw_acquisition(),
        tile=tile,
        loci=loci,
        schema=schema,
        known_source_labels={1: "known_wr"},
    )
    db_path = tmp_path / "pool.duckdb"
    acquisition_dir = tmp_path / "acquisition"
    manifest_dir = tmp_path / "manifests"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("test: true\n", encoding="utf-8")
    envelope = {
        "sha256": "envelope",
        "color_bounds": {},
        "variants": [],
    }
    coverage = {"coverage_fraction": 1.0}
    initialize_exact_union_database(
        db_path,
        pool_build_id="build",
        config_path=config_path,
        envelope=envelope,
        coverage=coverage,
        schema=schema,
        loci=loci,
    )
    result = persist_acquisition_tile(
        frame,
        tile=tile,
        pool_build_id="build",
        acquisition_dir=acquisition_dir,
        manifest_dir=manifest_dir,
        compression="zstd",
        schema=schema,
        envelope=envelope,
        loci=loci,
        adql="SELECT test",
        gaia_job_id="job",
        row_limit=None,
    )
    with duckdb.connect(str(db_path)) as con:
        con.execute(
            """
            INSERT INTO exact_union_tiles (
                pool_build_id, tile_id, ra_min, ra_max, dec_min, dec_max,
                status, updated_at
            ) VALUES ('build', 'tile', 0, 5, -5, 0, 'running', current_timestamp)
            """
        )
    register_completed_tile(db_path, result)
    refresh_exact_union_views(db_path, acquisition_dir)

    assert completed_tile_is_valid(
        db_path,
        acquisition_dir=acquisition_dir,
        pool_build_id="build",
        tile_id="tile",
    )
    with duckdb.connect(str(db_path), read_only=True) as con:
        acquisition_ids = con.execute(
            "SELECT source_id FROM acquisition_sources ORDER BY source_id"
        ).fetchall()
        eligible_ids = con.execute(
            "SELECT source_id FROM prediction_pool_sources ORDER BY source_id"
        ).fetchall()
    assert acquisition_ids == [(1,), (2,), (3,)]
    assert eligible_ids == [(2,)]
