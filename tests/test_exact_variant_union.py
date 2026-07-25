from __future__ import annotations

import pandas as pd
import pytest

from wr_detector.pipelines.exact_variant_union import (
    DEFAULT_VARIANT_BIT_REGISTRY,
    add_compatible_variant_mask,
    build_tile_count_record,
    build_variant_mask_schema,
    filter_sources_for_model_variant,
    merge_resumable_sources,
    merge_resumable_tiles,
    variant_astrometry_mask,
)


VARIANTS = ["strict_photometry", "relaxed_poe_3", "relaxed_photometry"]


def _eligibility_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_id": [1, 2, 3, 4],
            "exact_variant_keep__strict_photometry": [True, False, False, True],
            "exact_variant_keep__relaxed_photometry": [True, True, False, True],
            "exact_variant_keep__relaxed_poe_3": [False, True, False, True],
        }
    )


def test_exact_union_mask_and_count_are_consistent():
    schema = build_variant_mask_schema(VARIANTS)
    result = add_compatible_variant_mask(_eligibility_frame(), schema)

    assert result["passes_any_exact_variant"].tolist() == [True, True, False, True]
    assert result["compatible_variant_count"].tolist() == [2, 2, 0, 3]
    assert result.loc[~result["passes_any_exact_variant"], "compatible_variant_mask"].eq(0).all()
    for variant in VARIANTS:
        bit = schema.bit_for(variant)
        decoded = result["compatible_variant_mask"].map(lambda value: bool(int(value) & (1 << bit)))
        assert decoded.equals(result[f"exact_variant_keep__{variant}"])


def test_variant_order_does_not_change_schema_or_results():
    forward = build_variant_mask_schema(VARIANTS)
    reverse = build_variant_mask_schema(list(reversed(VARIANTS)))
    assert forward.version == reverse.version
    assert forward.sha256 == reverse.sha256
    left = add_compatible_variant_mask(_eligibility_frame(), forward)
    right = add_compatible_variant_mask(_eligibility_frame(), reverse)
    assert left["compatible_variant_mask"].equals(right["compatible_variant_mask"])


def test_new_variant_changes_schema_hash_and_derived_version_without_moving_old_bits():
    base = build_variant_mask_schema(VARIANTS)
    registry = {**DEFAULT_VARIANT_BIT_REGISTRY, "relaxed_poe_4": 14}
    extended = build_variant_mask_schema([*VARIANTS, "relaxed_poe_4"], registry=registry)

    assert extended.sha256 != base.sha256
    assert extended.version != base.version
    for variant in VARIANTS:
        assert extended.bit_for(variant) == base.bit_for(variant)


def test_astrometric_conditions_match_variant_definitions():
    frame = pd.DataFrame(
        {
            "parallax": [1.0, 1.0, -1.0, None],
            "parallax_over_error": [3.0, 2.99, 10.0, None],
        }
    )
    assert variant_astrometry_mask(frame, "relaxed_photometry").tolist() == [True] * 4
    assert variant_astrometry_mask(frame, "relaxed_parallax_soft").tolist() == [True, True, False, False]
    assert variant_astrometry_mask(frame, "relaxed_poe_3").tolist() == [True, False, False, False]


def test_scoring_subset_requires_the_model_variant_bit():
    schema = build_variant_mask_schema(VARIANTS)
    result = add_compatible_variant_mask(_eligibility_frame(), schema)
    scored = filter_sources_for_model_variant(result, "relaxed_poe_3", schema)

    assert scored["source_id"].tolist() == [2, 4]
    assert scored["exact_variant_keep__relaxed_poe_3"].all()


def test_tile_manifest_counts_reconcile_with_source_rows():
    schema = build_variant_mask_schema(VARIANTS)
    result = add_compatible_variant_mask(_eligibility_frame(), schema)
    result["known_excluded"] = [False, True, False, False]
    record = build_tile_count_record(
        result,
        pool_build_id="pool_test",
        tile_id="tile_1",
        schema=schema,
        known_excluded_column="known_excluded",
    )

    assert record["acquired_pre_locus"] == 4
    assert record["accepted_union"] == 3
    assert record["known_excluded"] == 1
    assert record["written"] == 2
    assert record["accepted_by_variant"]["relaxed_poe_3"] == 2


def test_resume_does_not_duplicate_sources_or_tiles_and_rejects_conflicts():
    sources = pd.DataFrame(
        {
            "pool_build_id": ["pool_test"],
            "tile_id": ["tile_1"],
            "source_id": [1],
            "source_hash_v1": ["same"],
        }
    )
    tiles = pd.DataFrame(
        {
            "pool_build_id": ["pool_test"],
            "tile_id": ["tile_1"],
            "acquisition_sha256": ["same"],
        }
    )
    assert len(merge_resumable_sources(sources, sources)) == 1
    assert len(merge_resumable_tiles(tiles, tiles)) == 1

    conflicting_sources = sources.assign(source_hash_v1="different")
    conflicting_tiles = tiles.assign(acquisition_sha256="different")
    with pytest.raises(ValueError, match="conflicting source hashes"):
        merge_resumable_sources(sources, conflicting_sources)
    with pytest.raises(ValueError, match="conflicting acquisition hashes"):
        merge_resumable_tiles(tiles, conflicting_tiles)
