import pandas as pd

from wr_detector.catalogs.simbad import (
    build_simbad_otype_tap_query,
    deduplicate_simbad_sources,
    exclude_wolf_rayet_rows,
    extract_gaia_dr3_source_id,
    normalize_simbad_result,
)


def test_extract_gaia_dr3_source_id_requires_explicit_dr3_identifier():
    assert extract_gaia_dr3_source_id("HD 1|Gaia DR3 123456|2MASS J000") == 123456
    assert extract_gaia_dr3_source_id("Gaia DR2 123456|HD 1") is None
    assert extract_gaia_dr3_source_id(pd.NA) is None


def test_build_simbad_otype_tap_query_filters_explicit_gaia_dr3_ids():
    query = build_simbad_otype_tap_query("Be*", row_limit=25)

    assert "TOP 25" in query
    assert "basic.otype = 'Be*'" in query
    assert "ident.id LIKE 'Gaia DR3 %'" in query


def test_normalize_simbad_result_preserves_provenance_and_deduplicates():
    raw = pd.DataFrame(
        {
            "MAIN_ID": ["A", "B", "C"],
            "RA": [1.0, 2.0, 3.0],
            "DEC": [4.0, 5.0, 6.0],
            "OTYPE": ["YSO", "Be*", "EclBin"],
            "OTYPES": ["YSO|Star", "Be*|Star", "EclBin|Star"],
            "SP_TYPE": ["O", "B", "A"],
            "IDS": ["Gaia DR3 10|A", "Gaia DR3 10|B", "Gaia DR3 20|C"],
            "simbad_query_type": ["YSO", "Be*", "EclBin"],
        }
    )

    normalized = normalize_simbad_result(raw)
    deduped = deduplicate_simbad_sources(normalized)

    assert normalized["source_id"].tolist() == [10, 10, 20]
    assert len(deduped) == 2
    assert set(["simbad_main_id", "simbad_main_type", "simbad_query_type"]).issubset(deduped.columns)


def test_exclude_wolf_rayet_rows_removes_type_matches_and_known_wr_ids():
    df = pd.DataFrame(
        {
            "source_id": [1, 2, 3],
            "simbad_main_type": ["YSO", "WolfRayet*", "Be*"],
            "simbad_other_types": ["Star", "Star", "Star"],
        }
    )

    out = exclude_wolf_rayet_rows(df, wr_source_ids=[3])

    assert out["source_id"].tolist() == [1]
