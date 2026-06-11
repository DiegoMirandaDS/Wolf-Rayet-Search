import pandas as pd

from wr_detector.catalogs.gwrc import clean_gwrc_table, extract_gaia_dr3_reference


def test_clean_gwrc_table_strips_nbsp_and_numeric_columns():
    raw = pd.DataFrame(
        {
            "ID": ["1\xa0"],
            "Alias1": [" DR3 12345\xa0"],
            "Galactic Longitude (deg)": ["122.5\xa0"],
            "WR#": [" 1\xa0"],
        }
    )

    cleaned = clean_gwrc_table(raw)

    assert cleaned.loc[0, "Alias1"] == "DR3 12345"
    assert cleaned.loc[0, "WR#"] == "1"
    assert cleaned.loc[0, "ID"] == 1
    assert cleaned.loc[0, "Galactic Longitude (deg)"] == 122.5


def test_extract_gaia_dr3_reference_builds_source_id_and_designation():
    df = pd.DataFrame(
        {
            "WR#": ["1", "2", "3"],
            "Alias1": ["DR3 12345", "HIP 1", pd.NA],
        }
    )

    ref = extract_gaia_dr3_reference(df)

    assert len(ref) == 1
    assert ref.loc[0, "source_id"] == 12345
    assert ref.loc[0, "gaia_designation"] == "Gaia DR3 12345"
    assert ref.loc[0, "wr_id"] == "1"
