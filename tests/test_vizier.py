import pandas as pd

from wr_detector.catalogs.vizier import select_duplicates_by_radius


def test_select_duplicates_by_radius_keeps_singletons_and_resolves_close_duplicate():
    df = pd.DataFrame(
        {
            "source_id": [1, 2, 2, 3, 3],
            "sep_arcsec": [3.0, 0.8, 2.0, 1.2, 1.3],
            "value": ["single", "best", "far", "amb1", "amb2"],
        }
    )

    selected, stats = select_duplicates_by_radius(
        df,
        primary_radius_arcsec=1.5,
        secondary_radius_arcsec=1.0,
    )

    assert set(selected["source_id"]) == {1, 2}
    assert selected.loc[selected["source_id"] == 2, "value"].item() == "best"
    assert stats["n_dropped_ambiguous"] == 1
    assert stats["dropped_source_ids"] == [3]
