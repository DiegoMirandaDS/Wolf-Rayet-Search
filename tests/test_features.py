import pandas as pd
import pytest

from wr_detector.db import write_reference_database, write_simbad_negative_database
from wr_detector.features import (
    COLOR_LOCUS_COLUMNS,
    add_color_features,
    annotate_color_locus,
    evaluate_color_locus_planes,
    export_color_locus_datasets,
    export_reference_datasets,
    export_simbad_negative_datasets,
    fit_log_color_locus,
    make_dataset_variants,
    make_positive_color_mask,
)


def test_add_color_features_uses_magnitude_differences():
    df = pd.DataFrame(
        {
            "BP": [12.0],
            "RP": [10.0],
            "G": [11.0],
            "J": [8.0],
            "H": [7.5],
            "Ks": [7.0],
            "W1": [6.8],
            "W2": [6.6],
        }
    )

    out = add_color_features(df)

    assert out.loc[0, "BP_RP"] == 2.0
    assert out.loc[0, "G_BP"] == -1.0
    assert out.loc[0, "J_K"] == 1.0
    assert out.loc[0, "W1_W2"] == pytest.approx(0.2)


def test_make_dataset_variants_applies_photometry_families_and_parallax_filters():
    df = pd.DataFrame(
        {
            "source_id": [1, 2, 3, 4, 5],
            "ra": [1.0, 2.0, 3.0, 4.0, 5.0],
            "dec": [1.0, 2.0, 3.0, 4.0, 5.0],
            "G": [10.0] * 5,
            "BP": [11.0] * 5,
            "RP": [9.0] * 5,
            "J": [8.0] * 5,
            "H": [7.5] * 5,
            "Ks": [7.0] * 5,
            "W1": [6.8] * 5,
            "W2": [6.6] * 5,
            "tmass_quality": ["AAA", "BAA", "AAA", "AAA", "CAA"],
            "wise_quality": ["AAAB", "AAAA", "BAAA", "AAAA", "AAAA"],
            "parallax": [0.1, 0.1, 0.1, 0.0, 0.1],
            "parallax_over_error": [5.0, 5.0, 0.5, 5.0, 5.0],
            "ruwe": [1.0] * 5,
        }
    )
    filters = {
        "photometry": {
            "required_columns": ["source_id", "ra", "dec", "G", "BP", "RP", "J", "H", "Ks", "W1", "W2"],
            "families": {
                "strict": {
                    "twomass_allowed_qualities": ["A"],
                    "wise_allowed_qualities": ["A"],
                },
                "relaxed": {
                    "twomass_allowed_qualities": ["A", "B"],
                    "wise_allowed_qualities": ["A", "B"],
                },
            },
        },
        "parallax_soft": {"min_parallax": 0.0},
        "parallax_over_error_variants": {
            "poe_1": {"min_parallax": 0.0, "min_parallax_over_error": 1.0},
            "poe_2": {"min_parallax": 0.0, "min_parallax_over_error": 2.0},
            "poe_3": {"min_parallax": 0.0, "min_parallax_over_error": 3.0},
        },
    }

    variants = make_dataset_variants(df, filters)

    assert set(variants) == {
        "strict_photometry",
        "strict_parallax_soft",
        "strict_poe_1",
        "strict_poe_2",
        "strict_poe_3",
        "relaxed_photometry",
        "relaxed_parallax_soft",
        "relaxed_poe_1",
        "relaxed_poe_2",
        "relaxed_poe_3",
    }
    assert variants["strict_photometry"]["source_id"].tolist() == [1, 4]
    assert variants["relaxed_photometry"]["source_id"].tolist() == [1, 2, 3, 4]
    assert variants["strict_parallax_soft"]["source_id"].tolist() == [1]
    assert variants["relaxed_parallax_soft"]["source_id"].tolist() == [1, 2, 3]
    assert variants["relaxed_poe_1"]["source_id"].tolist() == [1, 2]
    assert variants["relaxed_poe_2"]["source_id"].tolist() == [1, 2]
    assert variants["relaxed_poe_3"]["source_id"].tolist() == [1, 2]
    assert variants["relaxed_poe_3"]["dataset_family"].unique().tolist() == ["relaxed"]
    assert variants["relaxed_poe_3"]["astrometric_subset"].unique().tolist() == ["poe_3"]


def test_make_positive_color_mask_requires_positive_values():
    df = pd.DataFrame(
        {
            "BP_J": [2.0, 0.0, -1.0, None],
            "BP_RP": [1.0, 1.0, 1.0, 1.0],
        }
    )

    mask = make_positive_color_mask(df, x_color="BP_J", y_color="BP_RP")

    assert mask.tolist() == [True, False, False, False]


def test_color_locus_fit_and_annotation_flags_outliers():
    df = pd.DataFrame(
        {
            "BP_J": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100],
            "BP_RP": [2, 4, 6, 8, 10, 12, 14, 16, 18, 2],
        }
    )

    fit = fit_log_color_locus(
        df,
        x_color="BP_J",
        y_color="BP_RP",
        min_positive_fraction=0.9,
        estimator="huber",
    )
    annotated = annotate_color_locus(df, fit, residual_sigma_threshold=2.5)

    assert fit["valid"] is True
    assert set(COLOR_LOCUS_COLUMNS).issubset(annotated.columns)
    assert annotated["color_locus_outlier"].sum() >= 1
    assert annotated["color_locus_keep"].sum() < len(annotated)


def test_evaluate_color_locus_planes_rejects_low_positive_fraction():
    df = pd.DataFrame(
        {
            "BP_J": [1.0, 2.0, -1.0, -2.0],
            "BP_RP": [1.0, 2.0, 3.0, 4.0],
            "G_K": [1.0, 2.0, 3.0, 4.0],
        }
    )

    fits = evaluate_color_locus_planes(
        df,
        [{"x": "BP_J", "y": "BP_RP"}, {"x": "G_K", "y": "BP_RP"}],
        min_positive_fraction=0.9,
        estimator="linear",
    )

    assert fits.iloc[0]["x"] == "G_K"
    assert bool(fits.iloc[0]["valid"]) is True
    assert bool(fits.iloc[1]["valid"]) is False


def test_export_color_locus_datasets_annotates_reference_and_simbad_variants(tmp_path):
    processed = tmp_path / "processed"
    simbad_processed = tmp_path / "simbad_processed"
    processed.mkdir()
    simbad_processed.mkdir()
    base = add_color_features(
        pd.DataFrame(
            {
                "source_id": range(1, 11),
                "BP": [11, 12, 13, 14, 15, 16, 17, 18, 19, 110],
                "RP": [10, 10, 10, 10, 10, 10, 10, 10, 10, 108],
                "G": [10.5] * 10,
                "J": [8, 8, 8, 8, 8, 8, 8, 8, 8, 108],
                "H": [7.5] * 10,
                "Ks": [7.0] * 10,
                "W1": [6.8] * 10,
                "W2": [6.6] * 10,
                "sample_label": ["wr"] * 10,
            }
        )
    )
    simbad = base.copy()
    simbad["sample_label"] = "non_wr_simbad"
    simbad["simbad_main_type"] = ["YSO"] * 10
    base.to_parquet(processed / "strict_poe3.parquet", index=False)
    simbad.to_parquet(simbad_processed / "simbad_strict_poe3.parquet", index=False)
    paths = tmp_path / "paths.yaml"
    paths.write_text(
        f"processed_reference_dir: {processed.as_posix()}\n"
        f"processed_simbad_negative_dir: {simbad_processed.as_posix()}\n",
        encoding="utf-8",
    )
    filters = tmp_path / "filters.yaml"
    filters.write_text(
        "\n".join(
            [
                f"paths_config: {paths.as_posix()}",
                "output_files:",
                "  strict_poe_3: strict_poe3.parquet",
                "simbad_negative_output_files:",
                "  strict_poe_3: simbad_strict_poe3.parquet",
                "color_locus:",
                "  dataset_variants:",
                "    - strict_poe_3",
                "  reference_output_template: wr_{variant}_color.parquet",
                "  simbad_negative_output_template: simbad_{variant}_color.parquet",
                "  estimator: linear",
                "  transform: signed_log1p",
                "  selection_policy: all_planes",
                "  min_positive_fraction: 0.9",
                "  residual_sigma_threshold: 2.5",
                "  candidate_planes:",
                "    - x: G_BP",
                "      y: G_RP",
            ]
        ),
        encoding="utf-8",
    )

    result = export_color_locus_datasets(filters)
    exported = pd.read_parquet(processed / "wr_strict_poe_3_color.parquet")
    simbad_exported = pd.read_parquet(simbad_processed / "simbad_strict_poe_3_color.parquet")

    assert result["variants"]["strict_poe_3"]["reference_rows"] == 10
    assert result["variants"]["strict_poe_3"]["simbad_negative_rows"] == 10
    assert len(exported) == 10
    assert len(simbad_exported) == 10
    assert set(COLOR_LOCUS_COLUMNS).issubset(exported.columns)
    assert "W1_W2" in exported.columns
    fit_planes = result["variants"]["strict_poe_3"]["candidate_fits"]
    assert all(fit["x"] != "W1_W2" and fit["y"] != "W1_W2" for fit in fit_planes)


def test_signed_log_color_locus_supports_negative_gaia_color():
    df = pd.DataFrame(
        {
            "G_BP": [-3.0, -2.0, -1.0, 0.0, 1.0],
            "G_RP": [-2.0, -1.0, 0.0, 1.0, 2.0],
        }
    )

    fit = fit_log_color_locus(
        df,
        x_color="G_BP",
        y_color="G_RP",
        min_positive_fraction=1.0,
        estimator="linear",
        transform="signed_log1p",
    )

    assert fit["valid"] is True
    assert fit["n"] == 5


def test_reference_and_simbad_exports_share_model_ready_schema(tmp_path):
    paths = tmp_path / "paths.yaml"
    ref_db = tmp_path / "wr.duckdb"
    simbad_db = tmp_path / "simbad.duckdb"
    ref_out = tmp_path / "ref"
    simbad_out = tmp_path / "simbad"
    paths.write_text(
        "\n".join(
            [
                f"wr_reference_db: {ref_db.as_posix()}",
                f"simbad_negative_db: {simbad_db.as_posix()}",
                f"processed_reference_dir: {ref_out.as_posix()}",
                f"processed_simbad_negative_dir: {simbad_out.as_posix()}",
            ]
        ),
        encoding="utf-8",
    )
    filters = tmp_path / "filters.yaml"
    filters.write_text(
        "\n".join(
            [
                f"paths_config: {paths.as_posix()}",
                "photometry:",
                "  required_columns: [source_id, ra, dec, G, BP, RP, J, H, Ks, W1, W2]",
                "  families:",
                "    strict:",
                "      twomass_allowed_qualities: [A]",
                "      wise_allowed_qualities: [A]",
                "    relaxed:",
                "      twomass_allowed_qualities: [A, B]",
                "      wise_allowed_qualities: [A, B]",
                "parallax_soft:",
                "  min_parallax: 0.0",
                "parallax_over_error_variants:",
                "  poe_1:",
                "    min_parallax: 0.0",
                "    min_parallax_over_error: 1.0",
                "  poe_3:",
                "    min_parallax: 0.0",
                "    min_parallax_over_error: 3.0",
                "output_files:",
                "  strict_photometry: wr_strict_photometry.parquet",
                "  strict_parallax_soft: wr_strict_soft.parquet",
                "  strict_poe_1: wr_strict_poe1.parquet",
                "  strict_poe_3: wr_strict_poe3.parquet",
                "  relaxed_photometry: wr_relaxed_photometry.parquet",
                "  relaxed_parallax_soft: wr_relaxed_soft.parquet",
                "  relaxed_poe_1: wr_relaxed_poe1.parquet",
                "  relaxed_poe_3: wr_relaxed_poe3.parquet",
                "simbad_negative_output_files:",
                "  strict_photometry: simbad_strict_photometry.parquet",
                "  strict_parallax_soft: simbad_strict_soft.parquet",
                "  strict_poe_1: simbad_strict_poe1.parquet",
                "  strict_poe_3: simbad_strict_poe3.parquet",
                "  relaxed_photometry: simbad_relaxed_photometry.parquet",
                "  relaxed_parallax_soft: simbad_relaxed_soft.parquet",
                "  relaxed_poe_1: simbad_relaxed_poe1.parquet",
                "  relaxed_poe_3: simbad_relaxed_poe3.parquet",
            ]
        ),
        encoding="utf-8",
    )
    gaia = pd.DataFrame(
        {
            "source_id": [1],
            "ra": [1.0],
            "dec": [2.0],
            "G": [10.0],
            "BP": [11.0],
            "RP": [9.0],
            "parallax": [0.1],
            "parallax_error": [0.01],
            "parallax_over_error": [5.0],
            "ruwe": [1.0],
            "pmra": [0.0],
            "pmdec": [0.0],
        }
    )
    tmass = pd.DataFrame({"source_id": [1], "J": [8.0], "H": [7.5], "Ks": [7.0], "tmass_quality": ["AAA"], "match_method": ["gaia_xmatch"]})
    wise = pd.DataFrame({"source_id": [1], "W1": [6.8], "W2": [6.6], "W3": [6.0], "W4": [5.5], "wise_quality": ["AAAA"], "match_method": ["gaia_xmatch"]})
    log = pd.DataFrame({"stage": [], "method": [], "rows": [], "notes": []})
    write_reference_database(
        ref_db,
        snapshot={},
        wr_catalog_raw=pd.DataFrame(),
        wr_reference=pd.DataFrame({"source_id": [1], "wr_id": ["1"]}),
        gaia_sources=gaia,
        twomass_matches=tmass,
        wise_matches=wise,
        crossmatch_log=log,
    )
    write_simbad_negative_database(
        simbad_db,
        simbad_negative_raw=pd.DataFrame(),
        simbad_negative_sources=pd.DataFrame({"source_id": [1], "simbad_main_id": ["A"], "simbad_main_type": ["YSO"], "simbad_other_types": ["Star"], "simbad_sp_type": ["O"], "simbad_query_type": ["YSO"]}),
        gaia_sources=gaia,
        twomass_matches=tmass,
        wise_matches=wise,
        crossmatch_log=log,
    )

    ref_counts = export_reference_datasets(filters)
    simbad_counts = export_simbad_negative_datasets(filters)
    ref_export = pd.read_parquet(ref_out / "wr_strict_poe3.parquet")
    simbad_export = pd.read_parquet(simbad_out / "simbad_strict_poe3.parquet")

    assert ref_counts["strict_poe_3"] == 1
    assert simbad_counts["strict_poe_3"] == 1
    assert ref_counts["relaxed_poe_3"] == 1
    assert simbad_counts["relaxed_poe_3"] == 1
    shared = {
        "source_id",
        "G",
        "BP",
        "RP",
        "J",
        "H",
        "Ks",
        "W1",
        "W2",
        "BP_RP",
        "J_K",
        "W1_W2",
        "dataset_family",
        "astrometric_subset",
        "sample_label",
    }
    assert shared.issubset(ref_export.columns)
    assert shared.issubset(simbad_export.columns)
    assert ref_export.loc[0, "dataset_family"] == "strict"
    assert ref_export.loc[0, "astrometric_subset"] == "poe_3"
    assert ref_export.loc[0, "sample_label"] == "wr"
    assert simbad_export.loc[0, "sample_label"] == "non_wr_simbad"
