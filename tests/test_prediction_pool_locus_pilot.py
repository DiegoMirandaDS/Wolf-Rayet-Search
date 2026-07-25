from __future__ import annotations

import pandas as pd

from wr_detector.analysis.prediction_pool_locus_pilot import (
    ExactLocus,
    LocusPlane,
    analyze_geometric_containment,
    compare_locus_policies,
)


def test_exact_union_can_retain_source_rejected_by_midpoint_aggregate():
    frame = pd.DataFrame(
        {
            "source_id": [1],
            "G": [1.0],
            "BP": [0.0],
            "RP": [0.0],
            "J": [1.0],
            "H": [0.0],
            "Ks": [0.0],
            "W1": [1.0],
            "W2": [0.0],
            "tmass_quality": ["AAA"],
            "wise_quality": ["AA"],
            "parallax": [1.0],
            "parallax_error": [0.1],
            "parallax_over_error": [10.0],
        }
    )
    # Two parallel exact planes at opposite extremes. The current midpoint
    # aggregate can reject a point that lies on one exact plane.
    exact = {
        "relaxed_photometry": ExactLocus(
            variant="relaxed_photometry",
            planes=(
                LocusPlane("G_BP", "G_RP", 0.0, 0.65, 0.05),
                LocusPlane("J_H", "J_K", 0.0, 0.65, 0.05),
            ),
            aggregate_min_outlier_planes=2,
            source_path="fixture",
            source_sha256="fixture",
        )
    }
    envelope = {
        "fits": [
            {"x": "G_BP", "y": "G_RP", "slope_min": 0.0, "slope_max": 0.0, "intercept_min": -0.65, "intercept_max": 0.65, "threshold_max": 0.05, "transform": "signed_log1p"},
            {"x": "J_H", "y": "J_K", "slope_min": 0.0, "slope_max": 0.0, "intercept_min": -0.65, "intercept_max": 0.65, "threshold_max": 0.05, "transform": "signed_log1p"},
        ]
    }

    compared = compare_locus_policies(
        frame,
        envelope=envelope,
        exact_loci=exact,
        feature_sets={"features": ["G_BP", "parallax"]},
    )

    assert compared["aggregated_locus_keep"].eq(False).all()
    assert compared["passes_any_exact_locus"].all()
    assert compared["exact_not_aggregate"].all()


def test_variant_conditions_and_feature_availability_are_recorded():
    frame = pd.DataFrame(
        {
            "source_id": [1, 2],
            "G": [10.0, 10.0],
            "BP": [10.0, 10.0],
            "RP": [10.0, 10.0],
            "J": [9.0, 9.0],
            "H": [9.0, 9.0],
            "Ks": [9.0, 9.0],
            "W1": [8.0, 8.0],
            "W2": [8.0, 8.0],
            "tmass_quality": ["AAA", "BAA"],
            "wise_quality": ["AA", "BA"],
            "parallax": [1.0, -1.0],
            "parallax_error": [0.1, None],
            "parallax_over_error": [10.0, None],
        }
    )
    plane = LocusPlane("G_BP", "G_RP", 1.0, 0.0, 1.0)
    exact = {
        variant: ExactLocus(variant, (plane,), 2, "fixture", "fixture")
        for variant in ["strict_poe_3", "relaxed_photometry"]
    }
    envelope = {
        "fits": [
            {"x": "G_BP", "y": "G_RP", "slope_min": 1.0, "slope_max": 1.0, "intercept_min": 0.0, "intercept_max": 0.0, "threshold_max": 1.0, "transform": "signed_log1p"}
        ]
    }

    compared = compare_locus_policies(
        frame,
        envelope=envelope,
        exact_loci=exact,
        feature_sets={"with_error": ["G_BP", "parallax_error"]},
    )

    assert compared["quality_keep__strict_poe_3"].tolist() == [True, False]
    assert compared["astrometry_keep__strict_poe_3"].tolist() == [True, False]
    assert compared["quality_keep__relaxed_photometry"].tolist() == [True, True]
    assert compared["features_available__with_error"].tolist() == [True, False]


def test_geometric_containment_audit_finds_envelope_valid_counterexample():
    envelope = {
        "color_bounds": {
            "G_BP": {"min": -1.0, "max": 1.0},
            "G_RP": {"min": -1.0, "max": 1.0},
            "BP_RP": {"min": -2.0, "max": 2.0},
            "J_H": {"min": -1.0, "max": 1.0},
            "H_K": {"min": -1.0, "max": 1.0},
            "J_K": {"min": -2.0, "max": 2.0},
        },
        "fits": [
            {
                "x": "G_BP",
                "y": "G_RP",
                "slope_min": 0.0,
                "slope_max": 0.0,
                "intercept_min": 0.0,
                "intercept_max": 0.0,
                "threshold_max": 0.1,
                "transform": "signed_log1p",
            }
        ],
    }
    exact_loci = {
        "relaxed_photometry": ExactLocus(
            variant="relaxed_photometry",
            planes=(LocusPlane("G_BP", "G_RP", 0.0, 0.5, 0.1),),
            aggregate_min_outlier_planes=1,
            source_path="fixture",
            source_sha256="fixture",
        )
    }

    summary, examples = analyze_geometric_containment(
        envelope=envelope,
        exact_loci=exact_loci,
        sobol_power=10,
        seed=7,
    )

    assert not bool(summary.loc[0, "analytic_all_planes_contained"])
    assert bool(summary.loc[0, "constructive_counterexample_found"])
    assert not examples.empty
    assert examples["sample_role"].eq("synthetic_geometric_witness_not_gaia_source").all()
