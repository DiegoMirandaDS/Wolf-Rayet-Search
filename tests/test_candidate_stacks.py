from __future__ import annotations

import pandas as pd

from wr_detector.apps.explorer_ui.charts import stack_recovery_lines, stack_tradeoff_bars
from wr_detector.modeling.candidate_stacks import (
    _recovery_curve,
    _tradeoff_curve,
    compatible_layer_results,
    validator_pairs,
)


def test_compatible_layer_results_requires_matching_lineage_and_kept_locus():
    first = pd.Series(
        {
            "dataset_variant": "relaxed_photometry",
            "dataset_sha256": "dataset-a",
            "models_config_sha256": "config-a",
        }
    )
    rows = pd.DataFrame(
        [
            _layer_row("WN", dataset_sha256="dataset-a"),
            _layer_row("WC", dataset_sha256="dataset-a"),
            _layer_row("WN", dataset_sha256="dataset-b"),
            {**_layer_row("WC", dataset_sha256="dataset-a"), "require_color_locus_keep": False},
        ]
    )

    compatible = compatible_layer_results(first, rows)

    assert compatible["subtype"].tolist() == ["WN", "WC"]
    assert set(compatible["dataset_sha256"]) == {"dataset-a"}


def test_validator_pairs_only_returns_complete_wn_wc_pairs():
    rows = pd.DataFrame(
        [
            _layer_row("WN", feature_set="enriched_tabular", method="gaussian_mixture"),
            _layer_row("WC", feature_set="enriched_tabular", method="gaussian_mixture"),
            _layer_row("WN", feature_set="current_colors", method="one_class_svm"),
        ]
    )

    pairs = validator_pairs(rows)

    assert pairs["pair_key"].tolist() == ["enriched_tabular::gaussian_mixture"]
    assert pairs.iloc[0]["wn_holdout_positive_retention"] == 0.9
    assert pairs.iloc[0]["wc_threshold_calibration_negative_pass_rate"] == 0.1


def test_stack_curves_report_fixed_budget_delta_and_conditional_filtering():
    first = pd.DataFrame(
        {
            "source_id": [1, 2, 3, 4, 5],
            "target": [1, 0, 1, 0, 1],
            "second_layer_pass": [True, False, False, True, True],
            "score": [0.9, 0.8, 0.7, 0.6, 0.5],
        }
    )
    pass_first = first.sort_values(
        ["second_layer_pass", "score"], ascending=[False, False], kind="mergesort"
    )

    recovery = _recovery_curve(first, pass_first, [2, 4])
    tradeoff = _tradeoff_curve(first, [2, 4])

    pass_wr_at_2 = recovery[
        recovery["policy"].eq("Pass-first") & recovery["budget"].eq(2)
    ]["wr_recovered"].item()
    assert pass_wr_at_2 == 1
    assert tradeoff.loc[tradeoff["input_budget"].eq(4), "wr_retention"].item() == 0.5
    assert tradeoff.loc[
        tradeoff["input_budget"].eq(4), "negative_removal_rate"
    ].item() == 0.5

    assert stack_recovery_lines(recovery).to_dict()["height"] == 320
    assert stack_tradeoff_bars(tradeoff).to_dict()["height"] == 300


def _layer_row(
    subtype: str,
    *,
    dataset_sha256: str = "dataset-a",
    feature_set: str = "enriched_tabular",
    method: str = "gaussian_mixture",
) -> dict[str, object]:
    return {
        "dataset_variant": "relaxed_photometry",
        "dataset_sha256": dataset_sha256,
        "models_config_sha256": "config-a",
        "require_color_locus_keep": True,
        "status": "accepted",
        "feature_set": feature_set,
        "method": method,
        "subtype": subtype,
        "model_path": f"models/{subtype}.joblib",
        "holdout_positive_count": 10,
        "holdout_positive_retention": 0.9,
        "holdout_negative_pass_rate": 0.1,
        "threshold_calibration_negative_pass_rate": 0.1,
        "holdout_average_precision": 0.5,
        "holdout_roc_auc": 0.9,
    }
