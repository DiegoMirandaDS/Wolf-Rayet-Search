from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from wr_detector.apps.explorer_ui.case_plots import (
    build_case_plot_base,
    build_interactive_case_plot,
    build_selected_source_layer,
    chart_payload_bytes,
    export_case_plot_png,
)
from wr_detector.modeling.case_visualization import (
    CasePlotSettings,
    case_data_fingerprint,
    mollweide_project,
    prepare_case_plot_data,
)


VISIBLE_BUDGET_STATES = (
    "Background",
    "Contaminant @K",
    "WR outside @K",
    "WR recovered @K",
)


def test_mollweide_uses_astronomical_orientation_and_standard_extent():
    x, y = mollweide_project(
        [0.0, 90.0, 270.0, 45.0, 45.0],
        [0.0, 0.0, 0.0, 90.0, -90.0],
    )

    assert x[0] == pytest.approx(0.0, abs=1e-12)
    assert x[1] < 0  # Galactic longitude grows to the left.
    assert x[2] > 0
    assert x[3] == pytest.approx(0.0, abs=1e-10)
    assert x[4] == pytest.approx(0.0, abs=1e-10)
    assert y[3] == pytest.approx(np.sqrt(2.0))
    assert y[4] == pytest.approx(-np.sqrt(2.0))
    assert (2.0 * np.sqrt(2.0)) / np.sqrt(2.0) == pytest.approx(2.0)


def test_preparation_limits_only_interactive_background_and_preserves_counts():
    cases = _case_frame(background_rows=5_100)
    density_settings = CasePlotSettings(
        top_k=2,
        visible_states=VISIBLE_BUDGET_STATES,
        background_mode="density",
        background_limit=500,
    )
    density = prepare_case_plot_data(
        cases,
        settings=density_settings,
        fingerprint="density-fingerprint",
    )
    frame = density.frame("photometric")

    assert frame.prepared_source_count == 5_103
    assert frame.background_source_count == 5_100
    assert frame.relevant_source_count == 3
    assert int(frame.density["count"].sum()) == 5_100
    assert frame.interactive_source_count == frame.prepared_source_count
    assert frame.interactive_mark_count == len(frame.density) + 3

    sample_settings = CasePlotSettings(
        top_k=2,
        visible_states=VISIBLE_BUDGET_STATES,
        background_mode="sample",
        background_limit=500,
        random_seed=42,
    )
    first = prepare_case_plot_data(
        cases,
        settings=sample_settings,
        fingerprint="sample-fingerprint",
    )
    second = prepare_case_plot_data(
        cases,
        settings=sample_settings,
        fingerprint="sample-fingerprint",
    )
    sampled = first.frame("photometric")

    assert sampled.prepared_source_count == 5_103
    assert sampled.relevant_source_count == 3
    assert len(sampled.interactive_background) == 500
    assert sampled.interactive_source_count == 503
    assert sampled.interactive_background["source_id"].tolist() == (
        second.frame("photometric").interactive_background["source_id"].tolist()
    )


def test_selection_layer_is_one_row_and_does_not_change_cached_base():
    cases = _case_frame(background_rows=250)
    settings = CasePlotSettings(
        top_k=2,
        visible_states=VISIBLE_BUDGET_STATES,
        background_mode="density",
    )
    prepared = prepare_case_plot_data(
        cases,
        settings=settings,
        fingerprint="selection-fingerprint",
    )
    base = build_case_plot_base("photometric", prepared)
    base_spec = base.to_dict()
    before = prepared.frame("photometric").full.copy(deep=True)

    selected_id = int(cases.iloc[-1]["source_id"])
    selected_layers = build_selected_source_layer(
        "photometric",
        prepared,
        selected_id,
    )
    chart = build_interactive_case_plot(
        "photometric",
        prepared,
        selected_source_id=selected_id,
        base_chart=base,
    )

    assert len(selected_layers) == 2
    for layer in selected_layers:
        datasets = layer.to_dict()["datasets"]
        assert sum(len(rows) for rows in datasets.values()) == 1
    pd.testing.assert_frame_equal(before, prepared.frame("photometric").full)
    assert chart_payload_bytes(chart) < 2_000_000
    assert base_spec == base.to_dict()


def test_png_uses_every_prepared_source_not_interactive_sample(tmp_path):
    cases = _case_frame(background_rows=800)
    settings = CasePlotSettings(
        top_k=2,
        visible_states=VISIBLE_BUDGET_STATES,
        background_mode="sample",
        background_limit=100,
    )
    prepared = prepare_case_plot_data(
        cases,
        settings=settings,
        fingerprint="export-fingerprint",
    )
    output = tmp_path / "photometric.png"

    exported = export_case_plot_png(
        "photometric",
        prepared,
        output,
        dpi=200,
    )

    assert output.exists()
    assert output.stat().st_size > 0
    assert exported.interactive_source_count == 103
    assert exported.png_source_count == exported.prepared_source_count == 803


def test_case_fingerprint_changes_when_reused_result_predictions_change(tmp_path):
    db_path = tmp_path / "history.duckdb"
    config_path = tmp_path / "models.yaml"
    config_path.write_text(
        "\n".join(
            [
                "outputs:",
                f"  training_history_db: {db_path.as_posix()}",
            ]
        ),
        encoding="utf-8",
    )
    with duckdb.connect(str(db_path)) as con:
        con.execute(
            """
            CREATE TABLE model_results AS
            SELECT
                'run_a'::VARCHAR AS run_id,
                'reused'::VARCHAR AS result_id,
                'xgboost'::VARCHAR AS model,
                'none'::VARCHAR AS sampler,
                'strict_photometry'::VARCHAR AS dataset_variant,
                'colors_parallax'::VARCHAR AS feature_set,
                0.5::DOUBLE AS selected_threshold,
                'abc'::VARCHAR AS model_sha256
            """
        )
        con.execute(
            """
            CREATE TABLE model_predictions AS
            SELECT
                'run_a'::VARCHAR AS run_id,
                'reused'::VARCHAR AS result_id,
                'holdout'::VARCHAR AS split,
                1::BIGINT AS source_id,
                1::BIGINT AS target,
                0.8::DOUBLE AS score,
                1::BIGINT AS predicted,
                0.5::DOUBLE AS threshold
            """
        )

    first = case_data_fingerprint(
        config_path,
        run_id="run_a",
        result_id="reused",
        split="holdout",
    )
    with duckdb.connect(str(db_path)) as con:
        con.execute(
            """
            UPDATE model_predictions
            SET score = 0.2, predicted = 0
            WHERE result_id = 'reused'
            """
        )
    second = case_data_fingerprint(
        config_path,
        run_id="run_a",
        result_id="reused",
        split="holdout",
    )

    assert first.value != second.value
    assert first.payload["predictions"] != second.payload["predictions"]


def _case_frame(*, background_rows: int) -> pd.DataFrame:
    total = background_rows + 3
    source_id = np.arange(1, total + 1)
    target = np.zeros(total, dtype=int)
    target[-2:] = 1
    predicted = np.zeros(total, dtype=int)
    predicted[-3] = 1
    predicted[-2] = 0
    predicted[-1] = 1
    rank = np.arange(3, total + 3)
    rank[-3:] = [2, total - 1, 1]
    return pd.DataFrame(
        {
            "source_id": source_id,
            "target": target,
            "predicted": predicted,
            "threshold": np.full(total, 0.5),
            "rank": rank,
            "score": np.linspace(0.01, 0.99, total),
            "object_name": [f"source {value}" for value in source_id],
            "BP_RP": np.linspace(0.5, 5.5, total),
            "G": np.linspace(9.0, 20.0, total),
            "ra": np.linspace(0.0, 359.0, total),
            "dec": np.linspace(-70.0, 70.0, total),
            "parallax": np.full(total, 0.5),
            "parallax_over_error": np.full(total, 5.0),
        }
    )
