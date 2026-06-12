from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from wr_detector.cli import EXPLORER_THEMES, explorer_theme_options, streamlit_model_explorer_command
from wr_detector.modeling.history import sync_second_layer_history
from wr_detector.modeling.layers import ValidationLayer, layer_history_db_path, list_layer_runs, load_layer_results
from wr_detector.modeling.second_layer import apply_color_locus_keep


def _layer(tmp_path: Path) -> ValidationLayer:
    models_yaml = tmp_path / "models.yaml"
    models_yaml.write_text(
        "\n".join(
            [
                "outputs:",
                f"  training_history_db: {(tmp_path / 'history.duckdb').as_posix()}",
            ]
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "second_layer.yaml"
    runs_root = tmp_path / "runs"
    config_path.write_text(
        "\n".join(
            [
                f"models_config: {models_yaml.as_posix()}",
                "outputs:",
                f"  run_dir_template: {runs_root.as_posix()}/{{run_id}}",
            ]
        ),
        encoding="utf-8",
    )
    return ValidationLayer(
        key="second_layer",
        title="Second layer",
        config_path=str(config_path),
        results_filename="second_layer_validation_results.csv",
        train_command="wr-detector train-second-layer",
    )


def _result_frame(run_id: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "run_id": run_id,
                "dataset_variant": "strict_poe_3",
                "method": "gaussian_mixture",
                "subtype": "WN",
                "status": "accepted",
                "holdout_positive_retention": 0.78,
                "threshold_calibration_negative_pass_rate": 0.08,
                "dataset_path": "data/processed/modeling/strict_poe_3_reduced.parquet",
                "dataset_sha256": "abc123",
            }
        ]
    )


def test_list_layer_runs_empty_when_no_runs(tmp_path):
    layer = _layer(tmp_path)
    runs = list_layer_runs(layer)
    assert runs.empty
    assert load_layer_results(layer, "missing_run").empty


def test_csv_only_runs_are_discovered(tmp_path):
    layer = _layer(tmp_path)
    run_dir = tmp_path / "runs" / "run_csv"
    run_dir.mkdir(parents=True)
    _result_frame("run_csv").to_csv(run_dir / layer.results_filename, index=False)
    (tmp_path / "runs" / "not_a_run").mkdir()

    runs = list_layer_runs(layer)
    loaded = load_layer_results(layer, "run_csv")

    assert runs["run_id"].tolist() == ["run_csv"]
    assert runs.iloc[0]["source"] == "csv"
    assert len(loaded) == 1


def test_synced_runs_come_from_history_db(tmp_path):
    layer = _layer(tmp_path)
    models_config = {"outputs": {"training_history_db": (tmp_path / "history.duckdb").as_posix()}}
    report = sync_second_layer_history(
        models_config,
        _result_frame("run_db"),
        run_id="run_db",
        source_csv="reports/x.csv",
        config_path=layer.config_path,
    )

    runs = list_layer_runs(layer)
    loaded = load_layer_results(layer, "run_db")

    assert report["rows"] == 1
    assert runs["run_id"].tolist() == ["run_db"]
    assert runs.iloc[0]["source"] == "history_db"
    assert loaded.iloc[0]["dataset_sha256"] == "abc123"
    with duckdb.connect(str(layer_history_db_path(layer)), read_only=True) as con:
        assert con.execute("SELECT COUNT(*) FROM second_layer_runs").fetchone()[0] == 1


def test_sync_replace_run_overwrites(tmp_path):
    layer = _layer(tmp_path)
    models_config = {"outputs": {"training_history_db": (tmp_path / "history.duckdb").as_posix()}}
    sync_second_layer_history(models_config, _result_frame("run_a"), run_id="run_a")
    try:
        sync_second_layer_history(models_config, _result_frame("run_a"), run_id="run_a")
        raise AssertionError("expected duplicate-run failure")
    except ValueError:
        pass
    sync_second_layer_history(models_config, _result_frame("run_a"), run_id="run_a", replace_run=True)
    assert len(load_layer_results(layer, "run_a")) == 1


def test_apply_color_locus_keep_filters_outliers():
    dataset = pd.DataFrame({"color_locus_keep": [True, False, True], "value": [1, 2, 3]})
    filtered, excluded = apply_color_locus_keep(dataset)
    assert excluded == 1
    assert filtered["value"].tolist() == [1, 3]

    no_column = pd.DataFrame({"value": [1]})
    unchanged, excluded = apply_color_locus_keep(no_column)
    assert excluded == 0
    assert len(unchanged) == 1


def test_explorer_theme_options():
    for theme in EXPLORER_THEMES:
        options = explorer_theme_options(theme)
        assert options[:2] == ["--theme.base", "dark"]
        assert "--theme.primaryColor" in options

    command = streamlit_model_explorer_command(Path("configs/models.yaml"), 8501, "nebula")
    assert "--theme.backgroundColor" in command
    assert command[command.index("--theme.primaryColor") + 1] == "#9d7bff"
