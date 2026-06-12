from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from wr_detector.modeling.second_layer import broad_wr_subtype, train_second_layer_validators


def test_broad_wr_subtype_parser_handles_main_groups():
    assert broad_wr_subtype("WN6h") == "WN"
    assert broad_wr_subtype("WC9d") == "WC"
    assert broad_wr_subtype("WO2") == "WO"
    assert broad_wr_subtype("WN4/WC") == "WN/WC"
    assert broad_wr_subtype("") == "other_or_missing"


def test_train_second_layer_validators_writes_results_and_model(tmp_path):
    config = _write_second_layer_fixture(tmp_path)

    results = train_second_layer_validators(
        config,
        variants=["strict_photometry"],
        feature_sets=["current_colors"],
        methods=["gaussian_mixture"],
        subtypes=["WN"],
        negative_ratios=[10],
        run_id="second_layer_test",
    )

    assert len(results) == 1
    row = results.iloc[0]
    assert row["status"] == "accepted"
    assert row["subtype"] == "WN"
    assert row["holdout_positive_count"] == 2
    assert pd.notna(row["holdout_average_precision"])
    assert Path(row["model_path"]).exists()
    assert (tmp_path / "reports" / "second_layer_validation_results.csv").exists()
    assert (
        tmp_path
        / "reports"
        / "modeling"
        / "second_layer"
        / "runs"
        / "second_layer_test"
        / "second_layer_validation_results.csv"
    ).exists()


def _write_second_layer_fixture(tmp_path: Path) -> Path:
    modeling_dir = tmp_path / "modeling"
    reports_dir = tmp_path / "reports"
    modeling_dir.mkdir()
    reports_dir.mkdir()

    rows = []
    source_id = 1
    for split, n in [("train", 6), ("holdout", 2)]:
        for _ in range(n):
            rows.append(_row(source_id, target=1, split=split, center=1.0))
            source_id += 1
    for split, n in [("train", 8), ("holdout", 8), ("threshold_calibration", 10)]:
        for _ in range(n):
            rows.append(_row(source_id, target=0, split=split, center=-1.0))
            source_id += 1
    pd.DataFrame(rows).to_parquet(modeling_dir / "strict_photometry_reduced.parquet", index=False)

    reference_db = tmp_path / "wr_reference.duckdb"
    with duckdb.connect(str(reference_db)) as con:
        con.execute('CREATE TABLE wr_reference (source_id BIGINT, wr_id VARCHAR, "Spectral Type" VARCHAR)')
        con.executemany(
            'INSERT INTO wr_reference VALUES (?, ?, ?)',
            [(idx, f"WR{idx}", "WN6h") for idx in range(1, 9)],
        )

    models_config = tmp_path / "models.yaml"
    models_config.write_text(
        "\n".join(
            [
                "dataset_variants: [strict_photometry]",
                "negative_reduction: {negative_to_wr_ratio: 10}",
                "outputs:",
                f"  modeling_data_dir: {modeling_dir.as_posix()}",
                '  reduced_dataset_template: "{variant}_reduced.parquet"',
                f"  training_history_db: {(tmp_path / 'history.duckdb').as_posix()}",
            ]
        ),
        encoding="utf-8",
    )
    second_layer_config = tmp_path / "second_layer.yaml"
    second_layer_config.write_text(
        "\n".join(
            [
                f"models_config: {models_config.as_posix()}",
                f"reference_db: {reference_db.as_posix()}",
                "dataset_variants: [strict_photometry]",
                "negative_ratios: [10]",
                "subtypes: [WN]",
                "target_positive_recall: 0.9",
                "min_train_positives: 3",
                "min_features: 4",
                "feature_sets:",
                "  current_colors: [BP_RP, J_K, W1_W2, parallax]",
                "methods:",
                "  gaussian_mixture: {n_components: 1, covariance_type: full, reg_covar: 0.0001, random_state: 42}",
                "outputs:",
                f"  run_dir_template: {(reports_dir / 'modeling' / 'second_layer' / 'runs' / '{run_id}').as_posix()}",
                f"  latest_results: {(reports_dir / 'second_layer_validation_results.csv').as_posix()}",
            ]
        ),
        encoding="utf-8",
    )
    return second_layer_config


def _row(source_id: int, *, target: int, split: str, center: float) -> dict[str, object]:
    return {
        "source_id": source_id,
        "target": target,
        "modeling_split": split,
        "BP_RP": center + source_id / 1000,
        "J_K": center + 0.1 + source_id / 1000,
        "W1_W2": center + 0.2 + source_id / 1000,
        "parallax": center + 0.3 + source_id / 1000,
    }
