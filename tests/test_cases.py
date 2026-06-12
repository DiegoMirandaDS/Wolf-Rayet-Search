from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from wr_detector.modeling.cases import (
    case_confusion,
    false_positive_composition,
    load_case_overlap,
    load_case_predictions,
    precision_recall_points,
    roc_points,
    subtype_recovery,
    wr_broad_subtype,
)


def test_wr_broad_subtype_families():
    assert wr_broad_subtype("WN4b") == "WN"
    assert wr_broad_subtype("WC7; WC8") == "WC"
    assert wr_broad_subtype("WO2") == "WO"
    assert wr_broad_subtype("WN/WC4") == "WN/WC"
    assert wr_broad_subtype("O3If*") == "other"
    assert wr_broad_subtype(None) == "unknown"


def test_case_predictions_enriched_with_identity_and_photometry(tmp_path):
    config = _write_databases(tmp_path)

    cases = load_case_predictions(config, run_id="run_a", result_id="r1", split="holdout")

    assert cases["rank"].tolist() == [1, 2, 3, 4]
    top = cases.iloc[0]
    assert top["object_name"] == "WR 1"
    assert top["wr_subtype"] == "WN"
    assert top["BP_RP"] == pytest.approx(1.0)
    assert top["J_K"] == pytest.approx(0.7)
    fp = cases[cases["source_id"].eq(20)].iloc[0]
    assert fp["object_name"] == "EM* Test 1"
    assert fp["simbad_main_type"] == "Em*"
    assert pd.isna(fp["wr_subtype"])
    unknown = cases[cases["source_id"].eq(30)].iloc[0]
    assert unknown["object_name"] == "Gaia DR3 30"


def test_case_predictions_without_reference_dbs(tmp_path):
    config = _write_databases(tmp_path, include_reference_dbs=False)

    cases = load_case_predictions(config, run_id="run_a", result_id="r1", split="holdout")

    assert len(cases) == 4
    assert cases["rank"].tolist() == [1, 2, 3, 4]
    assert "spectral_type" not in cases.columns
    assert cases.iloc[0]["object_name"] == "Gaia DR3 10"


def test_case_aggregations(tmp_path):
    config = _write_databases(tmp_path)
    cases = load_case_predictions(config, run_id="run_a", result_id="r1", split="holdout")

    recovery = subtype_recovery(cases, ks=[1, 2])
    wn = recovery[recovery["wr_subtype"].eq("WN")].iloc[0]
    assert wn["total"] == 1
    assert wn["recovered_at_1"] == 1

    composition = false_positive_composition(cases, top_k=2)
    assert composition.iloc[0]["simbad_main_type"] == "Em*"
    assert composition.iloc[0]["count"] == 1

    confusion = case_confusion(cases)
    assert confusion["true_positive"] == 1
    assert confusion["false_negative"] == 1
    assert confusion["false_positive"] == 1
    assert confusion["true_negative"] == 1


def test_case_overlap_across_models(tmp_path):
    config = _write_databases(tmp_path)

    contaminants = load_case_overlap(config, run_id="run_a", kind="false_positive", top_k=2)
    missed = load_case_overlap(config, run_id="run_a", kind="missed_wr", top_k=2)

    assert contaminants.iloc[0]["source_id"] == 20
    assert contaminants.iloc[0]["n_models"] == 2
    assert contaminants.iloc[0]["n_total_models"] == 2
    assert contaminants.iloc[0]["model_share"] == pytest.approx(1.0)
    assert contaminants.iloc[0]["object_name"] == "EM* Test 1"
    assert missed.iloc[0]["source_id"] == 40
    assert missed.iloc[0]["object_name"] == "WR 2"


def test_ranking_curve_points():
    cases = pd.DataFrame(
        {
            "score": [0.9, 0.8, 0.6, 0.4],
            "target": [1, 0, 1, 0],
        }
    )

    pr = precision_recall_points(cases)
    roc = roc_points(cases)

    assert pr["recall"].tolist() == pytest.approx([0.5, 0.5, 1.0, 1.0])
    assert pr["precision"].tolist() == pytest.approx([1.0, 0.5, 2 / 3, 0.5])
    assert roc["fpr"].tolist() == pytest.approx([0.0, 0.5, 0.5, 1.0])
    assert roc["tpr"].tolist() == pytest.approx([0.5, 0.5, 1.0, 1.0])


def test_ranking_curves_downsample_and_handle_degenerate_input():
    many = pd.DataFrame({"score": [1 - i / 2000 for i in range(2000)], "target": [i % 2 for i in range(2000)]})
    assert len(precision_recall_points(many, max_points=100)) <= 102

    no_positives = pd.DataFrame({"score": [0.5, 0.4], "target": [0, 0]})
    assert precision_recall_points(no_positives).empty
    assert roc_points(no_positives).empty


def test_case_overlap_rejects_unknown_kind(tmp_path):
    config = _write_databases(tmp_path)
    with pytest.raises(ValueError, match="Unsupported overlap kind"):
        load_case_overlap(config, run_id="run_a", kind="bogus")


def _write_databases(tmp_path: Path, *, include_reference_dbs: bool = True) -> Path:
    history_db = tmp_path / "history.duckdb"
    wr_db = tmp_path / "wr_reference.duckdb"
    neg_db = tmp_path / "simbad_negative.duckdb"
    paths_yaml = tmp_path / "paths.yaml"
    config_yaml = tmp_path / "models.yaml"

    paths_lines = []
    if include_reference_dbs:
        paths_lines = [
            f"wr_reference_db: {wr_db.as_posix()}",
            f"simbad_negative_db: {neg_db.as_posix()}",
        ]
    paths_yaml.write_text("\n".join(paths_lines) or "{}", encoding="utf-8")
    config_yaml.write_text(
        "\n".join(
            [
                f"paths_config: {paths_yaml.as_posix()}",
                "outputs:",
                f"  training_history_db: {history_db.as_posix()}",
            ]
        ),
        encoding="utf-8",
    )

    # source 10: WR recovered, 20: contaminant, 30: unlabeled negative, 40: missed WR.
    predictions = pd.DataFrame(
        [
            _prediction("r1", 10, target=1, score=0.95, predicted=1),
            _prediction("r1", 20, target=0, score=0.90, predicted=1),
            _prediction("r1", 30, target=0, score=0.30, predicted=0),
            _prediction("r1", 40, target=1, score=0.20, predicted=0),
            _prediction("r2", 10, target=1, score=0.99, predicted=1),
            _prediction("r2", 20, target=0, score=0.91, predicted=1),
            _prediction("r2", 30, target=0, score=0.10, predicted=0),
            _prediction("r2", 40, target=1, score=0.05, predicted=0),
        ]
    )
    with duckdb.connect(str(history_db)) as con:
        con.register("predictions", predictions)
        con.execute("CREATE TABLE model_predictions AS SELECT * FROM predictions")

    if include_reference_dbs:
        with duckdb.connect(str(wr_db)) as con:
            con.execute(
                'CREATE TABLE wr_reference ("WR#" VARCHAR, "Spectral Type" VARCHAR, source_id BIGINT)'
            )
            con.execute("INSERT INTO wr_reference VALUES ('1', 'WN4b', 10), ('2', 'WC8', 40)")
            _create_photometry_tables(con, [(10, 12.0), (40, 14.0)])
        with duckdb.connect(str(neg_db)) as con:
            con.execute(
                "CREATE TABLE simbad_negative_sources (simbad_main_id VARCHAR, simbad_main_type VARCHAR, simbad_sp_type VARCHAR, source_id BIGINT)"
            )
            con.execute("INSERT INTO simbad_negative_sources VALUES ('EM* Test 1', 'Em*', 'OB', 20)")
            _create_photometry_tables(con, [(20, 13.0)])
    return config_yaml


def _prediction(result_id: str, source_id: int, *, target: int, score: float, predicted: int) -> dict[str, object]:
    return {
        "run_id": "run_a",
        "result_id": result_id,
        "split": "holdout",
        "row_id": source_id,
        "source_id": source_id,
        "target": target,
        "score": score,
        "predicted": predicted,
        "threshold": 0.5,
    }


def _create_photometry_tables(con: duckdb.DuckDBPyConnection, rows: list[tuple[int, float]]) -> None:
    con.execute(
        "CREATE TABLE gaia_sources (source_id BIGINT, ra DOUBLE, dec DOUBLE, G DOUBLE, BP DOUBLE, RP DOUBLE,"
        " parallax DOUBLE, parallax_over_error DOUBLE, ruwe DOUBLE)"
    )
    con.execute(
        "CREATE TABLE twomass_matches (source_id BIGINT, J DOUBLE, H DOUBLE, Ks DOUBLE, tmass_quality VARCHAR)"
    )
    con.execute("CREATE TABLE wise_matches (source_id BIGINT, W1 DOUBLE, W2 DOUBLE, wise_quality VARCHAR)")
    for source_id, g_mag in rows:
        con.execute(
            "INSERT INTO gaia_sources VALUES (?, 10.0, -60.0, ?, ?, ?, 0.5, 3.0, 1.0)",
            [source_id, g_mag, g_mag + 0.5, g_mag - 0.5],
        )
        con.execute(
            "INSERT INTO twomass_matches VALUES (?, ?, ?, ?, 'AAA')",
            [source_id, g_mag - 1.0, g_mag - 1.5, g_mag - 1.7],
        )
        con.execute("INSERT INTO wise_matches VALUES (?, ?, ?, 'AA')", [source_id, g_mag - 2.0, g_mag - 2.2])
