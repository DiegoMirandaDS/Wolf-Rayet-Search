from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


def available_training_runs(root: str | Path = ".") -> pd.DataFrame:
    db_path = Path(root) / "data/databases/training_history.duckdb"
    if not db_path.exists():
        return pd.DataFrame()
    with duckdb.connect(str(db_path), read_only=True) as con:
        tables = {row[0] for row in con.execute("SELECT table_name FROM information_schema.tables").fetchall()}
        if "training_runs" not in tables:
            return pd.DataFrame()
        if "model_results" not in tables:
            return con.execute("SELECT * FROM training_runs ORDER BY imported_at DESC").fetchdf()
        model_columns = set(con.execute("DESCRIBE model_results").fetchdf()["column_name"])
        wr_at_50 = "max(m.holdout_wr_at_50)" if "holdout_wr_at_50" in model_columns else "NULL"
        wr_at_100 = "max(m.holdout_wr_at_100)" if "holdout_wr_at_100" in model_columns else "NULL"
        wr_at_500 = "max(m.holdout_wr_at_500)" if "holdout_wr_at_500" in model_columns else "NULL"
        average_precision = "max(m.holdout_average_precision)" if "holdout_average_precision" in model_columns else "NULL"
        recall_fpr = "max(m.holdout_recall_at_fpr_0p005)" if "holdout_recall_at_fpr_0p005" in model_columns else "NULL"
        return con.execute(
            f"""
            SELECT
                r.run_id,
                r.run_name,
                r.imported_at,
                r.notes,
                r.row_count,
                {wr_at_50} AS best_holdout_wr_at_50,
                {wr_at_100} AS best_holdout_wr_at_100,
                {wr_at_500} AS best_holdout_wr_at_500,
                {average_precision} AS best_holdout_average_precision,
                {recall_fpr} AS best_holdout_recall_at_fpr_0p005
            FROM training_runs r
            LEFT JOIN model_results m USING (run_id)
            GROUP BY r.run_id, r.run_name, r.imported_at, r.notes, r.row_count
            ORDER BY r.imported_at DESC
            """
        ).fetchdf()


def load_training_results_for_notebook(root: str | Path, run_id: str) -> pd.DataFrame:
    if not run_id:
        raise ValueError("Set TRAINING_RUN_ID to one of the available run_id values.")
    db_path = Path(root) / "data/databases/training_history.duckdb"
    if not db_path.exists():
        raise FileNotFoundError(f"Training history DB not found: {db_path}")
    with duckdb.connect(str(db_path), read_only=True) as con:
        tables = {row[0] for row in con.execute("SELECT table_name FROM information_schema.tables").fetchall()}
        if "model_results" not in tables:
            raise ValueError(f"model_results table not found in {db_path}")
        results = con.execute("SELECT * FROM model_results WHERE run_id = ?", [run_id]).fetchdf()
    if results.empty:
        raise ValueError(f"No model results found for TRAINING_RUN_ID={run_id!r}")
    return results
