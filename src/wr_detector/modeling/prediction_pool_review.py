"""Read-only queries for prediction-pool status and candidate review.

This module is the application-facing boundary for the completed exact-union
pool, mass-scoring run and persisted candidate-review database. Streamlit pages
must consume these functions instead of embedding SQL or recomputing rankings.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterable

import duckdb
import numpy as np
import pandas as pd

from wr_detector.config import load_yaml, resolve_path
from wr_detector.modeling.cases import add_case_spatial_coordinates


FOLLOWUP_EXCLUSIONS = {"known_wr", "catalogued_non_wr"}


@dataclass(frozen=True)
class PredictionPoolReviewPaths:
    candidate_config: Path
    candidate_db: Path
    scoring_config: Path
    scoring_db: Path
    pool_config: Path
    pool_db: Path


def prediction_pool_review_paths(
    config_path: str | Path = "configs/prediction_pool_candidates.yaml",
) -> PredictionPoolReviewPaths:
    candidate_config = resolve_path(config_path)
    candidate = load_yaml(candidate_config)
    scoring_config = resolve_path(candidate["scoring_config"])
    scoring = load_yaml(scoring_config)
    pool_config = resolve_path(scoring["pool_config"])
    pool = load_yaml(pool_config)
    return PredictionPoolReviewPaths(
        candidate_config=candidate_config,
        candidate_db=resolve_path(candidate["outputs"]["database"]),
        scoring_config=scoring_config,
        scoring_db=resolve_path(scoring["output_db"]),
        pool_config=pool_config,
        pool_db=resolve_path(pool["output_db"]),
    )


def prediction_pool_review_revision(
    config_path: str | Path = "configs/prediction_pool_candidates.yaml",
) -> str:
    paths = prediction_pool_review_paths(config_path)
    parts: list[str] = []
    for path in [paths.candidate_db, paths.scoring_db, paths.pool_db]:
        if path.exists():
            stat = path.stat()
            parts.append(f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}")
        else:
            parts.append(f"{path.resolve()}:missing")
    return sha256("|".join(parts).encode("utf-8")).hexdigest()


def review_availability(
    config_path: str | Path = "configs/prediction_pool_candidates.yaml",
) -> dict[str, object]:
    paths = prediction_pool_review_paths(config_path)
    return {
        "candidate_db": str(paths.candidate_db),
        "candidate_db_exists": paths.candidate_db.exists(),
        "scoring_db": str(paths.scoring_db),
        "scoring_db_exists": paths.scoring_db.exists(),
        "pool_db": str(paths.pool_db),
        "pool_db_exists": paths.pool_db.exists(),
        "ready": all(
            path.exists()
            for path in [paths.candidate_db, paths.scoring_db, paths.pool_db]
        ),
    }


def load_candidate_review_runs(
    config_path: str | Path = "configs/prediction_pool_candidates.yaml",
) -> pd.DataFrame:
    paths = prediction_pool_review_paths(config_path)
    if not paths.candidate_db.exists():
        return pd.DataFrame()
    with _connect(paths.candidate_db) as con:
        if not _table_exists(con, "candidate_review_runs"):
            return pd.DataFrame()
        return con.execute(
            """
            SELECT *
            FROM candidate_review_runs
            ORDER BY created_at DESC, review_run_id
            """
        ).fetchdf()


def load_pool_status(
    config_path: str | Path = "configs/prediction_pool_candidates.yaml",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = prediction_pool_review_paths(config_path)
    if not paths.pool_db.exists():
        return pd.DataFrame(), pd.DataFrame()
    with _connect(paths.pool_db) as con:
        if not {
            "exact_union_builds",
            "exact_union_tiles",
        }.issubset(_tables(con)):
            return pd.DataFrame(), pd.DataFrame()
        summary = con.execute(
            """
            SELECT
                b.pool_build_id,
                b.status,
                b.envelope_sha256,
                b.bitmask_schema_version,
                b.bitmask_schema_sha256,
                b.updated_at,
                COUNT(*) AS registered_tiles,
                COUNT(*) FILTER (WHERE t.status='completed') AS completed_tiles,
                COUNT(*) FILTER (WHERE t.status<>'completed') AS incomplete_tiles,
                SUM(t.acquired_pre_locus) AS acquired_rows,
                SUM(t.accepted_union) AS accepted_union_rows,
                SUM(t.known_excluded) AS known_excluded_rows,
                SUM(t.written) AS operational_rows,
                SUM(t.parquet_bytes) AS parquet_bytes
            FROM exact_union_builds b
            JOIN exact_union_tiles t USING (pool_build_id)
            GROUP BY ALL
            ORDER BY b.updated_at DESC
            """
        ).fetchdf()
        tiles = con.execute(
            """
            SELECT pool_build_id, tile_id, ra_min, ra_max, dec_min, dec_max,
                   status, acquired_pre_locus, accepted_union,
                   known_excluded, written, parquet_bytes, updated_at
            FROM exact_union_tiles
            ORDER BY dec_min, ra_min, tile_id
            """
        ).fetchdf()
    if not tiles.empty:
        tiles["ra_center"] = (
            pd.to_numeric(tiles["ra_min"], errors="coerce")
            + pd.to_numeric(tiles["ra_max"], errors="coerce")
        ) / 2.0
        tiles["dec_center"] = (
            pd.to_numeric(tiles["dec_min"], errors="coerce")
            + pd.to_numeric(tiles["dec_max"], errors="coerce")
        ) / 2.0
        acquired = pd.to_numeric(tiles["acquired_pre_locus"], errors="coerce")
        written = pd.to_numeric(tiles["written"], errors="coerce")
        tiles["retention_fraction"] = written.div(acquired.replace(0, np.nan))
    return summary, tiles


def load_scoring_status(
    config_path: str | Path = "configs/prediction_pool_candidates.yaml",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = prediction_pool_review_paths(config_path)
    if not paths.scoring_db.exists():
        return pd.DataFrame(), pd.DataFrame()
    with _connect(paths.scoring_db) as con:
        if not {
            "prediction_scoring_runs",
            "prediction_scoring_tiles",
        }.issubset(_tables(con)):
            return pd.DataFrame(), pd.DataFrame()
        runs = con.execute(
            """
            SELECT scoring_run_id, model_run_id, pool_build_id, status,
                   model_selection_sha256, config_sha256, created_at, updated_at
            FROM prediction_scoring_runs
            ORDER BY updated_at DESC, scoring_run_id
            """
        ).fetchdf()
        models = con.execute(
            """
            SELECT
                t.scoring_run_id,
                t.result_id,
                ANY_VALUE(t.dataset_variant) AS dataset_variant,
                COUNT(*) AS work_units,
                COUNT(*) FILTER (WHERE t.status='completed') AS completed_units,
                COUNT(*) FILTER (WHERE t.status<>'completed') AS incomplete_units,
                SUM(t.rows_variant_compatible) AS eligible_rows,
                SUM(t.rows_missing_features) AS missing_feature_rows,
                SUM(t.rows_scored) AS scored_rows,
                SUM(t.rows_predicted_positive) AS predicted_positive_rows,
                SUM(t.output_bytes) AS output_bytes
            FROM prediction_scoring_tiles t
            GROUP BY t.scoring_run_id, t.result_id
            ORDER BY t.scoring_run_id, t.result_id
            """
        ).fetchdf()
    if not models.empty:
        if paths.candidate_db.exists():
            with _connect(paths.candidate_db) as candidate_con:
                if _table_exists(candidate_con, "candidate_models"):
                    metadata = candidate_con.execute(
                        """
                        SELECT result_id, role, model, sampler, feature_set
                        FROM candidate_models
                        """
                    ).fetchdf()
                    models = models.merge(
                        metadata,
                        on="result_id",
                        how="left",
                        validate="many_to_one",
                    )
        if "role" in models.columns:
            models["model_label"] = models["role"].fillna(
                models["result_id"].astype(str).str.slice(0, 8)
            )
        else:
            models["model_label"] = models["result_id"].astype(str).str.slice(0, 8)
        eligible = pd.to_numeric(models["eligible_rows"], errors="coerce")
        scored = pd.to_numeric(models["scored_rows"], errors="coerce")
        models["scored_fraction"] = scored.div(eligible.replace(0, np.nan))
        models["missing_feature_fraction"] = pd.to_numeric(
            models["missing_feature_rows"], errors="coerce"
        ).div(eligible.replace(0, np.nan))
    return runs, models


def load_candidate_models(
    config_path: str | Path = "configs/prediction_pool_candidates.yaml",
) -> pd.DataFrame:
    paths = prediction_pool_review_paths(config_path)
    if not paths.candidate_db.exists():
        return pd.DataFrame()
    with _connect(paths.candidate_db) as con:
        if not _table_exists(con, "candidate_models"):
            return pd.DataFrame()
        return con.execute(
            """
            SELECT *
            FROM candidate_models
            ORDER BY holdout_average_precision DESC NULLS LAST, result_id
            """
        ).fetchdf()


def load_candidate_summary(
    config_path: str | Path = "configs/prediction_pool_candidates.yaml",
) -> dict[str, object]:
    paths = prediction_pool_review_paths(config_path)
    if not paths.candidate_db.exists():
        return {}
    with _connect(paths.candidate_db) as con:
        required = {
            "candidate_review",
            "candidate_consensus",
            "candidate_model_ranks",
            "candidate_recommendations",
        }
        if not required.issubset(_tables(con)):
            return {}
        queried_at_expression = (
            "MAX(queried_at)"
            if "queried_at" in _columns(con, "candidate_review")
            else "NULL"
        )
        row = con.execute(
            f"""
            SELECT
                (SELECT COUNT(*) FROM candidate_model_ranks) AS model_rank_rows,
                (SELECT COUNT(*) FROM candidate_consensus) AS consensus_rows,
                (SELECT COUNT(*) FROM candidate_review) AS review_rows,
                (SELECT COUNT(*) FROM candidate_recommendations)
                    AS recommendation_rows,
                COUNT(*) FILTER (WHERE match_found) AS simbad_matches,
                COUNT(*) FILTER (WHERE match_method='exact_gaia_dr3_id')
                    AS exact_matches,
                COUNT(*) FILTER (WHERE match_method='position_unique')
                    AS positional_unique_matches,
                COUNT(*) FILTER (WHERE match_method='position_ambiguous')
                    AS positional_ambiguous_matches,
                COUNT(*) FILTER (WHERE review_disposition='no_exact_match')
                    AS no_exact_matches,
                COUNT(*) FILTER (WHERE review_disposition='known_wr')
                    AS known_wr,
                COUNT(*) FILTER (
                    WHERE review_disposition='catalogued_non_wr'
                ) AS catalogued_non_wr,
                {queried_at_expression} AS simbad_queried_at
            FROM candidate_review
            """
        ).fetchone()
        columns = [
            "model_rank_rows",
            "consensus_rows",
            "review_rows",
            "recommendation_rows",
            "simbad_matches",
            "exact_matches",
            "positional_unique_matches",
            "positional_ambiguous_matches",
            "no_exact_matches",
            "known_wr",
            "catalogued_non_wr",
            "simbad_queried_at",
        ]
    return dict(zip(columns, row, strict=True))


def load_disposition_counts(
    config_path: str | Path = "configs/prediction_pool_candidates.yaml",
) -> pd.DataFrame:
    paths = prediction_pool_review_paths(config_path)
    if not paths.candidate_db.exists():
        return pd.DataFrame(columns=["review_disposition", "rows"])
    with _connect(paths.candidate_db) as con:
        if not _table_exists(con, "candidate_review"):
            return pd.DataFrame(columns=["review_disposition", "rows"])
        return con.execute(
            """
            SELECT review_disposition, COUNT(*) AS rows
            FROM candidate_review
            GROUP BY review_disposition
            ORDER BY rows DESC, review_disposition
            """
        ).fetchdf()


def load_candidate_rankings(
    config_path: str | Path = "configs/prediction_pool_candidates.yaml",
    *,
    view: str = "consensus",
    result_id: str | None = None,
    limit: int = 500,
    dispositions: Iterable[str] | None = None,
    min_model_support: int = 1,
    followup_only: bool = False,
    require_halpha: bool = False,
) -> pd.DataFrame:
    paths = prediction_pool_review_paths(config_path)
    if not paths.candidate_db.exists():
        return pd.DataFrame()
    limit = max(1, min(int(limit), 10_000))
    dispositions = [str(value) for value in (dispositions or [])]
    with _connect(paths.candidate_db) as con:
        tables = _tables(con)
        if view == "specialist":
            if result_id is None:
                raise ValueError("Specialist rankings require result_id.")
            if "candidate_model_ranks" not in tables:
                return pd.DataFrame()
            return con.execute(
                """
                SELECT
                    r.model_rank AS rank,
                    r.source_id,
                    r.score,
                    r.predicted,
                    r.threshold,
                    r.result_id,
                    r.dataset_variant,
                    r.model,
                    r.role,
                    c.consensus_rank,
                    c.rrf_score,
                    c.model_support,
                    c.variant_support,
                    c.estimator_support,
                    c.best_model_rank,
                    c.gaia_designation,
                    c.G, c.BP_RP, c.J_K, c.W1_W2,
                    c.parallax_over_error, c.ruwe, c.halpha_ew,
                    c.espels_class, c.review_disposition,
                    c.simbad_main_id, c.review_comment
                FROM candidate_model_ranks r
                LEFT JOIN candidate_review c USING (source_id)
                WHERE r.result_id=?
                ORDER BY r.model_rank, r.source_id
                LIMIT ?
                """,
                [str(result_id), limit],
            ).fetchdf()
        if view != "consensus":
            raise ValueError(f"Unknown candidate ranking view: {view}.")
        if "candidate_review" not in tables:
            return pd.DataFrame()
        clauses = ["model_support >= ?"]
        params: list[object] = [int(min_model_support)]
        if dispositions:
            placeholders = ", ".join("?" for _ in dispositions)
            clauses.append(f"review_disposition IN ({placeholders})")
            params.extend(dispositions)
        if followup_only:
            clauses.append(
                "review_disposition NOT IN ('known_wr', 'catalogued_non_wr')"
            )
        if require_halpha:
            clauses.append("halpha_ew IS NOT NULL")
        params.append(limit)
        where = " AND ".join(clauses)
        return con.execute(
            f"""
            SELECT consensus_rank AS rank, *
            FROM candidate_review
            WHERE {where}
            ORDER BY consensus_rank, source_id
            LIMIT ?
            """,
            params,
        ).fetchdf()


def load_candidate_detail(
    config_path: str | Path,
    *,
    source_id: int,
) -> pd.DataFrame:
    paths = prediction_pool_review_paths(config_path)
    if not paths.candidate_db.exists():
        return pd.DataFrame()
    with _connect(paths.candidate_db) as con:
        if not _table_exists(con, "candidate_review"):
            return pd.DataFrame()
        return con.execute(
            """
            SELECT *
            FROM candidate_review
            WHERE source_id=?
            """,
            [int(source_id)],
        ).fetchdf()


def load_candidate_model_evidence(
    config_path: str | Path,
    *,
    source_id: int,
) -> pd.DataFrame:
    paths = prediction_pool_review_paths(config_path)
    if not paths.candidate_db.exists():
        return pd.DataFrame()
    with _connect(paths.candidate_db) as con:
        if not {
            "candidate_model_ranks",
            "candidate_models",
        }.issubset(_tables(con)):
            return pd.DataFrame()
        rrf_k = 60.0
        if (
            _table_exists(con, "candidate_review_runs")
            and "rrf_k" in _columns(con, "candidate_review_runs")
        ):
            stored_rrf_k = con.execute(
                """
                SELECT rrf_k
                FROM candidate_review_runs
                WHERE rrf_k IS NOT NULL
                ORDER BY created_at DESC
                LIMIT 1
                """
            ).fetchone()
            if stored_rrf_k:
                rrf_k = float(stored_rrf_k[0])
        return con.execute(
            """
            SELECT
                r.result_id,
                m.role,
                m.dataset_variant,
                m.model,
                m.sampler,
                m.feature_set,
                r.model_rank,
                r.score,
                r.predicted,
                r.threshold,
                1.0 / (? + r.model_rank) AS rrf_contribution
            FROM candidate_model_ranks r
            JOIN candidate_models m USING (result_id)
            WHERE r.source_id=?
            ORDER BY r.model_rank, r.result_id
            """,
            [rrf_k, int(source_id)],
        ).fetchdf()


def load_candidate_jaccard(
    config_path: str | Path = "configs/prediction_pool_candidates.yaml",
    *,
    top_k: int = 500,
) -> pd.DataFrame:
    paths = prediction_pool_review_paths(config_path)
    if not paths.candidate_db.exists():
        return pd.DataFrame()
    top_k = max(1, min(int(top_k), 10_000))
    with _connect(paths.candidate_db) as con:
        if not {
            "candidate_model_ranks",
            "candidate_models",
        }.issubset(_tables(con)):
            return pd.DataFrame()
        ranks = con.execute(
            """
            SELECT r.result_id, r.source_id, m.role
            FROM candidate_model_ranks r
            JOIN candidate_models m USING (result_id)
            WHERE r.model_rank <= ?
            ORDER BY r.result_id, r.model_rank
            """,
            [top_k],
        ).fetchdf()
    groups = {
        result_id: set(group["source_id"].astype("int64"))
        for result_id, group in ranks.groupby("result_id", sort=True)
    }
    roles = (
        ranks[["result_id", "role"]]
        .drop_duplicates("result_id")
        .set_index("result_id")["role"]
        .to_dict()
    )
    rows: list[dict[str, object]] = []
    for left_id, left in groups.items():
        for right_id, right in groups.items():
            union = left | right
            rows.append(
                {
                    "left_result_id": left_id,
                    "right_result_id": right_id,
                    "left_role": roles.get(left_id, left_id),
                    "right_role": roles.get(right_id, right_id),
                    "intersection": len(left & right),
                    "union": len(union),
                    "jaccard": len(left & right) / len(union) if union else np.nan,
                    "top_k": top_k,
                }
            )
    return pd.DataFrame(rows)


def load_candidate_plot_frame(
    config_path: str | Path = "configs/prediction_pool_candidates.yaml",
    *,
    min_parallax_over_error: float = 2.0,
    max_distance_kpc: float = 15.0,
) -> pd.DataFrame:
    frame = load_candidate_rankings(
        config_path,
        view="consensus",
        limit=500,
    )
    if frame.empty:
        return frame
    return add_case_spatial_coordinates(
        frame,
        min_parallax_over_error=min_parallax_over_error,
        max_distance_kpc=max_distance_kpc,
    )


def _connect(path: Path) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(path), read_only=True)


def _tables(con: duckdb.DuckDBPyConnection) -> set[str]:
    return {str(row[0]) for row in con.execute("SHOW TABLES").fetchall()}


def _table_exists(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    return table in _tables(con)


def _columns(con: duckdb.DuckDBPyConnection, table: str) -> set[str]:
    return {
        str(row[0])
        for row in con.execute(f"DESCRIBE {table}").fetchall()
    }
