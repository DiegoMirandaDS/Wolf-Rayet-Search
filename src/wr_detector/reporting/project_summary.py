"""Generate the concise technical handoff report and its public figures.

This module is intentionally read-only with respect to scientific inputs. It
queries the existing DuckDB, Parquet and CSV artifacts, writes a dated evidence
snapshot and renders the static figures used by the editable LaTeX handoff.
The older ReportLab rendering remains available as an optional preview.

Run from the repository root:

    python -m wr_detector.reporting.project_summary
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from astropy.coordinates import SkyCoord
import astropy.units as u
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve

from wr_detector.features import fit_log_color_locus, inverse_transform_color_values
from wr_detector.modeling.explorer import load_run_results, rank_models
from wr_detector.pipelines.prediction_pool import make_sky_tiles
from wr_detector.pipelines.prediction_pool_exact_union import (
    ensure_exact_union_builds_status,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "public"
DEFAULT_RUN_ID = "run_v3_main"
LEADING_RESULT_ID = "9e08cb4f7517ab76d5b152e7a7dc53e02c826321"
AP_LEADER_RESULT_ID = "49957d2295543512d384313f5c352a619fac7be8"
STABLE_RESULT_ID = "2efb4b2059dac012e52b9ff3f11e17fe4995dc66"
HGB_RESULT_ID = "9275e7d7da706b3ada8ea1f8c05dae6b833417d1"

INK = "#172033"
MUTED = "#627083"
GRID = "#dfe5ec"
BLUE = "#3568a8"
BLUE_LIGHT = "#a9c4e4"
ORANGE = "#e7842a"
PINK = "#c85686"
GOLD = "#caa232"
OLIVE = "#748b4a"
PALE = "#f4f7fa"
NEGATIVE = "#8fa4b8"


@dataclass(frozen=True)
class Evidence:
    snapshot_date: str
    catalogue: dict[str, Any]
    negative: dict[str, Any]
    locus: pd.DataFrame
    split: pd.DataFrame
    negative_types: pd.DataFrame
    model_results: pd.DataFrame
    predictions: pd.DataFrame
    feature_importance: pd.DataFrame
    pool_audit: dict[str, Any]
    pool_status: dict[str, Any]
    reference_relaxed: pd.DataFrame
    negative_relaxed: pd.DataFrame
    reference_positions: pd.DataFrame
    negative_positions: pd.DataFrame
    top_candidates: pd.DataFrame
    candidate_context: pd.DataFrame


def _path(relative: str | Path) -> Path:
    return PROJECT_ROOT / Path(relative)


def _read_yaml(relative: str | Path) -> dict[str, Any]:
    return yaml.safe_load(_path(relative).read_text(encoding="utf-8"))


def collect_evidence(run_id: str = DEFAULT_RUN_ID) -> Evidence:
    """Load the bounded evidence needed by the report and public figures."""
    reference_db = _path("data/databases/wr_reference.duckdb")
    negative_db = _path("data/databases/simbad_negative.duckdb")
    history_db = _path("data/databases/training_history.duckdb")

    with duckdb.connect(str(reference_db), read_only=True) as con:
        snapshot = con.execute(
            """
            SELECT version, row_count, fetched_at_utc, sha256, url
            FROM catalog_snapshots
            ORDER BY fetched_at_utc DESC
            LIMIT 1
            """
        ).fetchone()
        catalogue = {
            "version": str(snapshot[0]),
            "catalogue_rows": int(snapshot[1]),
            "fetched_at_utc": str(snapshot[2]),
            "sha256": str(snapshot[3]),
            "url": str(snapshot[4]),
            "gaia_alias_rows": int(con.execute("SELECT COUNT(*) FROM wr_reference").fetchone()[0]),
            "twomass_rows": int(con.execute("SELECT COUNT(*) FROM twomass_matches").fetchone()[0]),
            "wise_rows": int(con.execute("SELECT COUNT(*) FROM wise_matches").fetchone()[0]),
        }
        reference_positions = con.execute(
            """
            SELECT
                CAST("Galactic Longitude (deg)" AS DOUBLE) AS galactic_l,
                CAST("Galactic Latitude (deg)" AS DOUBLE) AS galactic_b,
                "Spectral Type" AS spectral_type
            FROM wr_reference
            WHERE "Galactic Longitude (deg)" IS NOT NULL
              AND "Galactic Latitude (deg)" IS NOT NULL
            """
        ).fetchdf()

    with duckdb.connect(str(negative_db), read_only=True) as con:
        negative = {
            "raw_rows": int(con.execute("SELECT COUNT(*) FROM simbad_negative_raw").fetchone()[0]),
            "enriched_rows": int(
                con.execute("SELECT COUNT(*) FROM simbad_negative_sources").fetchone()[0]
            ),
            "twomass_rows": int(con.execute("SELECT COUNT(*) FROM twomass_matches").fetchone()[0]),
            "wise_rows": int(con.execute("SELECT COUNT(*) FROM wise_matches").fetchone()[0]),
        }
        negative_types = con.execute(
            """
            SELECT simbad_query_type AS object_group, COUNT(*) AS rows
            FROM simbad_negative_sources
            GROUP BY simbad_query_type
            ORDER BY rows DESC, object_group
            """
        ).fetchdf()
        negative_positions = con.execute(
            """
            SELECT g.ra, g.dec
            FROM simbad_negative_sources s
            JOIN gaia_sources g USING (source_id)
            WHERE g.ra IS NOT NULL AND g.dec IS NOT NULL
            """
        ).fetchdf()

    variants = [
        "strict_photometry",
        "strict_parallax_soft",
        "strict_poe_2",
        "strict_poe_3",
        "relaxed_photometry",
        "relaxed_parallax_soft",
        "relaxed_poe_2",
        "relaxed_poe_3",
    ]
    locus_rows: list[dict[str, Any]] = []
    with duckdb.connect() as con:
        for variant in variants:
            for class_label, root, prefix in [
                ("WR", "reference", "wr_reference"),
                ("negative", "simbad_negative", "simbad_negative"),
            ]:
                plain = _path(f"data/processed/{root}/{prefix}_{variant}.parquet")
                annotated = _path(
                    f"data/processed/{root}/{prefix}_{variant}_color_locus.parquet"
                )
                counts = con.execute(
                    """
                    SELECT
                        COUNT(*) AS rows,
                        COUNT(*) FILTER (
                            WHERE isfinite(G_BP) AND isfinite(G_RP)
                              AND isfinite(BP_RP) AND isfinite(J_H)
                              AND isfinite(J_K) AND isfinite(H_K)
                        ) AS finite_six,
                        COUNT(*) FILTER (
                            WHERE G_BP > 0 AND G_RP > 0 AND BP_RP > 0
                              AND J_H > 0 AND J_K > 0 AND H_K > 0
                        ) AS positive_six
                    FROM read_parquet(?)
                    """,
                    [str(plain)],
                ).fetchone()
                locus_counts = con.execute(
                    """
                    SELECT
                        COUNT(*) FILTER (WHERE color_locus_keep) AS kept,
                        COUNT(*) FILTER (WHERE color_locus_outlier) AS outliers,
                        COUNT(*) FILTER (WHERE color_locus_valid) AS valid
                    FROM read_parquet(?)
                    """,
                    [str(annotated)],
                ).fetchone()
                locus_rows.append(
                    {
                        "variant": variant,
                        "class": class_label,
                        "rows": int(counts[0]),
                        "finite_six": int(counts[1]),
                        "positive_six": int(counts[2]),
                        "locus_keep": int(locus_counts[0]),
                        "outliers": int(locus_counts[1]),
                        "valid": int(locus_counts[2]),
                    }
                )
    locus = pd.DataFrame(locus_rows)
    locus["locus_keep_fraction"] = locus["locus_keep"] / locus["rows"]
    locus["positive_six_fraction"] = locus["positive_six"] / locus["rows"]

    relaxed_reduced = _path("data/processed/modeling/relaxed_photometry_reduced.parquet")
    with duckdb.connect() as con:
        split = con.execute(
            """
            SELECT modeling_split, target, COUNT(*) AS rows
            FROM read_parquet(?)
            GROUP BY modeling_split, target
            ORDER BY modeling_split, target
            """,
            [str(relaxed_reduced)],
        ).fetchdf()
        reference_relaxed = con.execute(
            "SELECT * FROM read_parquet(?)",
            [str(_path(
                "data/processed/reference/"
                "wr_reference_relaxed_photometry_color_locus.parquet"
            ))],
        ).fetchdf()
        negative_relaxed = con.execute(
            """
            SELECT *
            FROM read_parquet(?)
            USING SAMPLE 30000 ROWS (reservoir, 42)
            """,
            [str(_path(
                "data/processed/simbad_negative/"
                "simbad_negative_relaxed_photometry_color_locus.parquet"
            ))],
        ).fetchdf()

    model_results = rank_models(
        load_run_results(_path("configs/models.yaml"), run_id=run_id)
    )
    selected_ids = [
        LEADING_RESULT_ID,
        AP_LEADER_RESULT_ID,
        HGB_RESULT_ID,
        STABLE_RESULT_ID,
    ]
    with duckdb.connect(str(history_db), read_only=True) as con:
        predictions = con.execute(
            """
            SELECT result_id, split, target, score
            FROM model_predictions
            WHERE run_id = ?
              AND result_id IN (?, ?, ?, ?)
              AND split = 'holdout'
            """,
            [run_id, *selected_ids],
        ).fetchdf()
        feature_importance = con.execute(
            """
            SELECT result_id, feature, importance_mean, importance_std, importance_type
            FROM feature_importance
            WHERE run_id = ? AND result_id = ?
            ORDER BY importance_mean DESC
            """,
            [run_id, LEADING_RESULT_ID],
        ).fetchdf()

    audit_dir = _latest_audit_directory()
    headline = pd.read_csv(audit_dir / "headline_metrics.csv").iloc[0].to_dict()
    wr_summary = pd.read_csv(audit_dir / "known_wr_summary.csv").iloc[0].to_dict()
    region = pd.read_csv(audit_dir / "region_summary.csv")
    pool_audit = {
        "run_dir": str(audit_dir.relative_to(PROJECT_ROOT)),
        "acquisition_sources": int(headline["acquisition_sources"]),
        "current_aggregate_keep": int(headline["current_aggregate_keep"]),
        "compatible_exact_union_keep": int(headline["compatible_exact_union_keep"]),
        "compatible_exact_not_aggregate": int(headline["compatible_exact_not_aggregate"]),
        "loss_fraction": float(
            headline["compatible_exact_not_aggregate"]
            / headline["compatible_exact_union_keep"]
        ),
        "regional_max_loss_fraction": float(
            region["exact_variant_not_aggregate_pct_exact_union"].max()
        ),
        "known_wr_controls": int(wr_summary["known_wr_controls"]),
        "known_wr_exact_not_current": int(wr_summary["exact_union_not_current"]),
    }
    pool_status = _prediction_pool_status()
    top_candidates = _load_top_candidates()
    candidate_context = _load_candidate_context()

    return Evidence(
        snapshot_date=datetime.now(ZoneInfo("America/Santiago")).date().isoformat(),
        catalogue=catalogue,
        negative=negative,
        locus=locus,
        split=split,
        negative_types=negative_types,
        model_results=model_results,
        predictions=predictions,
        feature_importance=feature_importance,
        pool_audit=pool_audit,
        pool_status=pool_status,
        reference_relaxed=reference_relaxed,
        negative_relaxed=negative_relaxed,
        reference_positions=reference_positions,
        negative_positions=negative_positions,
        top_candidates=top_candidates,
        candidate_context=candidate_context,
    )


def _load_top_candidates() -> pd.DataFrame:
    review_run_id = (
        _read_yaml("configs/prediction_pool_candidates.yaml")["review"]["review_run_id"]
    )
    path = _path(
        f"reports/analysis/prediction_pool_candidates/{review_run_id}/top_5_candidates.csv"
    )
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_candidate_context() -> pd.DataFrame:
    """The real consensus top 100 supplies context for the five follow-ups."""
    review_run_id = _read_yaml("configs/prediction_pool_candidates.yaml")["review"]["review_run_id"]
    path = _path(
        f"reports/analysis/prediction_pool_candidates/{review_run_id}/top_500_candidates.csv"
    )
    if not path.exists():
        return pd.DataFrame()
    columns = [
        "consensus_rank", "source_id", "galactic_l", "galactic_b",
        "BP_RP", "G", "W1_W2", "rrf_score",
    ]
    context = pd.read_csv(path, usecols=columns)
    return (
        context.sort_values(["consensus_rank", "source_id"], kind="mergesort")
        .drop_duplicates("source_id")
        .head(100)
        .reset_index(drop=True)
    )


def _latest_audit_directory() -> Path:
    root = _path("reports/analysis/prediction_pool_locus_audit")
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "headline_metrics.csv").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No completed exact-union audit found under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _prediction_pool_status() -> dict[str, Any]:
    config = _read_yaml("configs/prediction_pool_exact_union.yaml")
    db_path = _path(config["output_db"])
    total_tiles = len(make_sky_tiles(config))
    if not db_path.exists():
        return {
            "status": "not_started",
            "total_tiles": total_tiles,
            "completed_tiles": 0,
            "running_tiles": 0,
            "acquired_rows": 0,
            "eligible_rows": 0,
            "parquet_bytes": 0,
            "estimated_rows": 154_000_000,
        }
    ensure_exact_union_builds_status(db_path)
    with duckdb.connect(str(db_path), read_only=True) as con:
        tables = set(con.execute("SHOW TABLES").fetchdf()["name"])
        if "exact_union_tiles" not in tables:
            return {
                "status": "initializing",
                "total_tiles": total_tiles,
                "completed_tiles": 0,
                "running_tiles": 0,
                "acquired_rows": 0,
                "eligible_rows": 0,
                "parquet_bytes": 0,
                "estimated_rows": 154_000_000,
            }
        rows = con.execute(
            """
            SELECT
                COUNT(*) FILTER (WHERE status = 'completed') AS completed_tiles,
                COUNT(*) FILTER (WHERE status = 'running') AS running_tiles,
                COUNT(*) FILTER (WHERE status = 'failed') AS failed_tiles,
                COALESCE(SUM(acquired_pre_locus), 0) AS acquired_rows,
                COALESCE(SUM(written), 0) AS eligible_rows,
                COALESCE(SUM(parquet_bytes), 0) AS parquet_bytes
            FROM exact_union_tiles
            """
        ).fetchone()
        build_status = con.execute(
            """
            SELECT status
            FROM exact_union_builds
            ORDER BY updated_at DESC
            LIMIT 1
            """
        ).fetchone()
    return {
        "status": str(build_status[0]) if build_status else "unknown",
        "total_tiles": total_tiles,
        "completed_tiles": int(rows[0]),
        "running_tiles": int(rows[1]),
        "failed_tiles": int(rows[2]),
        "acquired_rows": int(rows[3]),
        "eligible_rows": int(rows[4]),
        "parquet_bytes": int(rows[5]),
        "estimated_rows": 154_000_000,
    }


def _set_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelcolor": INK,
            "axes.edgecolor": "#9aa8b7",
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "grid.alpha": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "legend.frameon": False,
        }
    )


def generate_figures(evidence: Evidence, figures_dir: Path) -> dict[str, Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    _set_style()
    outputs = {
        "pipeline": figures_dir / "pipeline_overview.png",
        "retention": figures_dir / "catalog_and_locus_retention.png",
        "negative_composition": figures_dir / "negative_sample_composition.png",
        "locus": figures_dir / "color_locus_relaxed.png",
        "sky": figures_dir / "sky_distribution.png",
        "model": figures_dir / "model_performance.png",
        "curves": figures_dir / "precision_recall_roc.png",
        "importance": figures_dir / "feature_importance.png",
        "pool": figures_dir / "prediction_pool_audit_and_status.png",
        "top5": figures_dir / "top5_candidates.png",
    }
    _plot_pipeline(outputs["pipeline"])
    _plot_retention(evidence, outputs["retention"])
    _plot_negative_composition(evidence, outputs["negative_composition"])
    _plot_color_locus(evidence, outputs["locus"])
    _plot_sky(evidence, outputs["sky"])
    _plot_model_performance(evidence, outputs["model"])
    _plot_curves(evidence, outputs["curves"])
    _plot_feature_importance(evidence, outputs["importance"])
    _plot_pool(evidence, outputs["pool"])
    _plot_top5_candidates(evidence, outputs["top5"])
    return outputs


def _save(fig: plt.Figure, output: Path) -> None:
    temporary = output.with_name(
        f".{output.stem}.{uuid4().hex}.tmp{output.suffix}"
    )
    try:
        fig.savefig(
            temporary,
            format=output.suffix.lstrip("."),
            dpi=200,
            bbox_inches="tight",
            facecolor="white",
        )
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
        plt.close(fig)


def _plot_pipeline(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 8.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfdff")
    stages = [
        ("1", "Labelled samples", "Crowther WR catalogue\nControlled SIMBAD comparison"),
        ("2", "Enrich sources", "Gaia DR3 · 2MASS · AllWISE\nPhotometry and astrometry"),
        (
            "3",
            "Variants + locus",
            "8 compatible dataset variants\n6 intra-survey colour planes",
        ),
        ("4", "Train + validate", "144 configurations\nAP · top-K recovery · low FPR"),
        ("5", "Score Gaia pool", "Versioned exact-union acquisition\nEligibility checked per model"),
        ("6", "Follow-up review", "Five-model RRF ranking\nSpectroscopy confirms candidates"),
    ]
    positions = [(0.28, 0.74), (0.72, 0.74), (0.72, 0.49),
                 (0.28, 0.49), (0.28, 0.24), (0.72, 0.24)]
    card_width, card_height = 0.36, 0.175

    ax.text(
        0.07,
        0.945,
        "From catalogues to WR candidates",
        ha="left",
        va="center",
        fontsize=20,
        weight="bold",
        color=INK,
    )
    ax.text(
        0.07,
        0.897,
        "Six reproducible stages; spectroscopic confirmation remains essential.",
        ha="left",
        va="center",
        fontsize=11.5,
        color=MUTED,
    )
    ax.scatter([0.91, 0.94, 0.89, 0.96], [0.94, 0.91, 0.89, 0.96],
               s=[30, 13, 10, 17], marker="*", color="#d8e5f1", zorder=0)

    arrow_segments = [
        ((0.465, 0.74), (0.535, 0.74)),
        ((0.72, 0.646), (0.72, 0.584)),
        ((0.535, 0.49), (0.465, 0.49)),
        ((0.28, 0.396), (0.28, 0.334)),
        ((0.465, 0.24), (0.535, 0.24)),
    ]
    for start, end in arrow_segments:
        ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=20,
            linewidth=2, color="#7798ba", zorder=1,
        ))

    for index, (number, title, detail) in enumerate(stages):
        x, y = positions[index]
        is_output = index == len(stages) - 1
        accent = ORANGE if is_output else BLUE
        face = "#fff5ec" if is_output else "#f6f9fd"
        box = FancyBboxPatch(
            (x - card_width / 2, y - card_height / 2),
            card_width,
            card_height,
            boxstyle="round,pad=0.006,rounding_size=0.014",
            linewidth=1.35,
            edgecolor="#e8b47f" if is_output else "#bfd1e5",
            facecolor=face,
            zorder=2,
        )
        ax.add_patch(box)
        number_box = FancyBboxPatch(
            (x - card_width / 2 + 0.017, y + 0.035),
            0.042,
            0.041,
            boxstyle="round,pad=0.004,rounding_size=0.009",
            linewidth=0,
            facecolor=accent,
            zorder=3,
        )
        ax.add_patch(number_box)
        ax.text(
            x - card_width / 2 + 0.038,
            y + 0.055,
            number,
            ha="center",
            va="center",
            fontsize=11,
            weight="bold",
            color="white",
            zorder=4,
        )
        ax.text(
            x - card_width / 2 + 0.076,
            y + 0.054,
            title,
            ha="left",
            va="center",
            fontsize=13,
            weight="bold",
            color=INK,
            zorder=4,
        )
        ax.text(
            x - card_width / 2 + 0.022,
            y - 0.035,
            detail,
            ha="left",
            va="center",
            fontsize=11.1,
            color=MUTED,
            linespacing=1.45,
            zorder=4,
        )

    backbone = FancyBboxPatch(
        (0.095, 0.045),
        0.81,
        0.085,
        boxstyle="round,pad=0.007,rounding_size=0.012",
        linewidth=0.8,
        edgecolor="#d9e5f0",
        facecolor="#eef4fa",
        zorder=1,
    )
    ax.add_patch(backbone)
    ax.text(
        0.50,
        0.101,
        "TRACEABILITY  ·  versioned configs  ·  source and model hashes",
        ha="center",
        va="center",
        fontsize=10.6,
        weight="bold",
        color=BLUE,
    )
    ax.text(
        0.50,
        0.067,
        "DuckDB lineage  ·  run/result IDs  ·  auditable artefacts",
        ha="center",
        va="center",
        fontsize=10.3,
        color=MUTED,
    )

    fig.savefig(
        output.with_suffix(".svg"),
        bbox_inches="tight",
        facecolor="white",
        format="svg",
    )
    _save(fig, output)


def _plot_retention(evidence: Evidence, output: Path) -> None:
    wr_relaxed = evidence.locus.query(
        "variant == 'relaxed_photometry' and `class` == 'WR'"
    ).iloc[0]
    neg_relaxed = evidence.locus.query(
        "variant == 'relaxed_photometry' and `class` == 'negative'"
    ).iloc[0]
    wr_values = [
        evidence.catalogue["catalogue_rows"],
        evidence.catalogue["gaia_alias_rows"],
        int(wr_relaxed["rows"]),
        int(wr_relaxed["locus_keep"]),
    ]
    neg_values = [
        evidence.negative["raw_rows"],
        evidence.negative["enriched_rows"],
        int(neg_relaxed["rows"]),
        int(neg_relaxed["locus_keep"]),
    ]
    labels = ["Catalogue / query", "Gaia DR3", "Relaxed photometry", "Locus keep"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, values, title, color in [
        (axes[0], wr_values, "WR reference", ORANGE),
        (axes[1], neg_values, "SIMBAD negative sample", BLUE),
    ]:
        y = np.arange(len(labels))
        ax.barh(y, values, color=color, alpha=0.88, edgecolor=INK, linewidth=0.4)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_title(title, loc="left")
        ax.set_xlabel("Sources")
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        max_value = max(values)
        for i, value in enumerate(values):
            ax.text(
                value + max_value * 0.015,
                i,
                f"{value:,}",
                va="center",
                fontsize=9,
            )
        ax.set_xlim(0, max_value * 1.18)
    fig.suptitle(
        "Stage retention in the relaxed_photometry variant",
        x=0.07,
        ha="left",
        fontsize=13,
        weight="bold",
    )
    fig.text(
        0.07,
        0.01,
        f"The locus keeps {int(wr_relaxed['locus_keep'])}/{int(wr_relaxed['rows'])} WR "
        f"({wr_relaxed['locus_keep_fraction']:.1%}) and reduces "
        f"{int(neg_relaxed['rows']):,} negatives to {int(neg_relaxed['locus_keep']):,}.",
        color=MUTED,
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.93])
    _save(fig, output)


def _plot_negative_composition(evidence: Evidence, output: Path) -> None:
    mapping = {
        "RGB*": "Red giants",
        "YSO": "Young objects",
        "LongPeriodV*": "Long-period variables",
        "EmLine*": "Emission-line stars",
        "C*": "Carbon stars",
        "EclBin": "Eclipsing binaries",
        "Eruptive*": "Eruptive variables",
        "TTauri*": "T Tauri",
        "AGB*": "AGB",
        "Be*": "Be",
        "BlueSG": "Blue supergiants",
        "HighMassXBin": "High-mass X-ray binaries",
    }
    data = evidence.negative_types.copy()
    data["label"] = data["object_group"].map(mapping).fillna(data["object_group"])
    data = data.sort_values("rows")
    fig, ax = plt.subplots(figsize=(9.5, 5.3))
    colors = [BLUE if value >= 5_000 else BLUE_LIGHT for value in data["rows"]]
    ax.barh(data["label"], data["rows"], color=colors, edgecolor=INK, linewidth=0.35)
    ax.set_title("SIMBAD sample by query class", loc="left")
    ax.set_xlabel("SIMBAD sources with Gaia DR3")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    max_value = float(data["rows"].max())
    for i, value in enumerate(data["rows"]):
        ax.text(
            value + max_value * 0.012,
            i,
            f"{int(value):,}",
            va="center",
            fontsize=8.5,
        )
    ax.set_xlim(0, max_value * 1.17)
    fig.text(
        0.125,
        0.01,
        "The sample collects the contaminant classes selected in the SIMBAD queries.",
        fontsize=8.8,
        color=MUTED,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    _save(fig, output)


def _signed_transform(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.log1p(np.abs(values))


def _plot_color_locus(evidence: Evidence, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    filters = _read_yaml("configs/filters.yaml")["color_locus"]
    for ax, x_name, y_name, title in [
        (axes[0], "G_BP", "G_RP", "Gaia plane: G-BP vs G-RP"),
        (axes[1], "J_H", "J_K", "2MASS plane: J-H vs J-Ks"),
    ]:
        wr = evidence.reference_relaxed
        neg = evidence.negative_relaxed
        fit = fit_log_color_locus(
            wr,
            x_color=x_name,
            y_color=y_name,
            min_positive_fraction=1.0,
            estimator="huber",
            transform="signed_log1p",
            residual_quantile=float(filters["residual_quantile"]),
        )
        ax.hexbin(
            neg[x_name],
            neg[y_name],
            gridsize=52,
            mincnt=1,
            cmap="Blues",
            bins="log",
            alpha=0.78,
            linewidths=0,
        )
        kept = wr["color_locus_keep"].astype(bool)
        ax.scatter(
            wr.loc[kept, x_name],
            wr.loc[kept, y_name],
            s=24,
            c=ORANGE,
            edgecolors="white",
            linewidths=0.45,
            alpha=0.9,
            label="Retained WR",
            zorder=3,
        )
        ax.scatter(
            wr.loc[~kept, x_name],
            wr.loc[~kept, y_name],
            s=34,
            c=PINK,
            marker="x",
            linewidths=1.2,
            label="WR outside locus",
            zorder=4,
        )
        low = float(np.nanquantile(pd.concat([wr[x_name], neg[x_name]]), 0.005))
        high = float(np.nanquantile(pd.concat([wr[x_name], neg[x_name]]), 0.995))
        xs = np.linspace(low, high, 400)
        tx = _signed_transform(xs)
        center_t = float(fit["intercept"]) + float(fit["slope"]) * tx
        threshold = float(fit["residual_quantile_threshold"])
        center = inverse_transform_color_values(center_t, transform="signed_log1p")
        lower = inverse_transform_color_values(
            center_t - threshold, transform="signed_log1p"
        )
        upper = inverse_transform_color_values(
            center_t + threshold, transform="signed_log1p"
        )
        ax.plot(xs, center, color=INK, lw=1.5, label="Huber")
        ax.plot(xs, lower, color=INK, lw=1.0, ls="--", label="q97.5% of |r|")
        ax.plot(xs, upper, color=INK, lw=1.0, ls="--")
        ax.set_title(title, loc="left")
        ax.set_xlabel(x_name.replace("_", " - "))
        ax.set_ylabel(y_name.replace("_", " - "))
        ax.set_xlim(low, high)
        y_values = pd.concat([wr[y_name], neg[y_name]])
        ax.set_ylim(
            float(np.nanquantile(y_values, 0.005)),
            float(np.nanquantile(y_values, 0.995)),
        )
        ax.grid(alpha=0.45)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle(
        "Colour locus in relaxed_photometry",
        x=0.07,
        ha="left",
        fontsize=13,
        weight="bold",
    )
    fig.tight_layout(rect=[0, 0.07, 1, 0.94])
    _save(fig, output)


def _plot_sky(evidence: Evidence, output: Path) -> None:
    negatives = evidence.negative_positions
    coords = SkyCoord(
        ra=negatives["ra"].to_numpy() * u.deg,
        dec=negatives["dec"].to_numpy() * u.deg,
        frame="icrs",
    ).galactic
    l_neg = coords.l.wrap_at(180 * u.deg).radian
    b_neg = coords.b.radian
    wr = evidence.reference_positions
    l_wr = np.deg2rad(((wr["galactic_l"].to_numpy() + 180) % 360) - 180)
    b_wr = np.deg2rad(wr["galactic_b"].to_numpy())

    fig = plt.figure(figsize=(11.5, 5.4))
    ax = fig.add_subplot(111, projection="mollweide")
    ax.hexbin(
        -l_neg,
        b_neg,
        gridsize=80,
        mincnt=1,
        bins="log",
        cmap="Blues",
        alpha=0.84,
        linewidths=0,
    )
    ax.scatter(
        -l_wr,
        b_wr,
        s=18,
        c=ORANGE,
        edgecolors="white",
        linewidths=0.35,
        alpha=0.9,
        label="WR with Gaia DR3 aliases",
        zorder=3,
    )
    ax.grid(True, color="#c8d1dc", alpha=0.75)
    ax.set_title("Galactic distribution of the reference and negative sample", pad=18)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.16))
    fig.text(
        0.5,
        0.015,
        "l = 0° at centre and longitude increases to the left. Positions are reserved for spatial diagnostics.",
        ha="center",
        fontsize=8.8,
        color=MUTED,
    )
    fig.tight_layout(rect=[0.02, 0.05, 0.98, 0.97])
    _save(fig, output)


def _plot_model_performance(evidence: Evidence, output: Path) -> None:
    results = evidence.model_results.copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    model_colors = {
        "xgboost": ORANGE,
        "hist_gradient_boosting": BLUE,
        "random_forest": OLIVE,
    }
    sampler_markers = {"none": "o", "smote": "s", "smote_enn": "^"}
    for (model, sampler), group in results.groupby(["model", "sampler"]):
        axes[0].scatter(
            group["holdout_recall_at_100"],
            group["holdout_average_precision"],
            s=38,
            c=model_colors.get(model, MUTED),
            marker=sampler_markers.get(sampler, "o"),
            alpha=0.66,
            edgecolors="white",
            linewidths=0.35,
        )
    highlights = results[results["result_id"].isin([LEADING_RESULT_ID, AP_LEADER_RESULT_ID, STABLE_RESULT_ID])]
    label_specs = {
        LEADING_RESULT_ID: ("broad shortlist", "*", (-70, 10)),
        AP_LEADER_RESULT_ID: ("AP leader", "D", (-30, 14)),
        STABLE_RESULT_ID: ("accepted/stable", "P", (10, -18)),
    }
    for _, row in highlights.iterrows():
        label, marker, offset = label_specs[str(row["result_id"])]
        x_value = row["holdout_recall_at_100"]
        y_value = row["holdout_average_precision"]
        axes[0].scatter(
            [x_value],
            [y_value],
            s=140,
            marker=marker,
            c=GOLD,
            edgecolors="white",
            linewidths=1.2,
            zorder=5,
        )
        axes[0].annotate(
            label,
            (x_value, y_value),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
            color=INK,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, shrinkA=0, shrinkB=4),
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=GRID, lw=0.6, alpha=0.92),
            zorder=6,
        )
    axes[0].set_title("AP vs Recall@100", loc="left")
    axes[0].set_xlabel("Recall @100")
    axes[0].set_ylabel("Holdout average precision")
    axes[0].set_xlim(0, 0.83)
    axes[0].set_ylim(0, max(0.62, float(results["holdout_average_precision"].max()) * 1.08))
    model_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=color, label=model)
        for model, color in model_colors.items()
    ]
    sampler_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="",
            markerfacecolor="white",
            markeredgecolor=INK,
            color=INK,
            label=sampler,
        )
        for sampler, marker in sampler_markers.items()
    ]
    axes[0].legend(handles=model_handles + sampler_handles, fontsize=7.7, ncol=2)

    comparable = results[
        results["result_id"].isin(
            [LEADING_RESULT_ID, AP_LEADER_RESULT_ID, HGB_RESULT_ID, STABLE_RESULT_ID]
        )
    ].copy()
    label_map = {
        LEADING_RESULT_ID: "Relaxed XGB / none",
        AP_LEADER_RESULT_ID: "Relaxed XGB / SMOTE",
        HGB_RESULT_ID: "Relaxed HGB / none",
        STABLE_RESULT_ID: "Strict poe2 XGB / SMOTE",
    }
    metrics = [
        ("holdout_average_precision", "AP"),
        ("holdout_recall_at_50", "R@50"),
        ("holdout_recall_at_100", "R@100"),
        ("holdout_precision_at_100", "P@100"),
        ("holdout_recall_at_fpr_0p005", "R @ FPR 0.5%"),
    ]
    x = np.arange(len(metrics))
    width = 0.19
    colors = [ORANGE, PINK, BLUE, OLIVE]
    for idx, (_, row) in enumerate(comparable.iterrows()):
        values = [float(row[column]) for column, _ in metrics]
        axes[1].bar(
            x + (idx - 1.5) * width,
            values,
            width,
            label=label_map[str(row["result_id"])],
            color=colors[idx],
            edgecolor=INK,
            linewidth=0.35,
        )
    axes[1].set_xticks(x, [label for _, label in metrics])
    axes[1].set_ylim(0, 0.82)
    axes[1].set_title("Selected result profiles", loc="left")
    axes[1].set_ylabel("Proportion")
    axes[1].legend(fontsize=7.4, loc="upper right")
    fig.suptitle(
        "run_v3_main results (144 configurations)",
        x=0.07,
        ha="left",
        fontsize=13,
        weight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, output)


def _plot_curves(evidence: Evidence, output: Path) -> None:
    selected = [
        (LEADING_RESULT_ID, "XGB / none", ORANGE, "-"),
        (AP_LEADER_RESULT_ID, "XGB / SMOTE", PINK, "--"),
        (HGB_RESULT_ID, "HGB / none", BLUE, "-."),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for result_id, label, color, linestyle in selected:
        rows = evidence.predictions[evidence.predictions["result_id"].eq(result_id)]
        y = rows["target"].astype(int).to_numpy()
        score = rows["score"].astype(float).to_numpy()
        precision, recall, _ = precision_recall_curve(y, score)
        ap = average_precision_score(y, score)
        fpr, tpr, _ = roc_curve(y, score)
        axes[0].plot(
            recall,
            precision,
            color=color,
            linestyle=linestyle,
            lw=1.8,
            label=f"{label} (AP={ap:.3f})",
        )
        axes[1].plot(
            fpr,
            tpr,
            color=color,
            linestyle=linestyle,
            lw=1.8,
            label=label,
        )
    prevalence = (
        evidence.predictions[
            evidence.predictions["result_id"].eq(LEADING_RESULT_ID)
        ]["target"].mean()
    )
    axes[0].axhline(prevalence, color=INK, lw=1.0, ls=":", label="Holdout prevalence")
    axes[0].set_title("Precision-Recall curves", loc="left")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(fontsize=8)
    axes[1].plot([0, 1], [0, 1], color=INK, lw=1.0, ls=":", label="Random")
    axes[1].set_title("ROC curves", loc="left")
    axes[1].set_xlabel("False-positive rate")
    axes[1].set_ylabel("True-positive rate")
    axes[1].set_xlim(0, 0.03)
    axes[1].set_ylim(0, 1.02)
    axes[1].legend(fontsize=8)
    fig.suptitle(
        "Comparison on the same relaxed_photometry holdout",
        x=0.07,
        ha="left",
        fontsize=13,
        weight="bold",
    )
    fig.text(
        0.07,
        0.01,
        "The ROC view is zoomed to 3% FPR: in a massive pool, small FPR differences dominate false-positive volume.",
        color=MUTED,
        fontsize=8.8,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    _save(fig, output)


def _plot_feature_importance(evidence: Evidence, output: Path) -> None:
    data = evidence.feature_importance.sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    colors = [ORANGE if feature == "W1_W2" else BLUE for feature in data["feature"]]
    ax.barh(
        data["feature"],
        data["importance_mean"],
        color=colors,
        edgecolor=INK,
        linewidth=0.35,
    )
    ax.set_title("Internal XGBoost importance: relaxed_photometry / none", loc="left")
    ax.set_xlabel("Normalized estimator importance")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    fig.text(
        0.125,
        0.01,
        "Estimator gain importance for this fit and this training set.",
        fontsize=8.8,
        color=MUTED,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    _save(fig, output)


def _plot_pool(evidence: Evidence, output: Path) -> None:
    audit = evidence.pool_audit
    status = evidence.pool_status
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    exact = audit["compatible_exact_union_keep"]
    missed = audit["compatible_exact_not_aggregate"]
    common = exact - missed
    axes[0].barh(
        ["Compatible exact union"],
        [common],
        color=BLUE,
        edgecolor=INK,
        linewidth=0.4,
        label="Also accepted by the aggregate",
    )
    axes[0].barh(
        ["Compatible exact union"],
        [missed],
        left=[common],
        color=PINK,
        edgecolor=INK,
        linewidth=0.4,
        label="Dropped by the legacy aggregate",
    )
    axes[0].set_title("Legacy filter audit", loc="left")
    axes[0].set_xlabel("Sources in 18 Gaia regions")
    axes[0].legend(fontsize=8, loc="lower right")
    axes[0].text(
        common + missed / 2,
        0,
        f"{missed:,}\n({audit['loss_fraction']:.2%})",
        ha="center",
        va="center",
        color="white",
        weight="bold",
        fontsize=9,
    )

    completed = status["completed_tiles"]
    total = status["total_tiles"]
    axes[1].barh(
        ["Tiles"],
        [completed],
        color=ORANGE,
        edgecolor=INK,
        linewidth=0.4,
        label="Completed",
    )
    axes[1].barh(
        ["Tiles"],
        [max(total - completed, 0)],
        left=[completed],
        color="#e8edf2",
        edgecolor=INK,
        linewidth=0.4,
        label="Pending",
    )
    axes[1].set_xlim(0, total)
    axes[1].set_title(
        (
            "Exact-union build completed"
            if status["status"] == "completed"
            else "Exact-union build in progress"
        ),
        loc="left",
    )
    axes[1].set_xlabel("Terminal tiles")
    axes[1].text(
        completed / 2 if completed else 2,
        0,
        f"{completed}/{total}",
        ha="center",
        va="center",
        color="white" if completed else INK,
        weight="bold",
    )
    axes[1].text(
        total * 0.5,
        -0.42,
        (
            f"{status['acquired_rows'] / 1e6:.1f} M acquired; "
            f"{status['eligible_rows'] / 1e6:.1f} M eligible for scoring"
        ),
        ha="center",
        fontsize=8.8,
        color=MUTED,
    )
    axes[1].legend(fontsize=8, loc="lower right")
    fig.suptitle(
        "Legacy pool audit and exact-union status",
        x=0.07,
        ha="left",
        fontsize=13,
        weight="bold",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    _save(fig, output)


def _plot_top5_candidates(evidence: Evidence, output: Path) -> None:
    data = evidence.top_candidates
    if data.empty:
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.axis("off")
        ax.text(
            0.5,
            0.55,
            "Top-5 candidates unavailable",
            ha="center",
            va="center",
            fontsize=13,
            weight="bold",
            color=INK,
        )
        ax.text(
            0.5,
            0.35,
            "Run build-prediction-pool-candidates to write top_5_candidates.csv.",
            ha="center",
            va="center",
            fontsize=10,
            color=MUTED,
        )
        _save(fig, output)
        return

    plot_data = data.sort_values("followup_rank").reset_index(drop=True)
    context = evidence.candidate_context
    if not context.empty:
        context = context.loc[~context["source_id"].isin(plot_data["source_id"])]

    fig = plt.figure(figsize=(10.4, 9.3), facecolor="white")
    grid = fig.add_gridspec(
        2, 2, left=0.09, right=0.96, top=0.76, bottom=0.20,
        height_ratios=[1.08, 1], hspace=0.39, wspace=0.28,
    )
    sky_ax = fig.add_subplot(grid[0, :], projection="mollweide")
    cmd_ax = fig.add_subplot(grid[1, 0])
    w12_ax = fig.add_subplot(grid[1, 1])

    fig.text(0.07, 0.965, "Five priorities in the candidate landscape",
             fontsize=18, weight="bold", color=INK, va="top")
    fig.text(0.07, 0.92,
             "Real consensus top-100 sources provide context for the five spectroscopic follow-ups."
             if not context.empty else "Five spectroscopic follow-ups from the persisted candidate review.",
             fontsize=11, color=MUTED, va="top")
    legend_handles = [
        Line2D([0], [0], marker="*", linestyle="", markersize=12,
               markerfacecolor=ORANGE, markeredgecolor=INK,
               label="Five follow-up priorities"),
    ]
    if not context.empty:
        legend_handles.insert(
            0, Line2D([0], [0], marker="o", linestyle="", markersize=6,
                      markerfacecolor="#a9bbcc", markeredgecolor="none",
                      label="Other top-100 sources"),
        )
    fig.legend(
        handles=legend_handles,
        loc="upper left", bbox_to_anchor=(0.07, 0.875), ncol=2,
        frameon=False, fontsize=10.5, handletextpad=0.5, columnspacing=1.8,
    )

    if not context.empty:
        sky_context = context.dropna(subset=["galactic_l", "galactic_b"])
        l_context = np.deg2rad(((sky_context["galactic_l"].to_numpy() + 180) % 360) - 180)
        sky_ax.scatter(-l_context, np.deg2rad(sky_context["galactic_b"]),
                       s=24, c="#a9bbcc", alpha=0.75, linewidths=0, zorder=2)
        cmd_context = context.dropna(subset=["BP_RP", "G"])
        cmd_ax.scatter(cmd_context["BP_RP"], cmd_context["G"],
                       s=26, c="#a9bbcc", alpha=0.67, linewidths=0, zorder=2)
        w12_context = context.dropna(subset=["W1_W2", "rrf_score"])
        w12_ax.scatter(w12_context["W1_W2"], w12_context["rrf_score"],
                       s=26, c="#a9bbcc", alpha=0.67, linewidths=0, zorder=2)

    cmd_label_offsets = {
        1: (-18, 8), 2: (7, -15), 3: (8, 9),
        4: (7, 7), 5: (-19, -12),
    }
    for row in plot_data.itertuples(index=False):
        l_rad = np.deg2rad(((row.galactic_l + 180) % 360) - 180)
        sky_ax.scatter(-l_rad, np.deg2rad(row.galactic_b),
                       s=165, marker="*", c=ORANGE, edgecolors=INK,
                       linewidths=0.8, zorder=5)
        for axis, x_value, y_value, label_offset in [
            (cmd_ax, row.BP_RP, row.G,
             cmd_label_offsets.get(int(row.followup_rank), (7, 6))),
            (w12_ax, row.W1_W2, row.rrf_score, (7, 6)),
        ]:
            axis.scatter(x_value, y_value, s=155, marker="*", c=ORANGE,
                         edgecolors=INK, linewidths=0.8, zorder=5)
            axis.annotate(
                str(int(row.followup_rank)), (x_value, y_value),
                xytext=label_offset, textcoords="offset points", fontsize=10.5,
                weight="bold", color=INK,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.8},
                zorder=6,
            )

    sky_ax.grid(True, color="#d6e0e9", alpha=0.85, linewidth=0.75)
    sky_ax.set_xticklabels([])
    sky_ax.tick_params(axis="y", labelsize=9, colors=MUTED)
    sky_ax.set_title("A  ·  Galactic sky", loc="left", fontsize=13, weight="bold", color=INK, pad=11)
    for axis in (cmd_ax, w12_ax):
        axis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        axis.tick_params(labelsize=9.5, colors=MUTED)
        for spine in axis.spines.values():
            spine.set_color("#a8b7c8")
        axis.margins(x=0.08, y=0.12)
    cmd_ax.invert_yaxis()
    cmd_ax.set(xlabel="Gaia BP − RP (mag)", ylabel="Gaia G (mag)")
    cmd_ax.set_title("B  ·  Gaia colour–magnitude", loc="left", fontsize=13,
                     weight="bold", color=INK, pad=10)
    w12_ax.set(xlabel="WISE W1 − W2 (mag)", ylabel="RRF score")
    w12_ax.set_title("C  ·  Infrared colour and consensus", loc="left",
                     fontsize=13, weight="bold", color=INK, pad=10)
    fig.text(0.07, 0.105,
             "Numbers are follow-up ranks. Galactic longitude increases left (l = 0° at centre).",
             color=MUTED, fontsize=10.5)
    fig.text(0.07, 0.070,
             "Grey sources are the remaining consensus top 100; RRF scores are not calibrated probabilities."
             if not evidence.candidate_context.empty else
             "Top-100 context unavailable; RRF scores are not calibrated probabilities.",
             color=MUTED, fontsize=10.5)
    _save(fig, output)


def _format_int(value: int | float) -> str:
    return f"{int(value):,}"


def _evidence_json(evidence: Evidence) -> dict[str, Any]:
    selected_columns = [
        "result_id",
        "dataset_variant",
        "feature_set",
        "model",
        "sampler",
        "selection_status",
        "wr_holdout",
        "negative_holdout",
        "holdout_average_precision",
        "holdout_roc_auc",
        "holdout_recall_at_50",
        "holdout_recall_at_100",
        "holdout_precision_at_100",
        "holdout_recall_at_fpr_0p005",
        "holdout_fpr",
        "ranking_score",
        "overfit_warning_flag",
        "overfit_risk_score",
    ]
    selected_models = evidence.model_results[
        evidence.model_results["result_id"].isin(
            [LEADING_RESULT_ID, AP_LEADER_RESULT_ID, STABLE_RESULT_ID]
        )
    ][selected_columns]
    return {
        "snapshot_date": evidence.snapshot_date,
        "catalogue": evidence.catalogue,
        "negative": evidence.negative,
        "locus_summary": evidence.locus.to_dict(orient="records"),
        "split_relaxed_photometry": evidence.split.to_dict(orient="records"),
        "selected_models": selected_models.to_dict(orient="records"),
        "prediction_pool_audit": evidence.pool_audit,
        "prediction_pool_status": evidence.pool_status,
    }


def write_source_notes(
    evidence: Evidence,
    figures: dict[str, Path],
    output_dir: Path,
) -> tuple[Path, Path]:
    snapshot_path = output_dir / "project_summary_evidence.json"
    snapshot_path.write_text(
        json.dumps(_evidence_json(evidence), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    chart_map = output_dir / "chart_map.md"
    lines = [
        "# Chart map",
        "",
        f"Evidence snapshot: `{snapshot_path.relative_to(PROJECT_ROOT)}`",
        "",
        "| Figure | Question | Chart family | Main source |",
        "|---|---|---|---|",
        (
            f"| `{figures['pipeline'].name}` | How does a source move through the project? "
            "| Process flow | Code/configuration contract |"
        ),
        (
            f"| `{figures['retention'].name}` | How much of each class remains by stage? "
            "| Horizontal bars | GWRC/SIMBAD DuckDB + relaxed Parquet |"
        ),
        (
            f"| `{figures['negative_composition'].name}` | Which contaminant groups form the controlled negative sample? "
            "| Ranked bars | `simbad_negative_sources` |"
        ),
        (
            f"| `{figures['locus'].name}` | Where do WR and negatives sit relative to the robust locus? "
            "| Density + scatter + model band | relaxed color-locus Parquet |"
        ),
        (
            f"| `{figures['sky'].name}` | Where are the labelled samples on the sky? "
            "| Mollweide density + points | reference/negative DuckDB |"
        ),
        (
            f"| `{figures['model'].name}` | What trade-offs appear across the 144-model grid? "
            "| Scatter + grouped bars | training history `run_v3_main` |"
        ),
        (
            f"| `{figures['curves'].name}` | How do comparable models behave under class imbalance? "
            "| PR/ROC lines | holdout predictions in training history |"
        ),
        (
            f"| `{figures['importance'].name}` | Which features does the leading broad model use? "
            "| Horizontal bars | stored model importance |"
        ),
        (
            f"| `{figures['pool'].name}` | Why was the legacy pool rejected and what is the new build status? "
            "| Stacked bars | 18-region audit + exact-union build registry |"
        ),
        (
            f"| `{figures['top5'].name}` | Where do the five follow-up priorities sit on the sky and in colour space? "
            "| Mollweide + CMD + W1-W2 support | `top_5_candidates.csv` and consensus top 100 from `top_500_candidates.csv` |"
        ),
        "",
        "Palette policy: one blue root for context, orange for WR/focal results, "
        "pink for exclusions or discrepancies, and neutral grey scaffolding.",
    ]
    chart_map.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return snapshot_path, chart_map


def _reportlab_imports() -> dict[str, Any]:
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.platypus import (
            BaseDocTemplate,
            Frame,
            HRFlowable,
            Image,
            KeepTogether,
            PageBreak,
            PageTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError as exc:
        raise RuntimeError(
            "ReportLab is required for PDF generation. Install with "
            "`python -m pip install -e \".[report]\"`."
        ) from exc
    return locals()


def build_pdf(
    evidence: Evidence,
    figures: dict[str, Path],
    output_pdf: Path,
) -> None:
    rl = _reportlab_imports()
    colors = rl["colors"]
    A4 = rl["A4"]
    cm = rl["cm"]
    pdfmetrics = rl["pdfmetrics"]
    TTFont = rl["TTFont"]
    BaseDocTemplate = rl["BaseDocTemplate"]
    Frame = rl["Frame"]
    PageTemplate = rl["PageTemplate"]
    Paragraph = rl["Paragraph"]
    Spacer = rl["Spacer"]
    Table = rl["Table"]
    TableStyle = rl["TableStyle"]
    Image = rl["Image"]
    PageBreak = rl["PageBreak"]
    KeepTogether = rl["KeepTogether"]
    HRFlowable = rl["HRFlowable"]
    ParagraphStyle = rl["ParagraphStyle"]
    getSampleStyleSheet = rl["getSampleStyleSheet"]
    TA_CENTER = rl["TA_CENTER"]
    TA_LEFT = rl["TA_LEFT"]

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    font_regular = Path("C:/Windows/Fonts/arial.ttf")
    font_bold = Path("C:/Windows/Fonts/arialbd.ttf")
    font_italic = Path("C:/Windows/Fonts/ariali.ttf")
    if font_regular.exists():
        pdfmetrics.registerFont(TTFont("ProjectSans", str(font_regular)))
        pdfmetrics.registerFont(TTFont("ProjectSans-Bold", str(font_bold)))
        pdfmetrics.registerFont(TTFont("ProjectSans-Italic", str(font_italic)))
        base_font = "ProjectSans"
        bold_font = "ProjectSans-Bold"
        italic_font = "ProjectSans-Italic"
    else:
        base_font = "Helvetica"
        bold_font = "Helvetica-Bold"
        italic_font = "Helvetica-Oblique"

    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleProject",
        parent=styles["Title"],
        fontName=bold_font,
        fontSize=25,
        leading=29,
        textColor=colors.HexColor(INK),
        alignment=TA_LEFT,
        spaceAfter=10,
    )
    subtitle = ParagraphStyle(
        "SubtitleProject",
        parent=styles["Normal"],
        fontName=base_font,
        fontSize=11,
        leading=15,
        textColor=colors.HexColor(MUTED),
        spaceAfter=12,
    )
    h1 = ParagraphStyle(
        "H1Project",
        parent=styles["Heading1"],
        fontName=bold_font,
        fontSize=16,
        leading=20,
        textColor=colors.HexColor(INK),
        spaceBefore=4,
        spaceAfter=8,
        keepWithNext=True,
    )
    h2 = ParagraphStyle(
        "H2Project",
        parent=styles["Heading2"],
        fontName=bold_font,
        fontSize=11.5,
        leading=15,
        textColor=colors.HexColor(BLUE),
        spaceBefore=7,
        spaceAfter=4,
        keepWithNext=True,
    )
    body = ParagraphStyle(
        "BodyProject",
        parent=styles["BodyText"],
        fontName=base_font,
        fontSize=9.15,
        leading=13.1,
        textColor=colors.HexColor(INK),
        alignment=TA_LEFT,
        spaceAfter=6,
    )
    small = ParagraphStyle(
        "SmallProject",
        parent=body,
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor(MUTED),
        spaceAfter=3,
    )
    callout = ParagraphStyle(
        "CalloutProject",
        parent=body,
        fontName=bold_font,
        fontSize=9.3,
        leading=13.3,
        textColor=colors.HexColor(INK),
        backColor=colors.HexColor(PALE),
        borderColor=colors.HexColor(BLUE_LIGHT),
        borderWidth=0.6,
        borderPadding=8,
        spaceBefore=4,
        spaceAfter=8,
    )
    formula = ParagraphStyle(
        "FormulaProject",
        parent=body,
        fontName=base_font,
        fontSize=9.2,
        leading=14,
        leftIndent=10,
        rightIndent=10,
        backColor=colors.HexColor("#f7f9fb"),
        borderColor=colors.HexColor("#d7e0e9"),
        borderWidth=0.5,
        borderPadding=7,
        spaceBefore=3,
        spaceAfter=7,
    )
    bullet = ParagraphStyle(
        "BulletProject",
        parent=body,
        bulletIndent=9,
        leftIndent=20,
        firstLineIndent=0,
        spaceAfter=3,
    )
    reference_style = ParagraphStyle(
        "ReferenceProject",
        parent=small,
        fontSize=7.15,
        leading=9.2,
        leftIndent=10,
        firstLineIndent=-10,
        textColor=colors.HexColor(INK),
    )

    page_width, page_height = A4
    left_margin = 1.55 * cm
    right_margin = 1.55 * cm
    top_margin = 1.55 * cm
    bottom_margin = 1.45 * cm
    frame = Frame(
        left_margin,
        bottom_margin,
        page_width - left_margin - right_margin,
        page_height - top_margin - bottom_margin,
        id="content",
    )

    def draw_page(canvas: Any, doc: Any) -> None:
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#d9e0e7"))
        canvas.setLineWidth(0.45)
        canvas.line(left_margin, 1.1 * cm, page_width - right_margin, 1.1 * cm)
        canvas.setFont(base_font, 7.2)
        canvas.setFillColor(colors.HexColor(MUTED))
        canvas.drawString(left_margin, 0.72 * cm, "Wolf-Rayet Search - informe técnico")
        canvas.drawRightString(
            page_width - right_margin,
            0.72 * cm,
            f"{evidence.snapshot_date}  |  {doc.page}",
        )
        canvas.restoreState()

    doc = BaseDocTemplate(
        str(output_pdf),
        pagesize=A4,
        leftMargin=left_margin,
        rightMargin=right_margin,
        topMargin=top_margin,
        bottomMargin=bottom_margin,
        title="Informe técnico - Wolf-Rayet Search",
        author="Diego Miranda",
        subject="Resumen técnico y estado del proyecto",
    )
    doc.addPageTemplates(PageTemplate(id="main", frames=[frame], onPage=draw_page))

    def p(text: str, style: Any = body, **kwargs: Any) -> Any:
        return Paragraph(text, style, **kwargs)

    def img(path: Path, width_cm: float, height_cm: float | None = None) -> Any:
        image = Image(str(path))
        original_width = image.imageWidth
        original_height = image.imageHeight
        image.drawWidth = width_cm * cm
        if height_cm is None:
            image.drawHeight = original_height * image.drawWidth / original_width
        else:
            image.drawHeight = height_cm * cm
        return image

    def caption(text: str) -> Any:
        return p(text, small)

    def table(data: list[list[Any]], widths: list[float]) -> Any:
        result = Table(data, colWidths=[value * cm for value in widths], repeatRows=1)
        result.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eaf0f6")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor(INK)),
                    ("FONTNAME", (0, 0), (-1, 0), bold_font),
                    ("FONTNAME", (0, 1), (-1, -1), base_font),
                    ("FONTSIZE", (0, 0), (-1, -1), 7.6),
                    ("LEADING", (0, 0), (-1, -1), 9.8),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cfd8e2")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
                        colors.white,
                        colors.HexColor("#f8fafc"),
                    ]),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        return result

    wr_relaxed = evidence.locus.query(
        "variant == 'relaxed_photometry' and `class` == 'WR'"
    ).iloc[0]
    neg_relaxed = evidence.locus.query(
        "variant == 'relaxed_photometry' and `class` == 'negative'"
    ).iloc[0]
    model_lookup = evidence.model_results.set_index("result_id")
    leading = model_lookup.loc[LEADING_RESULT_ID]
    ap_leader = model_lookup.loc[AP_LEADER_RESULT_ID]
    stable = model_lookup.loc[STABLE_RESULT_ID]
    pool = evidence.pool_status
    audit = evidence.pool_audit

    story: list[Any] = []
    story.extend(
        [
            Spacer(1, 0.55 * cm),
            p("Informe técnico", subtitle),
            p("Wolf-Rayet Search", title),
            p(
                "Construcción reproducible de muestras y ranking fotométrico de candidatas Wolf-Rayet galácticas",
                subtitle,
            ),
            HRFlowable(
                width="100%",
                thickness=1.1,
                color=colors.HexColor(BLUE),
                spaceBefore=2,
                spaceAfter=12,
            ),
            p("Resumen técnico", h1),
            p(
                "El proyecto produce una <b>lista priorizada de fuentes compatibles con estrellas Wolf-Rayet (WR)</b> "
                "a partir de fotometría Gaia DR3, 2MASS y WISE. No intenta declarar nuevas WR de forma automática: "
                "la confirmación final sigue siendo espectroscópica. El problema se trata como ranking de objetos raros, "
                "con énfasis en recuperación dentro de presupuestos top-K y comportamiento a baja tasa de falsos positivos.",
                body,
            ),
            p(
                f"La referencia local parte del Galactic Wolf Rayet Catalogue {evidence.catalogue['version']} "
                f"y contiene {_format_int(evidence.catalogue['gaia_alias_rows'])} fuentes con alias Gaia DR3 explícito. "
                f"La muestra negativa controlada contiene {_format_int(evidence.negative['enriched_rows'])} "
                "fuentes SIMBAD y cubre clases capaces de ocupar regiones fotométricas similares.",
                body,
            ),
            p(
                f"En la variante más amplia usada por los modelos, <b>relaxed_photometry</b>, el locus conserva "
                f"{_format_int(wr_relaxed['locus_keep'])}/{_format_int(wr_relaxed['rows'])} WR "
                f"({wr_relaxed['locus_keep_fraction']:.1%}) y deja "
                f"{_format_int(neg_relaxed['locus_keep'])}/{_format_int(neg_relaxed['rows'])} negativos. "
                "El locus reduce la región de conflicto fotométrico sin pretender recuperar toda estrella del plano galáctico.",
                callout,
            ),
            img(figures["pipeline"], 16.2),
            caption(
                "Figura 1. Flujo completo del proyecto. Las etapas de adquisición, elegibilidad y scoring quedan separadas."
            ),
            p("Estado actual", h2),
            p(
                f"La grilla principal <b>{DEFAULT_RUN_ID}</b> terminó sus 144 configuraciones. "
                "La nueva prediction pool exact-union está en construcción y todavía no está aprobada para scoring definitivo. "
                f"En el corte usado para este informe hay {pool['completed_tiles']}/{pool['total_tiles']} tiles terminados, "
                f"{pool['acquired_rows'] / 1e6:.1f} millones de filas adquiridas y "
                f"{pool['eligible_rows'] / 1e6:.1f} millones de filas elegibles escritas. "
                "Estas cifras son progreso de ingeniería, no un resultado científico final.",
                body,
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            p("1. Datos de referencia y muestra de comparación", h1),
            p(
                "La referencia positiva se obtiene desde el catálogo mantenido por Paul Crowther. "
                "La versión descargada se guarda como snapshot HTML/CSV con fecha y SHA-256. "
                "Solo se incorporan al pipeline reproducible las entradas con un alias explícito `Gaia DR3 <source_id>`; "
                "no se infiere la identidad Gaia desde coordenadas cuando el catálogo no la declara.",
                body,
            ),
            p(
                "Las fuentes Gaia se enriquecen con fotometría 2MASS y AllWISE. Se usa primero el crossmatch oficial de Gaia; "
                "si falta una contraparte, se aplica un cone search VizieR y se conserva el método y la separación angular. "
                "Esto permite auditar la procedencia de cada medición. Una prueba de sensibilidad posterior mostró que "
                "entrenar con todas las WR disponibles no rindió peor que restringir la referencia a contrapartes IR nativas de Gaia.",
                body,
            ),
            img(figures["retention"], 16.2),
            caption(
                "Figura 2. Retención de fuentes por etapa para relaxed_photometry. Los conteos proceden de los DuckDB y Parquet locales."
            ),
            p(
                "Los negativos se consultan en SIMBAD por grupos de objetos plausibles como contaminantes: estrellas de emisión, "
                "Be, T Tauri, objetos jóvenes, gigantes rojas, AGB, carbono, variables, binarias y supergigantes azules. "
                "Se excluyen identificadores y tipos compatibles con WR antes del enriquecimiento. Esta muestra es deliberadamente "
                "controlada: sirve para comparar y estresar el ranking, pero no estima por sí sola la prevalencia real de no-WR en Gaia.",
                body,
            ),
            img(figures["negative_composition"], 15.7),
            caption(
                "Figura 3. Composición por clase de consulta SIMBAD. Los límites de 10.000 filas por algunas clases reflejan el diseño de adquisición."
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            p("2. Familias fotométricas y locus de color", h1),
            p(
                "Las variantes combinan una familia fotométrica con un subconjunto astrométrico. "
                "<b>strict</b> exige calidad A en las bandas requeridas de 2MASS y WISE; "
                "<b>relaxed</b> acepta A o B. El eje astrométrico usa photometry (sin corte de paralaje), "
                "parallax_soft (paralaje positiva), poe_2 o poe_3 "
                "(paralaje positiva y parallax_over_error por sobre 2 o 3). "
                "poe_5 existe como auditoría, pero se excluye de la grilla por la pérdida adicional de WR.",
                body,
            ),
            p(
                "El locus usa solo colores intra-misión. En Gaia se ajustan los planos "
                "(G-BP, G-RP), (G-BP, BP-RP) y (G-RP, BP-RP); en 2MASS, "
                "(J-H, J-Ks), (J-Ks, H-Ks) y (J-H, H-Ks). W1-W2 queda como feature del modelo, "
                "pero WISE no define un plano porque W3/W4 no son bandas obligatorias. "
                "Los colores cruzados entre misiones no se usan para evitar que épocas, calibraciones y crossmatches "
                "definan la frontera del locus.",
                body,
            ),
            p(
                "<b>Transformación y ajuste por plano</b><br/>"
                "T(c) = sign(c) ln(1 + |c|)<br/>"
                "T(y<sub>i</sub>) = beta<sub>0</sub> + beta<sub>1</sub>T(x<sub>i</sub>) + epsilon<sub>i</sub><br/>"
                "r<sub>i</sub> = T(y<sub>i</sub>) - [beta<sub>0</sub> + beta<sub>1</sub>T(x<sub>i</sub>)]",
                formula,
            ),
            p(
                "Se usa HuberRegressor para que unas pocas desviaciones grandes no controlen la recta. "
                "El umbral de cada plano es el percentil 97,5 de |r| medido en la referencia WR. "
                "Una fuente se marca como outlier agregado solo si queda fuera de al menos dos planos:",
                body,
            ),
            p(
                "outlier<sub>i</sub> = I { sum<sub>p</sub> I(|r<sub>ip</sub>| &gt; q<sub>0.975,p</sub>) &gt;= 2 }<br/>"
                "color_locus_keep<sub>i</sub> = valid<sub>i</sub> AND NOT outlier<sub>i</sub>",
                formula,
            ),
            p(
                "La elección de signed_log1p reemplaza el intento de trabajar con log10 de colores positivos. "
                "Esa formulación no era compatible con colores que pueden ser negativos por definición. "
                f"En relaxed_photometry solo {_format_int(wr_relaxed['positive_six'])}/"
                f"{_format_int(wr_relaxed['rows'])} WR tienen simultáneamente positivos los seis colores; "
                "en strict_photometry son 0/306. signed_log1p conserva el signo, está definida en cero y usa "
                "las 347 filas finitas de la variante relaxed.",
                callout,
            ),
            img(figures["locus"], 16.2),
            caption(
                "Figura 4. Dos de los seis planos del locus. La densidad azul corresponde a negativos y los puntos naranjos a WR."
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            p("3. Reducción de negativos, split y features", h1),
            p(
                "El locus se calcula antes del split y color_locus_keep se aplica a todas las capas de modelado. "
                "El holdout se determina con un hash estable de source_id y se estratifica por clase: aproximadamente "
                "20% de WR y 20% de negativos quedan fuera del entrenamiento. Después se conservan todas las WR no-holdout "
                "y se muestrean negativos de entrenamiento a razón 10:1 mediante estratos de cuantiles de color. "
                "Los negativos no seleccionados se mantienen como threshold_calibration; no desaparecen del experimento.",
                body,
            ),
            p(
                "En relaxed_photometry, antes de exigir completitud de features, el train contiene 262 WR y 2.620 negativos; "
                "el holdout contiene 69 WR y 6.634 negativos; threshold_calibration contiene 24.089 negativos. "
                "Para el modelo XGBoost/none destacado, la evaluación completa usa 68 WR y 6.558 negativos.",
                body,
            ),
            p(
                "Los dos sets de features son: (1) siete colores intra-misión + paralaje; "
                "(2) los mismos colores + paralaje, error de paralaje y parallax_over_error. "
                "No se incorporan coordenadas ni colores cruzados. La paralaje se interpreta como información de escala/distancia "
                "y no como una distancia exacta; por eso también se prueban variantes sin cortes astrométricos fuertes.",
                body,
            ),
            img(figures["sky"], 16.2),
            caption(
                "Figura 5. Distribución galáctica de los ejemplos etiquetados. El mapa es diagnóstico; l y b no forman parte de los features actuales."
            ),
            p("Modelos y remuestreo", h2),
            p(
                "La grilla cruza ocho variantes, dos feature sets, Random Forest, HistGradientBoosting y XGBoost, "
                "con none, SMOTE y SMOTE-ENN. El remuestreo se ejecuta dentro de cada fold para evitar leakage. "
                "none usa ponderación nativa del estimador; las configuraciones muestreadas usan pesos neutros para no corregir "
                "dos veces el mismo desbalance. BayesSearchCV ejecuta 25 iteraciones por configuración y optimiza average precision.",
                body,
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            p("4. Evaluación como ranking de objetos raros", h1),
            p(
                "Accuracy no es la métrica de decisión: en un pool de decenas o cientos de millones, predecir casi todo como no-WR "
                "puede dar accuracy alta sin producir una lista útil. La curva Precision-Recall y Average Precision enfatizan la clase "
                "positiva y son más informativas que ROC cuando el conjunto está fuertemente desbalanceado [4]. "
                "ROC se conserva como diagnóstico, especialmente ampliada en la región de FPR bajo.",
                body,
            ),
            p(
                "Precision = TP/(TP+FP) &nbsp;&nbsp;&nbsp; Recall = TP/(TP+FN) "
                "&nbsp;&nbsp;&nbsp; FPR = FP/(FP+TN)<br/>"
                "AP = sum<sub>n</sub> (R<sub>n</sub> - R<sub>n-1</sub>) P<sub>n</sub>",
                formula,
            ),
            p(
                "Además de AP se reportan WR recuperadas, Recall@K, Precision@K y candidatas revisadas por cada WR recuperada. "
                "El ranking de la aplicación combina Recall@100, AP, Precision@100, Recall@50 y el punto operativo calibrado, "
                "con una penalización suave por FPR. Las brechas train-CV-holdout y un score multicomponente de sobreajuste "
                "se revisan aparte; un valor alto de AP no cancela una advertencia de inestabilidad.",
                body,
            ),
            img(figures["model"], 16.2),
            caption(
                "Figura 6. La grilla muestra un frente de compromiso, no un único ganador universal. Los holdouts cambian de denominador entre variantes."
            ),
            p(
                f"El perfil broad-shortlist relaxed/XGBoost/none recupera "
                f"{int(leading['holdout_wr_at_100'])}/{int(leading['wr_holdout'])} WR en top 100 "
                f"(Recall={leading['holdout_recall_at_100']:.1%}, Precision@100={leading['holdout_precision_at_100']:.1%}, "
                f"AP={leading['holdout_average_precision']:.3f}), pero mantiene una advertencia de sobreajuste. "
                f"El líder de AP relaxed/XGBoost/SMOTE alcanza AP={ap_leader['holdout_average_precision']:.3f} "
                f"con Recall@100={ap_leader['holdout_recall_at_100']:.1%}. "
                f"El perfil strict_poe_2/XGBoost/SMOTE está aceptado por los controles de estabilidad y alcanza "
                f"AP={stable['holdout_average_precision']:.3f}, Recall@100={stable['holdout_recall_at_100']:.1%}, "
                "aunque con una lista top-100 menos pura y un holdout WR más pequeño.",
                body,
            ),
            img(figures["curves"], 16.2),
            caption(
                "Figura 7. Curvas de tres modelos entrenados y evaluados sobre la misma variante relaxed_photometry."
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            p("5. Interpretación y segunda capa", h1),
            p(
                "La importancia interna del XGBoost broad-shortlist está dominada por W1-W2 y paralaje, seguida por colores 2MASS y Gaia. "
                "Esto es físicamente plausible como combinación de exceso infrarrojo y escala de brillo, pero no prueba causalidad. "
                "Antes de interpretar una feature como estable se requiere importancia por permutación en holdout y sensibilidad por subtipo.",
                body,
            ),
            img(figures["importance"], 14.5),
            caption(
                "Figura 8. Importancia del estimador para el perfil broad-shortlist. La suma está normalizada por el modelo."
            ),
            p(
                "La segunda capa entrena validadores one-class separados para WN y WC: Gaussian Mixture, one-class SVM, "
                "Isolation Forest y robust covariance. Su función es medir compatibilidad o reordenar candidatas del primer modelo, "
                "no imponer un descarte automático. El audit run_v3_second_layer mostró que los pares que remueven más negativos "
                "también pierden WR dentro del mismo presupuesto top-K; los pares de alta retención remueven demasiado poco. "
                "Por ahora la segunda capa queda como diagnóstico de revisión.",
                body,
            ),
            p(
                "La ampliación natural del proyecto es espectral. Gaia DR3 ya ofrece espectros BP/RP medios y productos RVS para subconjuntos "
                "de fuentes [2,3]. También hay información H-alpha en astrophysical_parameters. Estos campos se conservaron como opciones "
                "de enriquecimiento en la nueva adquisición, pero no se vuelven obligatorios hasta auditar cobertura homogénea en WR, "
                "negativos y prediction pool.",
                body,
            ),
            p("6. Prediction pool: decisión de reconstrucción", h1),
            p(
                "El primer pool de 58.037.788 filas aplicó un filtro agregado construido a partir de los loci de las variantes. "
                "La auditoría posterior demostró que ese agregado no era un superset geométrico de la unión exacta. "
                f"En 18 regiones Gaia, {_format_int(audit['compatible_exact_not_aggregate'])} de "
                f"{_format_int(audit['compatible_exact_union_keep'])} fuentes compatibles con alguna variante exacta "
                f"quedaban fuera ({audit['loss_fraction']:.2%}); la pérdida regional máxima fue "
                f"{audit['regional_max_loss_fraction']:.2%}. También se afectaron "
                f"{audit['known_wr_exact_not_current']} controles WR conocidos.",
                body,
            ),
            img(figures["pool"], 16.2),
            caption(
                "Figura 9. La auditoría motivó una nueva adquisición física inmutable y máscaras exactas locales por variante."
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            p("7. Alcance, limitaciones y siguiente entrega", h1),
            p(
                "<b>Alcance fotométrico.</b> El proyecto no intenta recuperar todas las estrellas del plano galáctico. "
                "Consulta todo el cielo, pero dentro de una envolvente amplia de colores observados en WR y luego aplica el locus exacto. "
                "El problema operativo es distinguir WR de contaminantes en esa región de solapamiento fotométrico.",
                bullet,
                bulletText="•",
            ),
            p(
                "<b>Etiquetas incompletas.</b> Las fuentes desconocidas de Gaia no son negativos confirmados. "
                "Por eso el pool no permite estimar todavía precisión científica real; produce candidatos para revisión.",
                bullet,
                bulletText="•",
            ),
            p(
                "<b>Sesgo de la muestra negativa.</b> SIMBAD representa clases seleccionadas y fuentes previamente catalogadas. "
                "El threshold_calibration mejora el stress test, pero no reproduce toda la distribución de Gaia.",
                bullet,
                bulletText="•",
            ),
            p(
                "<b>Dependencia de crossmatch y cobertura.</b> La referencia admite fallback VizieR; el pool usa las rutas de crossmatch de Gaia. "
                "Las diferencias de procedencia se mantienen explícitas y se deben auditar al agregar espectros.",
                bullet,
                bulletText="•",
            ),
            p(
                "<b>Modelo operativo aún no congelado.</b> Los perfiles líderes muestran compromisos entre AP, recuperación, pureza y estabilidad. "
                "La selección final debe fijar un presupuesto de seguimiento y repetirse sobre el pool exact-union terminado.",
                bullet,
                bulletText="•",
            ),
            p("Siguiente secuencia", h2),
            p(
                "1. Terminar y auditar el build exact-union.<br/>"
                "2. Seleccionar explícitamente uno o más result_id según presupuesto top-K y estabilidad.<br/>"
                "3. Ejecutar scoring por tile y auditar hashes, cobertura y features faltantes.<br/>"
                "4. Revisar la lista corta con tipos SIMBAD, subtipos WN/WC y compatibilidad de segunda capa.<br/>"
                "5. Incorporar Gaia XP/H-alpha como enriquecimiento opcional y medir la mejora a presupuesto fijo.<br/>"
                "6. Confirmar candidatas mediante espectroscopía.",
                body,
            ),
            p(
                "La prediction pool completa todavía no forma parte de esta entrega. El código de adquisición, resume, bitmask y scoring ya existe; "
                "la limitación actual es terminar la consulta masiva de Gaia y aprobar su auditoría final.",
                callout,
            ),
            p("Referencias", h1),
            p(
                '[1] Rosslowe, C. K. & Crowther, P. A. (2015). "Spatial distribution of Galactic Wolf-Rayet stars and implications for the global population". '
                'MNRAS 447, 2322-2347. <link href="https://doi.org/10.1093/mnras/stu2525" color="#3568a8">doi:10.1093/mnras/stu2525</link>.',
                reference_style,
            ),
            p(
                f'[2] Crowther, P. A. Galactic Wolf Rayet Catalogue, snapshot local {evidence.catalogue["version"]}. '
                '<link href="https://pacrowther.staff.shef.ac.uk/WRcat/" color="#3568a8">pacrowther.staff.shef.ac.uk/WRcat</link>.',
                reference_style,
            ),
            p(
                '[3] Gaia Collaboration, Vallenari, A. et al. (2023). "Gaia Data Release 3: Summary of the content and survey properties". '
                'A&A 674, A1. <link href="https://doi.org/10.1051/0004-6361/202243940" color="#3568a8">doi:10.1051/0004-6361/202243940</link>.',
                reference_style,
            ),
            p(
                '[4] Saito, T. & Rehmsmeier, M. (2015). "The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets". '
                'PLOS ONE 10(3), e0118432. <link href="https://doi.org/10.1371/journal.pone.0118432" color="#3568a8">doi:10.1371/journal.pone.0118432</link>.',
                reference_style,
            ),
            p(
                '[5] Skrutskie, M. F. et al. (2006). "The Two Micron All Sky Survey (2MASS)". AJ 131, 1163-1183. '
                '<link href="https://doi.org/10.1086/498708" color="#3568a8">doi:10.1086/498708</link>.',
                reference_style,
            ),
            p(
                '[6] Wright, E. L. et al. (2010). "The Wide-field Infrared Survey Explorer (WISE): Mission Description and Initial On-orbit Performance". '
                'AJ 140, 1868-1881. <link href="https://doi.org/10.1088/0004-6256/140/6/1868" color="#3568a8">doi:10.1088/0004-6256/140/6/1868</link>.',
                reference_style,
            ),
        ]
    )

    doc.build(story)


def generate_project_summary(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    run_id: str = DEFAULT_RUN_ID,
    build_reportlab_pdf: bool = True,
) -> dict[str, str]:
    output_dir = output_dir.resolve()
    figures_dir = output_dir / "figures"
    evidence = collect_evidence(run_id=run_id)
    figures = generate_figures(evidence, figures_dir)
    snapshot_path, chart_map = write_source_notes(evidence, figures, output_dir)
    result = {
        "evidence": str(snapshot_path),
        "chart_map": str(chart_map),
        "figures_dir": str(figures_dir),
    }
    if build_reportlab_pdf:
        pdf_path = output_dir / "InformeTecnico.pdf"
        build_pdf(evidence, figures, pdf_path)
        result["pdf"] = str(pdf_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for figures, evidence and the optional ReportLab PDF.",
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Regenerate figures and evidence without the legacy ReportLab PDF.",
    )
    args = parser.parse_args()
    result = generate_project_summary(
        args.output_dir,
        run_id=args.run_id,
        build_reportlab_pdf=not args.figures_only,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
