"""Prepared, auditable data for the Case Review visualizations.

This module owns the scientific and data-reduction semantics used by the
interactive charts.  Streamlit pages only choose settings and render the
already-prepared frames.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from typing import Literal

import duckdb
import numpy as np
import pandas as pd

from wr_detector.config import resolve_path
from wr_detector.modeling.cases import (
    DIAGNOSTIC_STATES,
    add_case_spatial_coordinates,
    classify_cases,
    reference_db_paths,
)
from wr_detector.modeling.explorer import explorer_db_path


CASE_PLOT_PIPELINE_VERSION = "case-review-v2"
BACKGROUND_STATES = frozenset({"Background", "True negative"})
PLOT_KINDS = ("photometric", "mollweide", "galactic_plane")


@dataclass(frozen=True)
class CasePlotSettings:
    """All controls that can alter a prepared Case Review visualization."""

    diagnostic_basis: str = "review_budget"
    top_k: int = 100
    threshold: float | None = None
    visible_states: tuple[str, ...] = tuple(DIAGNOSTIC_STATES["review_budget"])
    x_axis: str = "BP_RP"
    y_axis: str = "G"
    color_color: bool = False
    min_parallax_over_error: float = 2.0
    max_distance_kpc: float = 15.0
    background_mode: Literal["density", "sample"] = "density"
    background_limit: int = 5_000
    random_seed: int = 42

    def cache_payload(self) -> dict[str, object]:
        return {**asdict(self), "pipeline_version": CASE_PLOT_PIPELINE_VERSION}


@dataclass(frozen=True)
class CaseDataFingerprint:
    """Stable cache identity for predictions, artifacts, and reference DBs."""

    value: str
    payload: dict[str, object]


@dataclass
class CasePlotFrame:
    """Full and interactive representations for one chart."""

    kind: str
    x: str
    y: str
    full: pd.DataFrame
    relevant: pd.DataFrame
    interactive_background: pd.DataFrame
    density: pd.DataFrame
    prepared_source_count: int
    background_source_count: int
    relevant_source_count: int
    interactive_source_count: int
    interactive_mark_count: int


@dataclass
class CasePlotData:
    """Prepared plot frames plus timing and provenance."""

    fingerprint: str
    settings: CasePlotSettings
    classified: pd.DataFrame
    frames: dict[str, CasePlotFrame]
    preparation_seconds: float

    def frame(self, kind: str) -> CasePlotFrame:
        if kind not in self.frames:
            raise KeyError(f"Unsupported Case Review plot kind: {kind}")
        return self.frames[kind]


@dataclass(frozen=True)
class CasePlotExportResult:
    """Audit metadata returned by a PNG export."""

    output_path: Path
    kind: str
    dpi: int
    prepared_source_count: int
    interactive_source_count: int
    interactive_mark_count: int
    png_source_count: int


def case_data_fingerprint(
    config_path: str | Path,
    *,
    run_id: str,
    result_id: str,
    split: str,
) -> CaseDataFingerprint:
    """Fingerprint every source that can make cached case rows stale."""
    history_path = explorer_db_path(config_path).resolve()
    payload: dict[str, object] = {
        "pipeline_version": CASE_PLOT_PIPELINE_VERSION,
        "run_id": str(run_id),
        "result_id": str(result_id),
        "split": str(split),
        "history_db": _file_revision(history_path),
        "reference_dbs": {
            key: _file_revision(path.resolve()) if path is not None else None
            for key, path in reference_db_paths(config_path).items()
        },
        "config": _file_revision(resolve_path(config_path).resolve()),
    }
    if history_path.exists():
        with duckdb.connect(str(history_path), read_only=True) as con:
            payload["result"] = _result_revision(
                con,
                run_id=str(run_id),
                result_id=str(result_id),
            )
            payload["predictions"] = _prediction_revision(
                con,
                run_id=str(run_id),
                result_id=str(result_id),
                split=str(split),
            )
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")
    return CaseDataFingerprint(sha256(encoded).hexdigest(), payload)


def prepare_case_plot_data(
    cases: pd.DataFrame,
    *,
    settings: CasePlotSettings,
    fingerprint: str,
) -> CasePlotData:
    """Prepare complete and compact interactive frames for all three charts."""
    started = perf_counter()
    classified = classify_cases(
        cases,
        basis=settings.diagnostic_basis,
        top_k=settings.top_k,
    )
    visible = set(settings.visible_states)
    classified = classified[
        classified["diagnostic_state"].astype(str).isin(visible)
    ].copy()
    spatial = add_case_spatial_coordinates(
        classified,
        min_parallax_over_error=settings.min_parallax_over_error,
        max_distance_kpc=settings.max_distance_kpc,
    )

    frame_specs = {
        "photometric": (settings.x_axis, settings.y_axis, None),
        "mollweide": ("mollweide_x", "mollweide_y", None),
        "galactic_plane": (
            "galactocentric_x_kpc",
            "galactocentric_y_kpc",
            spatial.get("distance_plotted"),
        ),
    }
    frames: dict[str, CasePlotFrame] = {}
    for kind, (x, y, eligibility) in frame_specs.items():
        frames[kind] = _prepare_frame(
            spatial,
            kind=kind,
            x=x,
            y=y,
            eligibility=eligibility,
            settings=settings,
        )
    return CasePlotData(
        fingerprint=fingerprint,
        settings=settings,
        classified=spatial,
        frames=frames,
        preparation_seconds=perf_counter() - started,
    )


def mollweide_project(
    longitude_deg: pd.Series | np.ndarray | list[float],
    latitude_deg: pd.Series | np.ndarray | list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Project Galactic longitude/latitude with the standard Mollweide formula.

    Galactic longitude zero is centered and positive longitude grows to the
    left, following the common astronomical sky-map convention.
    """
    longitude = np.asarray(longitude_deg, dtype=float)
    latitude = np.asarray(latitude_deg, dtype=float)
    wrapped = (longitude + 180.0) % 360.0 - 180.0
    lam = -np.deg2rad(wrapped)
    phi = np.deg2rad(np.clip(latitude, -90.0, 90.0))
    theta = phi.copy()
    poles = np.isclose(np.abs(phi), np.pi / 2.0)
    theta[poles] = np.sign(phi[poles]) * np.pi / 2.0
    active = np.isfinite(theta) & ~poles
    for _ in range(12):
        current = theta[active]
        numerator = 2.0 * current + np.sin(2.0 * current) - np.pi * np.sin(phi[active])
        denominator = 2.0 + 2.0 * np.cos(2.0 * current)
        step = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=np.abs(denominator) > 1e-12,
        )
        theta[active] = current - step
        if step.size == 0 or np.nanmax(np.abs(step)) < 1e-12:
            break
    x = 2.0 * np.sqrt(2.0) / np.pi * lam * np.cos(theta)
    y = np.sqrt(2.0) * np.sin(theta)
    x[~np.isfinite(longitude) | ~np.isfinite(latitude)] = np.nan
    y[~np.isfinite(longitude) | ~np.isfinite(latitude)] = np.nan
    return x, y


def _prepare_frame(
    data: pd.DataFrame,
    *,
    kind: str,
    x: str,
    y: str,
    eligibility: pd.Series | None,
    settings: CasePlotSettings,
) -> CasePlotFrame:
    columns = _plot_columns(data, x=x, y=y, kind=kind)
    frame = data[columns].copy()
    mask = frame[x].notna() & frame[y].notna()
    if eligibility is not None:
        mask &= eligibility.reindex(frame.index).fillna(False).astype(bool)
    frame = frame.loc[mask].reset_index(drop=True)
    background_mask = frame["diagnostic_state"].isin(BACKGROUND_STATES)
    background = frame.loc[background_mask].copy()
    relevant = frame.loc[~background_mask].copy()

    if settings.background_mode == "sample":
        interactive_background = _sample_background(
            background,
            limit=settings.background_limit,
            seed=settings.random_seed,
        )
        density = pd.DataFrame()
        represented_background = len(interactive_background)
        background_marks = len(interactive_background)
    else:
        interactive_background = pd.DataFrame(columns=frame.columns)
        density = _density_frame(background, kind=kind, x=x, y=y)
        represented_background = len(background)
        background_marks = len(density)

    return CasePlotFrame(
        kind=kind,
        x=x,
        y=y,
        full=frame,
        relevant=relevant.reset_index(drop=True),
        interactive_background=interactive_background.reset_index(drop=True),
        density=density.reset_index(drop=True),
        prepared_source_count=len(frame),
        background_source_count=len(background),
        relevant_source_count=len(relevant),
        interactive_source_count=len(relevant) + represented_background,
        interactive_mark_count=len(relevant) + background_marks,
    )


def _sample_background(
    background: pd.DataFrame,
    *,
    limit: int,
    seed: int,
) -> pd.DataFrame:
    if len(background) <= limit:
        return background.copy()
    sampled = background.sample(n=int(limit), random_state=int(seed))
    sort_columns = [column for column in ["source_id", "rank"] if column in sampled]
    return sampled.sort_values(sort_columns).reset_index(drop=True) if sort_columns else sampled


def _density_frame(
    background: pd.DataFrame,
    *,
    kind: str,
    x: str,
    y: str,
) -> pd.DataFrame:
    columns = [
        "x0",
        "x1",
        "y0",
        "y1",
        "x",
        "y",
        "count",
        "opacity",
        "diagnostic_state",
    ]
    if background.empty:
        return pd.DataFrame(columns=columns)
    bins = {
        "photometric": (72, 48),
        "mollweide": (72, 36),
        "galactic_plane": (60, 60),
    }[kind]
    values = background[[x, y]].to_numpy(dtype=float)
    finite = np.isfinite(values).all(axis=1)
    values = values[finite]
    if not len(values):
        return pd.DataFrame(columns=columns)
    x_range = _padded_range(values[:, 0], fixed=(-2.86, 2.86) if kind == "mollweide" else None)
    y_range = _padded_range(values[:, 1], fixed=(-1.44, 1.44) if kind == "mollweide" else None)
    counts, x_edges, y_edges = np.histogram2d(
        values[:, 0],
        values[:, 1],
        bins=bins,
        range=[x_range, y_range],
    )
    xi, yi = np.nonzero(counts)
    observed = counts[xi, yi]
    max_log = float(np.log1p(observed).max()) if len(observed) else 1.0
    state = (
        str(background["diagnostic_state"].mode().iloc[0])
        if "diagnostic_state" in background and not background.empty
        else "Background"
    )
    return pd.DataFrame(
        {
            "x0": x_edges[xi],
            "x1": x_edges[xi + 1],
            "y0": y_edges[yi],
            "y1": y_edges[yi + 1],
            "x": (x_edges[xi] + x_edges[xi + 1]) / 2.0,
            "y": (y_edges[yi] + y_edges[yi + 1]) / 2.0,
            "count": observed.astype(int),
            "opacity": 0.05 + 0.20 * np.log1p(observed) / max(max_log, 1e-9),
            "diagnostic_state": state,
        }
    )


def _padded_range(
    values: np.ndarray,
    *,
    fixed: tuple[float, float] | None = None,
) -> tuple[float, float]:
    if fixed is not None:
        return fixed
    low, high = np.nanquantile(values, [0.002, 0.998])
    if not np.isfinite(low) or not np.isfinite(high):
        low, high = np.nanmin(values), np.nanmax(values)
    if np.isclose(low, high):
        low -= 0.5
        high += 0.5
    pad = (high - low) * 0.03
    return float(low - pad), float(high + pad)


def _plot_columns(data: pd.DataFrame, *, x: str, y: str, kind: str) -> list[str]:
    desired = [
        "source_id",
        "target",
        "diagnostic_state",
        "object_name",
        "rank",
        "score",
        "threshold",
        "simbad_main_type",
        "spectral_type",
        "galactic_l",
        "galactic_b",
        "distance_kpc",
        x,
        y,
    ]
    if kind == "galactic_plane":
        desired.append("distance_plotted")
    return list(dict.fromkeys(column for column in desired if column in data.columns))


def _file_revision(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _result_revision(
    con: duckdb.DuckDBPyConnection,
    *,
    run_id: str,
    result_id: str,
) -> dict[str, object]:
    if not _table_exists(con, "model_results"):
        return {}
    result = con.execute(
        "SELECT * FROM model_results WHERE run_id = ? AND result_id = ?",
        [run_id, result_id],
    ).fetchdf()
    if result.empty:
        return {}
    keep = [
        "run_id",
        "result_id",
        "imported_at",
        "dataset_variant",
        "feature_set",
        "model",
        "sampler",
        "selected_threshold",
        "model_path",
        "model_sha256",
        "dataset_sha256",
        "models_config_sha256",
        "paths_config_sha256",
        "filters_config_sha256",
        "code_git_commit",
        "code_worktree_sha256",
    ]
    row = result.iloc[0]
    payload = {column: row[column] for column in keep if column in result.columns}
    model_path = payload.get("model_path")
    if model_path:
        payload["model_file"] = _file_revision(resolve_path(str(model_path)).resolve())
    return payload


def _prediction_revision(
    con: duckdb.DuckDBPyConnection,
    *,
    run_id: str,
    result_id: str,
    split: str,
) -> dict[str, object]:
    if not _table_exists(con, "model_predictions"):
        return {}
    columns = {
        row[0]
        for row in con.execute("DESCRIBE model_predictions").fetchall()
    }
    hashed = [
        column
        for column in [
            "row_id",
            "source_id",
            "target",
            "score",
            "predicted",
            "threshold",
            "dataset_variant",
            "feature_set",
            "model",
            "sampler",
        ]
        if column in columns
    ]
    where = []
    params: list[object] = []
    for column, value in [
        ("run_id", run_id),
        ("result_id", result_id),
        ("split", split),
    ]:
        if column in columns:
            where.append(f"{column} = ?")
            params.append(value)
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""
    hash_sql = (
        f"CAST(COALESCE(bit_xor(hash({', '.join(hashed)})), 0) AS VARCHAR)"
        if hashed
        else "'0'"
    )
    row = con.execute(
        f"""
        SELECT
            COUNT(*) AS row_count,
            {hash_sql} AS row_hash
        FROM model_predictions
        {where_sql}
        """,
        params,
    ).fetchone()
    return {"row_count": int(row[0]), "row_hash": str(row[1])}


def _table_exists(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    return (
        con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
            [table],
        ).fetchone()[0]
        > 0
    )


def _json_default(value: object) -> object:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.integer, np.floating)):
        if pd.isna(value):
            return None
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    return str(value)
