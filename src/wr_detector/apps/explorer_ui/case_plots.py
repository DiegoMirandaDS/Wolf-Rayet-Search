"""Optimized interactive and static plots for Case Review."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import altair as alt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from wr_detector.modeling.case_visualization import (
    BACKGROUND_STATES,
    CasePlotData,
    CasePlotExportResult,
    CasePlotFrame,
    mollweide_project,
)


PlotKind = Literal["photometric", "mollweide", "galactic_plane"]

DIAGNOSTIC_ORDER = [
    "Background",
    "True negative",
    "Contaminant @K",
    "False positive",
    "WR outside @K",
    "False negative",
    "WR recovered @K",
    "True positive",
]
DIAGNOSTIC_COLORS = {
    "Background": "#4e79a7",
    "True negative": "#4e79a7",
    "Contaminant @K": "#ff79c6",
    "False positive": "#ff79c6",
    "WR outside @K": "#edc948",
    "False negative": "#edc948",
    "WR recovered @K": "#f28e2b",
    "True positive": "#f28e2b",
}
DIAGNOSTIC_SHAPES = {
    "Background": "circle",
    "True negative": "circle",
    "Contaminant @K": "triangle-up",
    "False positive": "triangle-up",
    "WR outside @K": "diamond",
    "False negative": "diamond",
    "WR recovered @K": "circle",
    "True positive": "circle",
}
DIAGNOSTIC_MARKS = {
    "Background": {"size": 14, "opacity": 0.16, "filled": True},
    "True negative": {"size": 14, "opacity": 0.16, "filled": True},
    "Contaminant @K": {"size": 54, "opacity": 0.82, "filled": True},
    "False positive": {"size": 54, "opacity": 0.82, "filled": True},
    "WR outside @K": {"size": 74, "opacity": 0.95, "filled": False},
    "False negative": {"size": 74, "opacity": 0.95, "filled": False},
    "WR recovered @K": {"size": 82, "opacity": 0.96, "filled": True},
    "True positive": {"size": 82, "opacity": 0.96, "filled": True},
}
BACKGROUND_BLUE = "#4e79a7"
SELECTION_OUTER = "#f8f8f2"
SELECTION_INNER = "#e15759"
CHART_HEIGHTS = {
    "photometric": 520,
    "mollweide": 400,
    "galactic_plane": 400,
}


def build_case_plot_base(kind: PlotKind, data: CasePlotData) -> alt.LayerChart | None:
    """Build the immutable heavy layers for one Case Review chart."""
    frame = data.frame(kind)
    if frame.full.empty:
        return None
    if kind == "photometric":
        layers = _photometric_layers(frame, color_color=data.settings.color_color)
    elif kind == "mollweide":
        layers = _mollweide_layers(frame)
    elif kind == "galactic_plane":
        layers = _galactic_plane_layers(frame)
    else:
        raise ValueError(f"Unsupported Case Review plot kind: {kind}")
    return alt.layer(*layers).properties(
        width="container",
        height=CHART_HEIGHTS[kind],
    )


def build_selected_source_layer(
    kind: PlotKind,
    data: CasePlotData,
    selected_source_id: int | None,
) -> list[alt.Chart]:
    """Build only the two one-row outlines used for the active source."""
    if selected_source_id is None:
        return []
    frame = data.frame(kind)
    if "source_id" not in frame.full:
        return []
    selected = frame.full[frame.full["source_id"].eq(selected_source_id)]
    if selected.empty:
        return []
    selected = selected.head(1)
    x_scale, y_scale = _scales(kind, frame, color_color=data.settings.color_color)
    tooltip = _case_tooltip(selected, frame.x, frame.y)
    encodings = {
        "x": alt.X(f"{frame.x}:Q", scale=x_scale),
        "y": alt.Y(f"{frame.y}:Q", scale=y_scale),
        "tooltip": tooltip,
    }
    outer = (
        alt.Chart(selected)
        .mark_point(
            shape="circle",
            size=280,
            filled=False,
            color=SELECTION_OUTER,
            strokeWidth=3.0,
        )
        .encode(**encodings)
    )
    inner = (
        alt.Chart(selected)
        .mark_point(
            shape="circle",
            size=205,
            filled=False,
            color=SELECTION_INNER,
            strokeWidth=1.5,
        )
        .encode(**encodings)
    )
    return [outer, inner]


def build_interactive_case_plot(
    kind: PlotKind,
    data: CasePlotData,
    *,
    selected_source_id: int | None = None,
    base_chart: alt.LayerChart | None = None,
) -> alt.LayerChart | None:
    """Compose a cached base with the cheap selected-source overlay."""
    base = base_chart or build_case_plot_base(kind, data)
    if base is None:
        return None
    selection = build_selected_source_layer(kind, data, selected_source_id)
    chart = alt.layer(*base.layer, *selection).properties(
        width="container",
        height=CHART_HEIGHTS[kind],
    ).resolve_scale(color="independent", shape="independent")
    if kind in {"photometric", "galactic_plane"}:
        chart = chart.interactive()
    return (
        chart.configure_axis(
            labelFontSize=12,
            titleFontSize=12,
            labelColor="#d8d8dc",
            titleColor="#f8f8f2",
            gridColor="#555866",
            gridOpacity=0.22,
        )
        .configure_legend(
            orient="bottom",
            direction="horizontal",
            columns=4,
            labelFontSize=12,
            titleFontSize=12,
            labelLimit=170,
            symbolSize=75,
        )
        .configure_view(stroke=None)
    )


def chart_payload_bytes(chart: alt.TopLevelMixin | None) -> int:
    """Return the serialized Vega-Lite JSON size before Streamlit transport."""
    if chart is None:
        return 0
    payload = json.dumps(
        chart.to_dict(),
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return len(payload)


def selected_layer_payload_bytes(
    kind: PlotKind,
    data: CasePlotData,
    selected_source_id: int | None,
) -> int:
    """Serialize only the one-row overlay for selection telemetry."""
    layers = build_selected_source_layer(kind, data, selected_source_id)
    return chart_payload_bytes(alt.layer(*layers)) if layers else 0


def export_case_plot_png(
    kind: PlotKind,
    data: CasePlotData,
    output_path: str | Path,
    *,
    dpi: int = 300,
    selected_source_id: int | None = None,
) -> CasePlotExportResult:
    """Export one chart from every filtered source, never the interactive cap."""
    if not 200 <= int(dpi) <= 400:
        raise ValueError("dpi must be between 200 and 400")
    frame = data.frame(kind)
    if frame.full.empty:
        raise ValueError(f"No eligible sources for {kind}")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if kind == "photometric":
        figure = Figure(figsize=(12, 6.4), facecolor="#282a36")
    elif kind == "mollweide":
        figure = Figure(figsize=(12, 6), facecolor="#282a36")
    else:
        figure = Figure(figsize=(8, 8), facecolor="#282a36")
    axis = figure.add_subplot(111)
    _style_matplotlib_axis(axis)
    if kind == "photometric":
        _export_photometric(axis, frame, color_color=data.settings.color_color)
    elif kind == "mollweide":
        _export_mollweide(axis, frame)
    else:
        _export_galactic_plane(axis, frame)
    _export_relevant_points(axis, frame)
    if selected_source_id is not None:
        _export_selected(axis, frame, selected_source_id)
    figure.tight_layout()
    FigureCanvasAgg(figure)
    figure.savefig(output, dpi=int(dpi), facecolor=figure.get_facecolor())
    return CasePlotExportResult(
        output_path=output,
        kind=kind,
        dpi=int(dpi),
        prepared_source_count=frame.prepared_source_count,
        interactive_source_count=frame.interactive_source_count,
        interactive_mark_count=frame.interactive_mark_count,
        png_source_count=len(frame.full),
    )


def plot_count_summary(frame: CasePlotFrame) -> str:
    if frame.density.empty:
        background = (
            f"{len(frame.interactive_background):,}/{frame.background_source_count:,} "
            "background points"
        )
    else:
        background = (
            f"{frame.background_source_count:,} background sources in "
            f"{len(frame.density):,} density cells"
        )
    return (
        f"{frame.prepared_source_count:,} eligible; "
        f"{frame.relevant_source_count:,} diagnostic cases; {background}."
    )


def _photometric_layers(
    frame: CasePlotFrame,
    *,
    color_color: bool,
) -> list[alt.Chart]:
    x_scale, y_scale = _scales("photometric", frame, color_color=color_color)
    x_axis = alt.Axis(title=_axis_label(frame.x), labelFontSize=12, titleFontSize=12)
    y_axis = alt.Axis(title=_axis_label(frame.y), labelFontSize=12, titleFontSize=12)
    layers = _background_layers(
        frame,
        x_scale=x_scale,
        y_scale=y_scale,
        x_axis=x_axis,
        y_axis=y_axis,
        density_rects=True,
    )
    layers.extend(
        _relevant_layers(
            frame,
            x_scale=x_scale,
            y_scale=y_scale,
            x_axis=x_axis,
            y_axis=y_axis,
        )
    )
    layers.append(_legend_layer(frame))
    return layers


def _mollweide_layers(frame: CasePlotFrame) -> list[alt.Chart]:
    x_scale, y_scale = _scales("mollweide", frame)
    hidden = _hidden_axis()
    boundary, grid, labels = _mollweide_guides(x_scale=x_scale, y_scale=y_scale)
    layers: list[alt.Chart] = [boundary, grid, labels]
    layers.extend(
        _background_layers(
            frame,
            x_scale=x_scale,
            y_scale=y_scale,
            x_axis=hidden,
            y_axis=hidden,
            density_rects=False,
        )
    )
    layers.extend(
        _relevant_layers(
            frame,
            x_scale=x_scale,
            y_scale=y_scale,
            x_axis=hidden,
            y_axis=hidden,
        )
    )
    layers.append(_legend_layer(frame))
    return layers


def _galactic_plane_layers(frame: CasePlotFrame) -> list[alt.Chart]:
    x_scale, y_scale = _scales("galactic_plane", frame)
    axis_x = alt.Axis(title="Galactocentric X (kpc)", labelFontSize=12, titleFontSize=12)
    axis_y = alt.Axis(title="Galactocentric Y (kpc)", labelFontSize=12, titleFontSize=12)
    rings, arms, landmarks = _galaxy_context(
        x_scale=x_scale,
        y_scale=y_scale,
        x_axis=axis_x,
        y_axis=axis_y,
    )
    layers: list[alt.Chart] = [rings, arms, landmarks]
    layers.extend(
        _background_layers(
            frame,
            x_scale=x_scale,
            y_scale=y_scale,
            x_axis=axis_x,
            y_axis=axis_y,
            density_rects=True,
        )
    )
    layers.extend(
        _relevant_layers(
            frame,
            x_scale=x_scale,
            y_scale=y_scale,
            x_axis=axis_x,
            y_axis=axis_y,
        )
    )
    layers.append(_legend_layer(frame))
    return layers


def _background_layers(
    frame: CasePlotFrame,
    *,
    x_scale: alt.Scale,
    y_scale: alt.Scale,
    x_axis: alt.Axis,
    y_axis: alt.Axis,
    density_rects: bool,
) -> list[alt.Chart]:
    if not frame.density.empty:
        chart = alt.Chart(frame.density)
        if density_rects:
            mark = chart.mark_rect(color=BACKGROUND_BLUE)
            encoded = mark.encode(
                x=alt.X("x0:Q", scale=x_scale, axis=x_axis),
                x2="x1:Q",
                y=alt.Y("y0:Q", scale=y_scale, axis=y_axis),
                y2="y1:Q",
                opacity=alt.Opacity(
                    "opacity:Q",
                    scale=None,
                    legend=None,
                ),
                tooltip=[alt.Tooltip("count:Q", title="background sources")],
            )
        else:
            encoded = chart.mark_square(
                color=BACKGROUND_BLUE,
                size=32,
            ).encode(
                x=alt.X("x:Q", scale=x_scale, axis=x_axis),
                y=alt.Y("y:Q", scale=y_scale, axis=y_axis),
                opacity=alt.Opacity("opacity:Q", scale=None, legend=None),
                tooltip=[alt.Tooltip("count:Q", title="background sources")],
            )
        return [encoded]
    if frame.interactive_background.empty:
        return []
    tooltip = [
        alt.Tooltip("object_name:N", title="object"),
        alt.Tooltip("rank:Q", title="rank"),
        alt.Tooltip("score:Q", title="score", format=".4f"),
    ]
    return [
        alt.Chart(frame.interactive_background)
        .mark_point(
            shape="circle",
            size=14,
            opacity=0.15,
            filled=True,
            color=BACKGROUND_BLUE,
        )
        .encode(
            x=alt.X(f"{frame.x}:Q", scale=x_scale, axis=x_axis),
            y=alt.Y(f"{frame.y}:Q", scale=y_scale, axis=y_axis),
            tooltip=tooltip,
        )
    ]


def _relevant_layers(
    frame: CasePlotFrame,
    *,
    x_scale: alt.Scale,
    y_scale: alt.Scale,
    x_axis: alt.Axis,
    y_axis: alt.Axis,
) -> list[alt.Chart]:
    present = set(frame.relevant["diagnostic_state"].dropna().astype(str))
    tooltip = _case_tooltip(frame.relevant, frame.x, frame.y)
    layers: list[alt.Chart] = []
    for state in DIAGNOSTIC_ORDER:
        if state not in present:
            continue
        style = DIAGNOSTIC_MARKS[state]
        subset = frame.relevant[frame.relevant["diagnostic_state"].eq(state)]
        layers.append(
            alt.Chart(subset)
            .mark_point(
                shape=DIAGNOSTIC_SHAPES[state],
                size=style["size"],
                opacity=style["opacity"],
                filled=style["filled"],
                fill=DIAGNOSTIC_COLORS[state] if style["filled"] else None,
                stroke=DIAGNOSTIC_COLORS[state],
                strokeWidth=1.7 if not style["filled"] else 0.7,
            )
            .encode(
                x=alt.X(f"{frame.x}:Q", scale=x_scale, axis=x_axis),
                y=alt.Y(f"{frame.y}:Q", scale=y_scale, axis=y_axis),
                tooltip=tooltip,
            )
        )
    return layers


def _legend_layer(frame: CasePlotFrame) -> alt.Chart:
    present = set(frame.full["diagnostic_state"].dropna().astype(str))
    states = [state for state in DIAGNOSTIC_ORDER if state in present]
    legend_data = pd.DataFrame({"diagnostic_state": states})
    return (
        alt.Chart(legend_data)
        .mark_point(opacity=0)
        .encode(
            color=alt.Color(
                "diagnostic_state:N",
                scale=alt.Scale(
                    domain=states,
                    range=[DIAGNOSTIC_COLORS[state] for state in states],
                ),
                legend=alt.Legend(
                    title=None,
                    orient="bottom",
                    direction="horizontal",
                    columns=4,
                    labelFontSize=12,
                    symbolSize=75,
                ),
            ),
            shape=alt.Shape(
                "diagnostic_state:N",
                scale=alt.Scale(
                    domain=states,
                    range=[DIAGNOSTIC_SHAPES[state] for state in states],
                ),
                legend=alt.Legend(
                    title=None,
                    orient="bottom",
                    direction="horizontal",
                    columns=4,
                    labelFontSize=12,
                    symbolSize=75,
                ),
            ),
        )
    )


def _mollweide_guides(
    *,
    x_scale: alt.Scale,
    y_scale: alt.Scale,
) -> tuple[alt.Chart, alt.Chart, alt.Chart]:
    boundary = pd.DataFrame(
        {
            "x": 2.0 * np.sqrt(2.0) * np.cos(np.linspace(0, 2 * np.pi, 361)),
            "y": np.sqrt(2.0) * np.sin(np.linspace(0, 2 * np.pi, 361)),
            "order": range(361),
        }
    )
    boundary_chart = (
        alt.Chart(boundary)
        .mark_line(color="#888b98", opacity=0.48, strokeWidth=0.9)
        .encode(
            x=alt.X("x:Q", scale=x_scale, axis=_hidden_axis()),
            y=alt.Y("y:Q", scale=y_scale, axis=_hidden_axis()),
            order="order:Q",
        )
    )
    rows: list[dict[str, object]] = []
    for longitude in [-135, -90, -45, 0, 45, 90, 135]:
        latitude = np.linspace(-89.5, 89.5, 181)
        x, y = mollweide_project(
            np.full_like(latitude, longitude, dtype=float),
            latitude,
        )
        rows.extend(
            {
                "line": f"l={longitude}",
                "x": x_value,
                "y": y_value,
                "order": order,
            }
            for order, (x_value, y_value) in enumerate(zip(x, y, strict=True))
        )
    for latitude in [-60, -30, 0, 30, 60]:
        longitude = np.linspace(-179.5, 179.5, 361)
        x, y = mollweide_project(
            longitude,
            np.full_like(longitude, latitude, dtype=float),
        )
        rows.extend(
            {
                "line": f"b={latitude}",
                "x": x_value,
                "y": y_value,
                "order": order,
            }
            for order, (x_value, y_value) in enumerate(zip(x, y, strict=True))
        )
    grid = (
        alt.Chart(pd.DataFrame(rows))
        .mark_line(color="#888b98", opacity=0.20, strokeWidth=0.7)
        .encode(
            x=alt.X("x:Q", scale=x_scale, axis=_hidden_axis()),
            y=alt.Y("y:Q", scale=y_scale, axis=_hidden_axis()),
            detail="line:N",
            order="order:Q",
        )
    )
    label_l = [135, 90, 45, 0, 315, 270, 225]
    x, y = mollweide_project(label_l, [0] * len(label_l))
    labels = pd.DataFrame(
        {
            "x": x,
            "y": np.full_like(y, -1.37),
            "label": [f"{value}°" for value in label_l],
        }
    )
    label_chart = (
        alt.Chart(labels)
        .mark_text(color="#b9bac3", fontSize=12, baseline="top")
        .encode(
            x=alt.X("x:Q", scale=x_scale, axis=_hidden_axis()),
            y=alt.Y("y:Q", scale=y_scale, axis=_hidden_axis()),
            text="label:N",
        )
    )
    return boundary_chart, grid, label_chart


def _galaxy_context(
    *,
    x_scale: alt.Scale,
    y_scale: alt.Scale,
    x_axis: alt.Axis,
    y_axis: alt.Axis,
) -> tuple[alt.Chart, alt.Chart, alt.Chart]:
    rings: list[dict[str, float]] = []
    for radius in [4.0, 8.0, 12.0, 16.0]:
        for order, angle in enumerate(np.linspace(0, 2 * np.pi, 181)):
            rings.append(
                {
                    "ring": radius,
                    "x": radius * np.cos(angle),
                    "y": radius * np.sin(angle),
                    "order": order,
                }
            )
    ring_chart = (
        alt.Chart(pd.DataFrame(rings))
        .mark_line(color="#6272a4", opacity=0.15, strokeWidth=0.75)
        .encode(
            x=alt.X("x:Q", scale=x_scale, axis=x_axis),
            y=alt.Y("y:Q", scale=y_scale, axis=y_axis),
            detail="ring:N",
            order="order:Q",
        )
    )
    arms: list[dict[str, float | int]] = []
    pitch = np.deg2rad(18.5)
    for arm in range(4):
        for order, radius in enumerate(np.linspace(2.6, 16.0, 180)):
            angle = np.log(radius / 2.6) / np.tan(pitch) + arm * np.pi / 2
            arms.append(
                {
                    "arm": arm,
                    "x": radius * np.cos(angle),
                    "y": radius * np.sin(angle),
                    "order": order,
                }
            )
    arm_chart = (
        alt.Chart(pd.DataFrame(arms))
        .mark_line(color="#bd93f9", opacity=0.16, strokeWidth=1.0)
        .encode(
            x=alt.X("x:Q", scale=x_scale, axis=x_axis),
            y=alt.Y("y:Q", scale=y_scale, axis=y_axis),
            detail="arm:N",
            order="order:Q",
        )
    )
    landmarks = pd.DataFrame(
        [
            {"x": 0.0, "y": 0.0, "label": "Galactic center", "kind": "center"},
            {"x": 8.122, "y": 0.0, "label": "Sun", "kind": "sun"},
        ]
    )
    landmark_chart = (
        alt.Chart(landmarks)
        .mark_point(size=105, filled=True)
        .encode(
            x=alt.X("x:Q", scale=x_scale, axis=x_axis),
            y=alt.Y("y:Q", scale=y_scale, axis=y_axis),
            shape=alt.Shape(
                "kind:N",
                scale=alt.Scale(domain=["center", "sun"], range=["circle", "diamond"]),
                legend=None,
            ),
            color=alt.Color(
                "kind:N",
                scale=alt.Scale(
                    domain=["center", "sun"],
                    range=["#f8f8f2", "#edc948"],
                ),
                legend=None,
            ),
            tooltip=alt.Tooltip("label:N"),
        )
    )
    return ring_chart, arm_chart, landmark_chart


def _scales(
    kind: str,
    frame: CasePlotFrame,
    *,
    color_color: bool = False,
) -> tuple[alt.Scale, alt.Scale]:
    if kind == "mollweide":
        return (
            alt.Scale(domain=[-2.9, 2.9], nice=False, zero=False),
            alt.Scale(domain=[-1.47, 1.47], nice=False, zero=False),
        )
    if kind == "galactic_plane":
        values = frame.full[[frame.x, frame.y]].to_numpy(dtype=float)
        extent = min(30.0, max(17.0, float(np.ceil(np.nanmax(np.abs(values)) + 1.0))))
        return (
            alt.Scale(domain=[-extent, extent], nice=False, zero=False),
            alt.Scale(domain=[-extent, extent], nice=False, zero=False),
        )
    x_domain = _numeric_domain(frame.full[frame.x])
    y_domain = _numeric_domain(frame.full[frame.y])
    return (
        alt.Scale(domain=x_domain, zero=False),
        alt.Scale(domain=y_domain, zero=False, reverse=not color_color),
    )


def _numeric_domain(values: pd.Series) -> list[float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return [0.0, 1.0]
    low, high = clean.quantile([0.001, 0.999]).tolist()
    if np.isclose(low, high):
        low -= 0.5
        high += 0.5
    padding = (high - low) * 0.03
    return [float(low - padding), float(high + padding)]


def _case_tooltip(
    data: pd.DataFrame,
    x: str,
    y: str,
) -> list[alt.Tooltip]:
    tooltip = [
        alt.Tooltip("object_name:N", title="object"),
        alt.Tooltip("diagnostic_state:N", title="state"),
        alt.Tooltip("rank:Q", title="rank"),
        alt.Tooltip("score:Q", title="score", format=".4f"),
        alt.Tooltip(f"{x}:Q", title=_axis_label(x), format=".3f"),
        alt.Tooltip(f"{y}:Q", title=_axis_label(y), format=".3f"),
    ]
    for column, title in [
        ("galactic_l", "Galactic l (deg)"),
        ("galactic_b", "Galactic b (deg)"),
        ("distance_kpc", "Distance (kpc)"),
        ("simbad_main_type", "SIMBAD type"),
        ("spectral_type", "spectral type"),
    ]:
        if column in data.columns:
            if column in {"galactic_l", "galactic_b", "distance_kpc"}:
                tooltip.append(
                    alt.Tooltip(
                        f"{column}:Q",
                        title=title,
                        format=".3f",
                    )
                )
            else:
                tooltip.append(alt.Tooltip(f"{column}:N", title=title))
    return tooltip


def _axis_label(column: str) -> str:
    labels = {
        "BP_RP": "BP - RP",
        "G_BP": "G - BP",
        "G_RP": "G - RP",
        "J_H": "J - H",
        "J_K": "J - Ks",
        "H_K": "H - Ks",
        "W1_W2": "W1 - W2",
        "mollweide_x": "Mollweide X",
        "mollweide_y": "Mollweide Y",
        "galactocentric_x_kpc": "Galactocentric X (kpc)",
        "galactocentric_y_kpc": "Galactocentric Y (kpc)",
    }
    if column in labels:
        return labels[column]
    if column in {"G", "BP", "RP", "J", "H", "Ks", "W1", "W2"}:
        return f"{column} (mag)"
    return column.replace("_", " ")


def _hidden_axis() -> alt.Axis:
    return alt.Axis(
        labels=False,
        ticks=False,
        domain=False,
        grid=False,
        title=None,
    )


def _style_matplotlib_axis(axis) -> None:
    axis.set_facecolor("#282a36")
    axis.tick_params(colors="#d8d8dc", labelsize=12)
    axis.xaxis.label.set_color("#f8f8f2")
    axis.yaxis.label.set_color("#f8f8f2")
    axis.grid(color="#555866", alpha=0.22, linewidth=0.6)
    for spine in axis.spines.values():
        spine.set_color("#777986")


def _export_background(axis, frame: CasePlotFrame) -> None:
    background = frame.full[frame.full["diagnostic_state"].isin(BACKGROUND_STATES)]
    if background.empty:
        return
    if frame.density.empty:
        axis.scatter(
            background[frame.x],
            background[frame.y],
            s=5,
            c=BACKGROUND_BLUE,
            alpha=0.15,
            linewidths=0,
        )
        return
    axis.hexbin(
        background[frame.x],
        background[frame.y],
        gridsize=70,
        mincnt=1,
        color=BACKGROUND_BLUE,
        alpha=0.28,
        linewidths=0,
    )


def _export_photometric(axis, frame: CasePlotFrame, *, color_color: bool) -> None:
    _export_background(axis, frame)
    axis.set_xlabel(_axis_label(frame.x), fontsize=12)
    axis.set_ylabel(_axis_label(frame.y), fontsize=12)
    if not color_color:
        axis.invert_yaxis()


def _export_mollweide(axis, frame: CasePlotFrame) -> None:
    boundary_angle = np.linspace(0, 2 * np.pi, 361)
    axis.plot(
        2 * np.sqrt(2) * np.cos(boundary_angle),
        np.sqrt(2) * np.sin(boundary_angle),
        color="#888b98",
        alpha=0.5,
        linewidth=0.8,
    )
    for longitude in [-135, -90, -45, 0, 45, 90, 135]:
        latitude = np.linspace(-89.5, 89.5, 181)
        x, y = mollweide_project(np.full_like(latitude, longitude), latitude)
        axis.plot(x, y, color="#888b98", alpha=0.2, linewidth=0.6)
    for latitude in [-60, -30, 0, 30, 60]:
        longitude = np.linspace(-179.5, 179.5, 361)
        x, y = mollweide_project(longitude, np.full_like(longitude, latitude))
        axis.plot(x, y, color="#888b98", alpha=0.2, linewidth=0.6)
    _export_background(axis, frame)
    axis.set_xlim(-2.9, 2.9)
    axis.set_ylim(-1.47, 1.47)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_xlabel("Galactic longitude (l = 0° centered; increases left)", fontsize=12)


def _export_galactic_plane(axis, frame: CasePlotFrame) -> None:
    for radius in [4.0, 8.0, 12.0, 16.0]:
        angle = np.linspace(0, 2 * np.pi, 181)
        axis.plot(
            radius * np.cos(angle),
            radius * np.sin(angle),
            color="#6272a4",
            alpha=0.15,
            linewidth=0.7,
        )
    pitch = np.deg2rad(18.5)
    for arm in range(4):
        radius = np.linspace(2.6, 16.0, 180)
        angle = np.log(radius / 2.6) / np.tan(pitch) + arm * np.pi / 2
        axis.plot(
            radius * np.cos(angle),
            radius * np.sin(angle),
            color="#bd93f9",
            alpha=0.16,
            linewidth=1.0,
        )
    _export_background(axis, frame)
    axis.scatter([0.0], [0.0], s=38, c="#f8f8f2", marker="o", zorder=4)
    axis.scatter([8.122], [0.0], s=42, c="#edc948", marker="D", zorder=4)
    extent = min(
        30.0,
        max(
            17.0,
            float(
                np.ceil(
                    np.nanmax(np.abs(frame.full[[frame.x, frame.y]].to_numpy())) + 1
                )
            ),
        ),
    )
    axis.set_xlim(-extent, extent)
    axis.set_ylim(-extent, extent)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Galactocentric X (kpc)", fontsize=12)
    axis.set_ylabel("Galactocentric Y (kpc)", fontsize=12)


def _export_relevant_points(axis, frame: CasePlotFrame) -> None:
    marker_map = {
        "circle": "o",
        "triangle-up": "^",
        "diamond": "D",
    }
    for state in DIAGNOSTIC_ORDER:
        subset = frame.relevant[frame.relevant["diagnostic_state"].eq(state)]
        if subset.empty:
            continue
        filled = DIAGNOSTIC_MARKS[state]["filled"]
        color = DIAGNOSTIC_COLORS[state]
        axis.scatter(
            subset[frame.x],
            subset[frame.y],
            s=28 if filled else 34,
            marker=marker_map[DIAGNOSTIC_SHAPES[state]],
            facecolors=color if filled else "none",
            edgecolors=color,
            linewidths=0.8 if filled else 1.2,
            alpha=DIAGNOSTIC_MARKS[state]["opacity"],
            label=state,
            zorder=5,
        )
    if not frame.relevant.empty:
        legend = axis.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.10),
            ncol=4,
            frameon=False,
            fontsize=12,
        )
        for text in legend.get_texts():
            text.set_color("#f8f8f2")


def _export_selected(axis, frame: CasePlotFrame, source_id: int) -> None:
    selected = frame.full[frame.full["source_id"].eq(source_id)]
    if selected.empty:
        return
    row = selected.iloc[0]
    axis.scatter(
        [row[frame.x]],
        [row[frame.y]],
        s=130,
        facecolors="none",
        edgecolors=SELECTION_OUTER,
        linewidths=2.2,
        zorder=8,
    )
    axis.scatter(
        [row[frame.x]],
        [row[frame.y]],
        s=90,
        facecolors="none",
        edgecolors=SELECTION_INNER,
        linewidths=1.2,
        zorder=9,
    )
