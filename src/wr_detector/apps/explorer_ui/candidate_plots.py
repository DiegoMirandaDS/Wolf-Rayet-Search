"""Altair charts for prediction-pool and candidate-review pages."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd


DISPOSITION_DOMAIN = [
    "known_wr",
    "catalogued_non_wr",
    "emission_or_ambiguous",
    "generic_or_uninformative",
    "no_exact_match",
    "ambiguous_positional_match",
]
DISPOSITION_RANGE = [
    "#f28e2b",
    "#4e79a7",
    "#edc948",
    "#59a14f",
    "#bab0ab",
    "#b07aa1",
]


def pool_tile_map(tiles: pd.DataFrame) -> alt.Chart | None:
    required = {
        "ra_min",
        "ra_max",
        "dec_min",
        "dec_max",
        "retention_fraction",
        "tile_id",
    }
    if tiles.empty or not required.issubset(tiles.columns):
        return None
    frame = tiles.dropna(
        subset=["ra_min", "ra_max", "dec_min", "dec_max"]
    ).copy()
    if frame.empty:
        return None
    return (
        alt.Chart(frame)
        .mark_rect(stroke="#282a36", strokeWidth=0.2)
        .encode(
            x=alt.X("ra_min:Q", title="RA [deg]", scale=alt.Scale(domain=[0, 360])),
            x2="ra_max:Q",
            y=alt.Y("dec_min:Q", title="Dec [deg]", scale=alt.Scale(domain=[-90, 90])),
            y2="dec_max:Q",
            color=alt.Color(
                "retention_fraction:Q",
                title="Operational / acquired",
                scale=alt.Scale(scheme="viridis"),
            ),
            tooltip=[
                alt.Tooltip("tile_id:N", title="Tile"),
                alt.Tooltip("status:N", title="Status"),
                alt.Tooltip("acquired_pre_locus:Q", title="Acquired", format=","),
                alt.Tooltip("written:Q", title="Operational", format=","),
                alt.Tooltip(
                    "retention_fraction:Q",
                    title="Retention",
                    format=".1%",
                ),
            ],
        )
        .properties(width="container", height=330)
    )


def scoring_coverage(models: pd.DataFrame) -> alt.Chart | None:
    required = {
        "model_label",
        "eligible_rows",
        "scored_rows",
        "missing_feature_rows",
    }
    if models.empty or not required.issubset(models.columns):
        return None
    frame = models[
        [
            "model_label",
            "eligible_rows",
            "scored_rows",
            "missing_feature_rows",
        ]
    ].melt(
        id_vars="model_label",
        var_name="measure",
        value_name="rows",
    )
    labels = {
        "eligible_rows": "Eligible",
        "scored_rows": "Scored",
        "missing_feature_rows": "Missing features",
    }
    frame["measure_label"] = frame["measure"].map(labels)
    return (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            x=alt.X("rows:Q", title="Sources", axis=alt.Axis(format="~s")),
            y=alt.Y("model_label:N", title=None, sort="-x"),
            color=alt.Color(
                "measure_label:N",
                title=None,
                scale=alt.Scale(
                    domain=["Eligible", "Scored", "Missing features"],
                    range=["#4e79a7", "#59a14f", "#e15759"],
                ),
                legend=alt.Legend(orient="bottom"),
            ),
            yOffset="measure_label:N",
            tooltip=[
                alt.Tooltip("model_label:N", title="Model"),
                alt.Tooltip("measure_label:N", title="Measure"),
                alt.Tooltip("rows:Q", title="Sources", format=","),
            ],
        )
        .properties(width="container", height=max(220, 52 * models.shape[0]))
    )


def disposition_bar(counts: pd.DataFrame) -> alt.Chart | None:
    if counts.empty or not {"review_disposition", "rows"}.issubset(counts.columns):
        return None
    return (
        alt.Chart(counts)
        .mark_bar()
        .encode(
            x=alt.X("rows:Q", title="Candidates"),
            y=alt.Y(
                "review_disposition:N",
                title=None,
                sort="-x",
            ),
            color=_disposition_color(),
            tooltip=[
                alt.Tooltip("review_disposition:N", title="Disposition"),
                alt.Tooltip("rows:Q", title="Candidates", format=","),
            ],
        )
        .properties(width="container", height=max(200, 35 * len(counts)))
    )


def jaccard_heatmap(frame: pd.DataFrame) -> alt.Chart | None:
    required = {"left_role", "right_role", "jaccard", "intersection", "top_k"}
    if frame.empty or not required.issubset(frame.columns):
        return None
    base = (
        alt.Chart(frame)
        .encode(
            x=alt.X("left_role:N", title=None, axis=alt.Axis(labelAngle=-35)),
            y=alt.Y("right_role:N", title=None),
            tooltip=[
                alt.Tooltip("left_role:N", title="Left"),
                alt.Tooltip("right_role:N", title="Right"),
                alt.Tooltip("jaccard:Q", title="Jaccard", format=".1%"),
                alt.Tooltip("intersection:Q", title="Intersection", format=","),
                alt.Tooltip("top_k:Q", title="Top K", format=","),
            ],
        )
    )
    heat = base.mark_rect().encode(
        color=alt.Color(
            "jaccard:Q",
            title="Jaccard",
            scale=alt.Scale(scheme="viridis", domain=[0, 1]),
        )
    )
    text = base.mark_text(fontSize=11).encode(
        text=alt.Text("jaccard:Q", format=".0%"),
        color=alt.condition(
            "datum.jaccard > 0.55",
            alt.value("#101010"),
            alt.value("#f8f8f2"),
        ),
    )
    return (heat + text).properties(width="container", height=340)


def support_distribution(frame: pd.DataFrame) -> alt.Chart | None:
    if frame.empty or "model_support" not in frame.columns:
        return None
    counts = (
        frame.groupby("model_support", as_index=False)
        .size()
        .rename(columns={"size": "candidates"})
    )
    return (
        alt.Chart(counts)
        .mark_bar(color="#6f42c1")
        .encode(
            x=alt.X(
                "model_support:O",
                title="Models supporting candidate",
            ),
            y=alt.Y("candidates:Q", title="Candidates"),
            tooltip=[
                alt.Tooltip("model_support:O", title="Models"),
                alt.Tooltip("candidates:Q", title="Candidates", format=","),
            ],
        )
        .properties(width="container", height=270)
    )


def rank_agreement(frame: pd.DataFrame) -> alt.Chart | None:
    required = {
        "consensus_rank",
        "best_model_rank",
        "model_support",
        "review_disposition",
        "source_id",
    }
    if frame.empty or not required.issubset(frame.columns):
        return None
    plot = frame.dropna(subset=["consensus_rank", "best_model_rank"]).copy()
    if plot.empty:
        return None
    return (
        alt.Chart(plot)
        .mark_circle(opacity=0.75)
        .encode(
            x=alt.X("consensus_rank:Q", title="Consensus rank"),
            y=alt.Y(
                "best_model_rank:Q",
                title="Best individual rank",
                scale=alt.Scale(type="log"),
            ),
            size=alt.Size(
                "model_support:Q",
                title="Model support",
                scale=alt.Scale(range=[30, 180]),
            ),
            color=_disposition_color(),
            tooltip=[
                alt.Tooltip("source_id:N", title="Gaia DR3"),
                alt.Tooltip("consensus_rank:Q", title="Consensus rank", format=","),
                alt.Tooltip("best_model_rank:Q", title="Best model rank", format=","),
                alt.Tooltip("model_support:Q", title="Model support"),
                alt.Tooltip("review_disposition:N", title="SIMBAD disposition"),
            ],
        )
        .properties(width="container", height=320)
        .interactive()
    )


def candidate_photometric(
    frame: pd.DataFrame,
    *,
    selected_source_id: int | None,
    x: str,
    y: str,
) -> alt.Chart | None:
    if frame.empty or not {x, y, "source_id"}.issubset(frame.columns):
        return None
    plot = frame.dropna(subset=[x, y]).copy()
    if plot.empty:
        return None
    base = (
        alt.Chart(plot)
        .mark_circle(size=45, opacity=0.6)
        .encode(
            x=alt.X(f"{x}:Q", title=_field_label(x)),
            y=alt.Y(
                f"{y}:Q",
                title=_field_label(y),
                scale=alt.Scale(reverse=y in {"G", "BP", "RP", "J", "H", "Ks", "W1", "W2"}),
            ),
            color=_disposition_color(),
            tooltip=_candidate_tooltip(x, y),
        )
    )
    selected = _selected_points(plot, selected_source_id).encode(
        x=alt.X(f"{x}:Q"),
        y=alt.Y(f"{y}:Q"),
    )
    return (base + selected).properties(width="container", height=390).interactive()


def candidate_mollweide(
    frame: pd.DataFrame,
    *,
    selected_source_id: int | None,
) -> alt.Chart | None:
    required = {"mollweide_x", "mollweide_y", "source_id"}
    if frame.empty or not required.issubset(frame.columns):
        return None
    plot = frame.dropna(subset=["mollweide_x", "mollweide_y"]).copy()
    if plot.empty:
        return None
    base = (
        alt.Chart(plot)
        .mark_circle(size=35, opacity=0.65)
        .encode(
            x=alt.X(
                "mollweide_x:Q",
                title="Galactic longitude (l=0 centered; increases left)",
                axis=alt.Axis(labels=False, ticks=False),
            ),
            y=alt.Y(
                "mollweide_y:Q",
                title="Galactic latitude",
                axis=alt.Axis(labels=False, ticks=False),
            ),
            color=_disposition_color(),
            tooltip=[
                alt.Tooltip("source_id:N", title="Gaia DR3"),
                alt.Tooltip("consensus_rank:Q", title="Rank", format=","),
                alt.Tooltip("galactic_l:Q", title="l", format=".2f"),
                alt.Tooltip("galactic_b:Q", title="b", format=".2f"),
                alt.Tooltip("review_disposition:N", title="Disposition"),
            ],
        )
    )
    selected = _selected_points(plot, selected_source_id).encode(
        x="mollweide_x:Q",
        y="mollweide_y:Q",
    )
    return (base + selected).properties(width="container", height=330)


def candidate_galactic_plane(
    frame: pd.DataFrame,
    *,
    selected_source_id: int | None,
) -> alt.Chart | None:
    required = {
        "galactocentric_x_kpc",
        "galactocentric_y_kpc",
        "distance_plotted",
        "source_id",
    }
    if frame.empty or not required.issubset(frame.columns):
        return None
    plot = frame[frame["distance_plotted"].fillna(False)].dropna(
        subset=["galactocentric_x_kpc", "galactocentric_y_kpc"]
    )
    if plot.empty:
        return None
    points = (
        alt.Chart(plot)
        .mark_circle(size=42, opacity=0.65)
        .encode(
            x=alt.X("galactocentric_x_kpc:Q", title="Galactocentric X [kpc]"),
            y=alt.Y("galactocentric_y_kpc:Q", title="Galactocentric Y [kpc]"),
            color=_disposition_color(),
            tooltip=[
                alt.Tooltip("source_id:N", title="Gaia DR3"),
                alt.Tooltip("consensus_rank:Q", title="Rank", format=","),
                alt.Tooltip("distance_kpc:Q", title="1/parallax [kpc]", format=".2f"),
                alt.Tooltip("parallax_over_error:Q", title="Parallax/error", format=".2f"),
                alt.Tooltip("review_disposition:N", title="Disposition"),
            ],
        )
    )
    anchors = pd.DataFrame(
        [
            {"x": 0.0, "y": 0.0, "label": "Galactic center"},
            {"x": 8.122, "y": 0.0, "label": "Sun"},
        ]
    )
    anchor_chart = (
        alt.Chart(anchors)
        .mark_point(shape="diamond", size=100, filled=True, color="#e15759")
        .encode(
            x="x:Q",
            y="y:Q",
            tooltip=alt.Tooltip("label:N", title=None),
        )
    )
    selected = _selected_points(plot, selected_source_id).encode(
        x="galactocentric_x_kpc:Q",
        y="galactocentric_y_kpc:Q",
    )
    return (points + anchor_chart + selected).properties(
        width="container",
        height=330,
    )


def model_rank_ladder(evidence: pd.DataFrame) -> alt.Chart | None:
    required = {"role", "model_rank", "score", "rrf_contribution"}
    if evidence.empty or not required.issubset(evidence.columns):
        return None
    return (
        alt.Chart(evidence)
        .mark_circle(size=130, color="#bd93f9")
        .encode(
            x=alt.X(
                "model_rank:Q",
                title="Original model rank",
                scale=alt.Scale(type="log"),
            ),
            y=alt.Y("role:N", title=None, sort="x"),
            tooltip=[
                alt.Tooltip("role:N", title="Role"),
                alt.Tooltip("model_rank:Q", title="Rank", format=","),
                alt.Tooltip("score:Q", title="Ranking score", format=".5f"),
                alt.Tooltip(
                    "rrf_contribution:Q",
                    title="RRF contribution",
                    format=".6f",
                ),
            ],
        )
        .properties(width="container", height=max(210, 48 * len(evidence)))
    )


def _selected_points(
    frame: pd.DataFrame,
    selected_source_id: int | None,
) -> alt.Chart:
    if selected_source_id is None:
        selected = frame.iloc[0:0]
    else:
        selected = frame[frame["source_id"].eq(int(selected_source_id))]
    return (
        alt.Chart(selected)
        .mark_point(
            size=230,
            filled=False,
            stroke="#e15759",
            strokeWidth=3,
        )
    )


def _disposition_color() -> alt.Color:
    return alt.Color(
        "review_disposition:N",
        title="SIMBAD disposition",
        scale=alt.Scale(
            domain=DISPOSITION_DOMAIN,
            range=DISPOSITION_RANGE,
        ),
        legend=alt.Legend(orient="bottom"),
    )


def _candidate_tooltip(x: str, y: str) -> list[alt.Tooltip]:
    return [
        alt.Tooltip("source_id:N", title="Gaia DR3"),
        alt.Tooltip("consensus_rank:Q", title="Rank", format=","),
        alt.Tooltip(f"{x}:Q", title=_field_label(x), format=".3f"),
        alt.Tooltip(f"{y}:Q", title=_field_label(y), format=".3f"),
        alt.Tooltip("model_support:Q", title="Model support"),
        alt.Tooltip("review_disposition:N", title="Disposition"),
        alt.Tooltip("simbad_main_id:N", title="SIMBAD ID"),
    ]


def _field_label(field: str) -> str:
    return {
        "BP_RP": "Gaia BP - RP",
        "G_BP": "Gaia G - BP",
        "G_RP": "Gaia G - RP",
        "J_H": "2MASS J - H",
        "J_K": "2MASS J - Ks",
        "H_K": "2MASS H - Ks",
        "W1_W2": "WISE W1 - W2",
        "G": "Gaia G",
        "BP": "Gaia BP",
        "RP": "Gaia RP",
        "J": "2MASS J",
        "H": "2MASS H",
        "Ks": "2MASS Ks",
        "W1": "WISE W1",
        "W2": "WISE W2",
    }.get(field, field)
