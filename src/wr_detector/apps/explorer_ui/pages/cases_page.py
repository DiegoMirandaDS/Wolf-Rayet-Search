"""Case-by-case review of per-source predictions for one model."""

from __future__ import annotations

from time import perf_counter

import pandas as pd
import streamlit as st

from wr_detector.apps.explorer_ui import case_plots, charts, data, ui
from wr_detector.modeling.case_visualization import CaseDataFingerprint, CasePlotSettings
from wr_detector.modeling.cases import (
    DIAGNOSTIC_BASES,
    DIAGNOSTIC_STATES,
    add_case_spatial_coordinates,
    classify_cases,
)


SPLIT_LABELS = {"holdout": "Holdout", "train_oof": "Train (out-of-fold)"}
VIEW_MODES = [
    "Top candidates",
    "Budget contaminants",
    "WR outside budget",
    "Threshold false positives",
    "Threshold false negatives",
    "All WR",
    "All sources",
]
CASE_SOURCE_KEY = "case_source_id"
CASE_INTERACTION_KEY = "case_review_interaction"
CASE_TELEMETRY_KEY = "case_review_telemetry"

COLOR_COLUMNS = ["BP_RP", "G_BP", "G_RP", "J_H", "J_K", "H_K", "W1_W2"]
MAGNITUDE_COLUMNS = ["G", "BP", "RP", "J", "H", "Ks", "W1", "W2"]

TABLE_COLUMNS = [
    ("rank", "#"),
    ("object_name", "object"),
    ("score", "score"),
    ("spectral_type", "WR type"),
    ("simbad_main_type", "SIMBAD type"),
    ("G", "G"),
    ("BP_RP", "BP-RP"),
    ("J_K", "J-Ks"),
    ("W1_W2", "W1-W2"),
    ("parallax", "parallax"),
    ("ruwe", "RUWE"),
]


def render() -> None:
    st.title("Case review")
    results = data.results()
    if results.empty:
        st.warning("The selected run has no model results.")
        return

    top = st.columns([2.4, 1.2, 1.6, 1])
    with top[0]:
        selected_model = ui.active_model_control(results, widget_key="cases_active_model")
    with top[1]:
        split = st.selectbox("Split", options=list(SPLIT_LABELS), format_func=SPLIT_LABELS.get)
    with top[2]:
        mode = st.selectbox("View", options=VIEW_MODES)
    with top[3]:
        top_k = int(st.number_input("Budget K", min_value=10, max_value=5000, value=100, step=10))

    result_id = str(selected_model["result_id"])
    fingerprint = data.case_fingerprint(result_id, split)
    cases = data.cases(result_id, split, fingerprint)
    if cases.empty:
        st.info("No synchronized predictions for this model and split.")
        return

    wr_total = int(cases["target"].eq(1).sum())
    recovered = int((cases["target"].eq(1) & cases["rank"].le(top_k)).sum())
    contaminants = int((cases["target"].eq(0) & cases["rank"].le(top_k)).sum())
    ui.kpi_row(
        [
            ("Sources in split", f"{len(cases):,}", None),
            ("WR in split", f"{wr_total:,}", None),
            (f"WR recovered @{top_k}", ui.fmt_count_pct(recovered, recovered / wr_total if wr_total else None), None),
            (f"Contaminants @{top_k}", f"{contaminants:,}", "Negatives inside the budget"),
        ]
    )

    _case_review_fragment(
        cases,
        result_id=result_id,
        split=split,
        mode=mode,
        top_k=top_k,
        fingerprint=fingerprint,
    )
    return

    subset = _subset(cases, mode, top_k)
    if subset.empty:
        st.info("No cases in this view.")
        return

    table_col, detail_col = st.columns([1.45, 1], gap="large")
    with table_col:
        _case_table(subset)
    with detail_col:
        _case_detail(subset, split, top_k)

    (
        plotted_cases,
        x_axis,
        y_axis,
        color_color,
        min_parallax_over_error,
        max_distance_kpc,
    ) = _visualization_controls(cases, top_k)
    spatial_cases = add_case_spatial_coordinates(
        plotted_cases,
        min_parallax_over_error=min_parallax_over_error,
        max_distance_kpc=max_distance_kpc,
    )
    selected_source_id = st.session_state.get(CASE_SOURCE_KEY)

    plot_col, map_col = st.columns([1.45, 1], gap="large")
    with plot_col:
        st.subheader("Photometric space")
        chart = charts.color_magnitude(
            plotted_cases,
            selected_source_id=selected_source_id,
            x=x_axis,
            y=y_axis,
            color_color=color_color,
        )
        if chart is not None:
            st.altair_chart(chart)
        else:
            st.info("No sources have both selected photometric fields.")
        if color_color:
            st.caption(
                "Both axes use the project's intra-mission colors. This "
                "diagnostic view does not create or add model features."
            )
        else:
            st.caption(
                "The default BP - RP versus G view is a Gaia color-magnitude "
                "diagram. Its color is intra-mission; the view is diagnostic "
                "and does not add model features."
            )

        st.subheader("Galactic sky (polar projection)")
        polar = charts.galactic_polar_chart(
            spatial_cases,
            selected_source_id=selected_source_id,
        )
        if polar is not None:
            st.altair_chart(polar)
            st.caption(
                "Longitude runs around the plot; Galactic north is at the "
                "center, the Galactic plane is the middle circle (b = 0°), "
                "and Galactic south lies outside it."
            )
        else:
            st.info("RA/Dec are unavailable for the selected layers.")

    with map_col:
        st.subheader("Galactic plane projection")
        plane = charts.galactic_plane_map(
            spatial_cases,
            selected_source_id=selected_source_id,
        )
        plotted_count = int(spatial_cases["distance_plotted"].sum())
        if plane is not None:
            st.altair_chart(plane)
            st.caption(
                f"{plotted_count:,} of {len(spatial_cases):,} visible sources "
                "have positive parallax, meet the selected parallax/error "
                f"floor and lie within {max_distance_kpc:.0f} kpc. Distances "
                "use 1/parallax and the spiral arms are schematic."
            )
        else:
            st.info(
                "No visible source meets the selected distance-quality controls."
            )


@st.fragment
def _case_review_fragment(
    cases: pd.DataFrame,
    *,
    result_id: str,
    split: str,
    mode: str,
    top_k: int,
    fingerprint: CaseDataFingerprint,
) -> None:
    fragment_started = perf_counter()
    subset = _subset(cases, mode, top_k)
    if subset.empty:
        st.info("No cases in this view.")
        return

    table_col, detail_col = st.columns([1.45, 1], gap="large")
    with table_col:
        _case_table(subset)
    with detail_col:
        _case_detail(subset, split, top_k)

    settings = _optimized_visualization_controls(cases, top_k)
    preparation_started = perf_counter()
    prepared = data.case_plot_data(result_id, split, settings, fingerprint)
    preparation_elapsed = perf_counter() - preparation_started
    bases = data.case_plot_bases(result_id, split, settings, fingerprint)
    selected_source_id = st.session_state.get(CASE_SOURCE_KEY)

    payloads: dict[str, int] = {}
    build_times: dict[str, float] = {}
    st.subheader("Photometric space")
    photometric = _render_case_chart(
        "photometric",
        prepared,
        bases,
        selected_source_id,
        payloads,
        build_times,
    )
    if not photometric:
        st.info("No sources have both selected photometric fields.")
    st.caption(
        case_plots.plot_count_summary(prepared.frame("photometric"))
        + " Intra-mission diagnostic; no model feature is added."
    )

    with st.container(horizontal=True, gap="large"):
        with st.container(width=680):
            st.subheader("Galactic sky (Mollweide)")
            mollweide = _render_case_chart(
                "mollweide",
                prepared,
                bases,
                selected_source_id,
                payloads,
                build_times,
            )
            if not mollweide:
                st.info("RA/Dec are unavailable for the selected layers.")
            st.caption(case_plots.plot_count_summary(prepared.frame("mollweide")))

        with st.container(width=680):
            st.subheader("Galactic plane projection")
            plane = _render_case_chart(
                "galactic_plane",
                prepared,
                bases,
                selected_source_id,
                payloads,
                build_times,
            )
            if not plane:
                st.info("No visible source meets the distance-quality controls.")
            st.caption(case_plots.plot_count_summary(prepared.frame("galactic_plane")))

    st.caption(
        "Mollweide: l=0° centered and longitude increases left. "
        "Galactocentric distances use d=1/parallax; arms are schematic."
    )
    interaction = st.session_state.pop(
        CASE_INTERACTION_KEY,
        {"kind": "initial_or_passive"},
    )
    st.session_state[CASE_TELEMETRY_KEY] = {
        "interaction": interaction,
        "result_id": result_id,
        "split": split,
        "fingerprint": prepared.fingerprint,
        "preparation_call_seconds": preparation_elapsed,
        "cached_preparation_seconds": prepared.preparation_seconds,
        "chart_build_seconds": build_times,
        "payload_bytes": payloads,
        "payload_combined_bytes": sum(payloads.values()),
        "fragment_server_seconds": perf_counter() - fragment_started,
    }
    if st.query_params.get("profile") == "1":
        with st.expander("Case Review performance", expanded=False):
            st.json(st.session_state[CASE_TELEMETRY_KEY])


def _render_case_chart(
    kind: str,
    prepared,
    bases: dict[str, dict[str, object]],
    selected_source_id: int | None,
    payloads: dict[str, int],
    build_times: dict[str, float],
) -> bool:
    started = perf_counter()
    base_entry = bases.get(kind, {})
    chart = case_plots.build_interactive_case_plot(
        kind,
        prepared,
        selected_source_id=selected_source_id,
        base_chart=base_entry.get("chart"),
    )
    build_times[kind] = perf_counter() - started
    payloads[kind] = int(base_entry.get("payload_bytes", 0)) + (
        case_plots.selected_layer_payload_bytes(
            kind,
            prepared,
            selected_source_id,
        )
    )
    if chart is None:
        return False
    st.altair_chart(chart, width="stretch", key=f"case_review_{kind}")
    return True


def _subset(cases: pd.DataFrame, mode: str, top_k: int) -> pd.DataFrame:
    if mode == "Top candidates":
        return cases.head(top_k)
    if mode == "Budget contaminants":
        return cases[cases["target"].eq(0) & cases["rank"].le(top_k)]
    if mode == "WR outside budget":
        return cases[cases["target"].eq(1) & cases["rank"].gt(top_k)]
    if mode == "Threshold false positives":
        return cases[cases["target"].eq(0) & cases["predicted"].eq(1)]
    if mode == "Threshold false negatives":
        return cases[cases["target"].eq(1) & cases["predicted"].eq(0)]
    if mode == "All WR":
        return cases[cases["target"].eq(1)]
    return cases


def _optimized_visualization_controls(
    cases: pd.DataFrame,
    top_k: int,
) -> CasePlotSettings:
    with st.expander("Visualization controls", expanded=False):
        with st.form("case_visualization_filters", border=False):
            available_colors = [
                column
                for column in COLOR_COLUMNS
                if column in cases.columns and cases[column].notna().any()
            ]
            available_y = [
                column
                for column in [*MAGNITUDE_COLUMNS, *COLOR_COLUMNS]
                if column in cases.columns and cases[column].notna().any()
            ]
            row1 = st.columns([1.35, 1, 1, 1])
            basis = row1[0].selectbox(
                "Diagnostic basis",
                options=list(DIAGNOSTIC_BASES),
                format_func=DIAGNOSTIC_BASES.get,
            )
            x_axis = row1[1].selectbox(
                "Color axis",
                options=available_colors or ["BP_RP"],
                index=(
                    available_colors.index("BP_RP")
                    if "BP_RP" in available_colors
                    else 0
                ),
                format_func=_photometry_label,
            )
            y_axis = row1[2].selectbox(
                "Y axis",
                options=available_y or ["G"],
                index=available_y.index("G") if "G" in available_y else 0,
                format_func=_photometry_label,
            )
            background_mode = row1[3].selectbox(
                "Background",
                options=["density", "sample"],
                format_func={
                    "density": "Density",
                    "sample": "Sampled points",
                }.get,
            )

            layer_roles = [
                "Context",
                "False positives / contaminants",
                "False negatives / WR outside K",
                "True positives / WR recovered",
            ]
            row2 = st.columns([2.2, 1, 1, 1])
            visible_roles = row2[0].multiselect(
                "Visible layers",
                options=layer_roles,
                default=layer_roles,
            )
            min_parallax_over_error = float(
                row2[1].select_slider(
                    "Minimum parallax/error",
                    options=[0.0, 1.0, 2.0, 3.0, 5.0],
                    value=2.0,
                )
            )
            max_distance_kpc = float(
                row2[2].slider(
                    "Maximum distance (kpc)",
                    min_value=5,
                    max_value=25,
                    value=15,
                    step=1,
                )
            )
            background_limit = int(
                row2[3].number_input(
                    "Background point limit",
                    min_value=500,
                    max_value=20_000,
                    value=5_000,
                    step=500,
                )
            )
            applied = st.form_submit_button(
                "Apply visualization filters",
                type="primary",
            )
            if applied:
                st.session_state[CASE_INTERACTION_KEY] = {
                    "kind": "apply_filters",
                    "started_server": perf_counter(),
                }
        if basis == "review_budget":
            st.caption(
                f"Top-K describes a review budget. A WR outside {top_k:,} "
                "is not automatically a threshold false negative."
            )
        else:
            st.caption(
                "Threshold states use the synchronized operating decision."
            )
    role_states = {
        "review_budget": {
            "Context": "Background",
            "False positives / contaminants": "Contaminant @K",
            "False negatives / WR outside K": "WR outside @K",
            "True positives / WR recovered": "WR recovered @K",
        },
        "operating_threshold": {
            "Context": "True negative",
            "False positives / contaminants": "False positive",
            "False negatives / WR outside K": "False negative",
            "True positives / WR recovered": "True positive",
        },
    }
    threshold_values = pd.to_numeric(
        cases.get("threshold", pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()
    threshold = (
        float(threshold_values.iloc[0])
        if not threshold_values.empty
        else None
    )
    return CasePlotSettings(
        diagnostic_basis=basis,
        top_k=top_k,
        threshold=threshold,
        visible_states=tuple(
            role_states[basis][role]
            for role in visible_roles
            if role in role_states[basis]
        ),
        x_axis=x_axis,
        y_axis=y_axis,
        color_color=y_axis in COLOR_COLUMNS,
        min_parallax_over_error=min_parallax_over_error,
        max_distance_kpc=max_distance_kpc,
        background_mode=background_mode,
        background_limit=background_limit,
    )


def _visualization_controls(
    cases: pd.DataFrame,
    top_k: int,
) -> tuple[pd.DataFrame, str, str, bool, float, float]:
    with st.expander("Visualization controls", expanded=False):
        row1 = st.columns([1.35, 1, 1, 1])
        basis = row1[0].selectbox(
            "Diagnostic basis",
            options=list(DIAGNOSTIC_BASES),
            format_func=DIAGNOSTIC_BASES.get,
        )
        plot_type = row1[1].selectbox(
            "Photometric view",
            options=["Color-magnitude", "Color-color"],
        )
        available_colors = [
            column
            for column in COLOR_COLUMNS
            if column in cases.columns and cases[column].notna().any()
        ]
        available_magnitudes = [
            column
            for column in MAGNITUDE_COLUMNS
            if column in cases.columns and cases[column].notna().any()
        ]
        if available_colors:
            x_axis = row1[2].selectbox(
                "X axis",
                options=available_colors,
                index=available_colors.index("BP_RP") if "BP_RP" in available_colors else 0,
                format_func=_photometry_label,
            )
        else:
            row1[2].caption("No color fields available")
            x_axis = "BP_RP"
        y_options = (
            [column for column in available_colors if column != x_axis]
            if plot_type == "Color-color"
            else available_magnitudes
        )
        default_y = "J_H" if plot_type == "Color-color" else "G"
        if y_options:
            y_axis = row1[3].selectbox(
                "Y axis",
                options=y_options,
                index=y_options.index(default_y) if default_y in y_options else 0,
                format_func=_photometry_label,
                key=f"case_y_axis_{plot_type}",
            )
        else:
            row1[3].caption("No compatible Y fields available")
            y_axis = "J_H" if plot_type == "Color-color" else "G"

        classified = classify_cases(cases, basis=basis, top_k=top_k)
        states = DIAGNOSTIC_STATES[basis]
        row2 = st.columns([2.4, 1, 1])
        visible_states = row2[0].multiselect(
            "Visible layers",
            options=states,
            default=states,
            key=f"case_visible_layers_{basis}",
        )
        min_parallax_over_error = float(
            row2[1].select_slider(
                "Minimum parallax/error",
                options=[0.0, 1.0, 2.0, 3.0, 5.0],
                value=2.0,
            )
        )
        max_distance_kpc = float(
            row2[2].slider(
                "Maximum distance (kpc)",
                min_value=5,
                max_value=25,
                value=15,
                step=1,
            )
        )
        if basis == "review_budget":
            st.caption(
                "Top-K states describe a manual-review budget. A WR outside "
                f"the first {top_k:,} is not automatically a threshold false negative."
            )
        else:
            st.caption(
                "Threshold states use the synchronized predicted label: "
                "score at or above the selected operating threshold is positive."
            )
    return (
        classified[classified["diagnostic_state"].isin(visible_states)].copy(),
        x_axis,
        y_axis,
        plot_type == "Color-color",
        min_parallax_over_error,
        max_distance_kpc,
    )


def _photometry_label(column: str) -> str:
    return {
        "BP_RP": "Gaia: BP - RP",
        "G_BP": "Gaia: G - BP",
        "G_RP": "Gaia: G - RP",
        "J_H": "2MASS: J - H",
        "J_K": "2MASS: J - Ks",
        "H_K": "2MASS: H - Ks",
        "W1_W2": "WISE: W1 - W2",
    }.get(column, column)


def _case_table(subset: pd.DataFrame) -> None:
    columns = [column for column, _ in TABLE_COLUMNS if column in subset.columns]
    view = subset[["source_id", *columns]].reset_index(drop=True)
    labels = dict(TABLE_COLUMNS)
    column_config: dict[str, object] = {"source_id": None}
    for column in columns:
        label = labels.get(column, column)
        if column == "score":
            column_config[column] = st.column_config.NumberColumn(label, format="%.4f")
        elif pd.api.types.is_float_dtype(view[column]):
            column_config[column] = st.column_config.NumberColumn(label, format="%.3f")
        else:
            view[column] = view[column].astype("string")
            column_config[column] = st.column_config.TextColumn(label)
    event = st.dataframe(
        view,
        hide_index=True,
        width="stretch",
        height=420,
        column_config=column_config,
        on_select="rerun",
        selection_mode="single-row",
        key="case_table",
    )
    selected_rows = event.selection.rows if event.selection else []
    if selected_rows:
        selected_source_id = int(view.iloc[selected_rows[0]]["source_id"])
        if st.session_state.get(CASE_SOURCE_KEY) != selected_source_id:
            st.session_state[CASE_INTERACTION_KEY] = {
                "kind": "table_selection",
                "started_server": perf_counter(),
            }
        st.session_state[CASE_SOURCE_KEY] = selected_source_id


def _case_detail(subset: pd.DataFrame, split: str, top_k: int) -> None:
    subset = subset.reset_index(drop=True)
    source_id = st.session_state.get(CASE_SOURCE_KEY)
    matches = subset.index[subset["source_id"].eq(source_id)].tolist() if source_id is not None else []
    idx = matches[0] if matches else 0
    case = subset.iloc[idx]
    st.session_state[CASE_SOURCE_KEY] = int(case["source_id"])

    nav_prev, nav_pos, nav_next = st.columns([1, 2, 1])
    if nav_prev.button("< Previous", disabled=idx <= 0, width="stretch"):
        st.session_state[CASE_SOURCE_KEY] = int(subset.iloc[idx - 1]["source_id"])
        st.session_state[CASE_INTERACTION_KEY] = {
            "kind": "previous",
            "started_server": perf_counter(),
        }
        st.rerun(scope="fragment")
    nav_pos.markdown(
        f"<div style='text-align:center;padding-top:6px'>case {idx + 1} of {len(subset)}</div>",
        unsafe_allow_html=True,
    )
    if nav_next.button("Next >", disabled=idx >= len(subset) - 1, width="stretch"):
        st.session_state[CASE_SOURCE_KEY] = int(subset.iloc[idx + 1]["source_id"])
        st.session_state[CASE_INTERACTION_KEY] = {
            "kind": "next",
            "started_server": perf_counter(),
        }
        st.rerun(scope="fragment")

    is_wr = int(case["target"]) == 1
    badge = ":orange-badge[WR]" if is_wr else ":blue-badge[negative]"
    st.markdown(f"### {case.get('object_name', case['source_id'])} {badge}")

    identity_bits = []
    if is_wr and pd.notna(case.get("spectral_type")):
        identity_bits.append(f"spectral type **{case['spectral_type']}**")
    if not is_wr and pd.notna(case.get("simbad_main_type")):
        identity_bits.append(f"SIMBAD type **{case['simbad_main_type']}**")
    if not is_wr and pd.notna(case.get("simbad_sp_type")) and str(case.get("simbad_sp_type")).strip():
        identity_bits.append(f"SIMBAD sp. type **{case['simbad_sp_type']}**")
    if identity_bits:
        st.markdown(" | ".join(identity_bits))

    ui.kpi_row(
        [
            ("Rank", f"{int(case['rank']):,}", f"of {len(subset):,} in this view" if len(subset) else None),
            ("Score", ui.fmt(case.get("score"), digits=4), None),
            ("Threshold", ui.fmt(case.get("threshold"), digits=4), "Selected operating threshold"),
        ]
    )

    overlap = _recurrence(case, split, top_k)
    if overlap is not None:
        kind_text = "inside the top" if int(case["target"]) == 0 else "outside the top"
        st.markdown(
            f"Flagged {kind_text} {top_k} by **{int(overlap['n_models'])} of {int(overlap['n_total_models'])} models** "
            f"in this run (mean score {overlap['score_mean']:.3f}, best rank {int(overlap['best_rank'])})."
        )

    _photometry_tables(case)

    gaia_id = int(case["source_id"])
    links = [f"[SIMBAD](https://simbad.cds.unistra.fr/simbad/sim-id?Ident=Gaia+DR3+{gaia_id})"]
    if pd.notna(case.get("ra")) and pd.notna(case.get("dec")):
        links.append(
            f"[Aladin Lite](https://aladin.cds.unistra.fr/AladinLite/?target={float(case['ra'])}%20{float(case['dec'])}&fov=0.05&survey=P%2FDSS2%2Fcolor)"
        )
        links.append(f"[ESASky](https://sky.esa.int/esasky/?target={float(case['ra'])}%20{float(case['dec'])}&fov=0.1)")
    st.markdown("Gaia DR3 `" + str(gaia_id) + "` | " + " | ".join(links))


def _recurrence(case: pd.Series, split: str, top_k: int) -> pd.Series | None:
    kind = "false_positive" if int(case["target"]) == 0 else "missed_wr"
    overlap = data.case_overlap(split, kind, top_k)
    if overlap.empty:
        return None
    match = overlap[overlap["source_id"].eq(case["source_id"])]
    return match.iloc[0] if not match.empty else None


def _photometry_tables(case: pd.Series) -> None:
    magnitudes = {
        band: case.get(band)
        for band in ["G", "BP", "RP", "J", "H", "Ks", "W1", "W2"]
        if band in case.index and pd.notna(case.get(band))
    }
    colors = {
        color: case.get(color)
        for color in ["BP_RP", "G_BP", "G_RP", "J_H", "J_K", "H_K", "W1_W2"]
        if color in case.index and pd.notna(case.get(color))
    }
    astrometry = {
        label: case.get(column)
        for label, column in [
            ("parallax", "parallax"),
            ("parallax / error", "parallax_over_error"),
            ("RUWE", "ruwe"),
            ("RA", "ra"),
            ("Dec", "dec"),
        ]
        if column in case.index and pd.notna(case.get(column))
    }
    if magnitudes:
        st.dataframe(
            pd.DataFrame([magnitudes]).round(3),
            hide_index=True,
            width="stretch",
        )
    if colors:
        st.dataframe(
            pd.DataFrame([colors]).round(3),
            hide_index=True,
            width="stretch",
        )
    if astrometry:
        st.dataframe(
            pd.DataFrame([astrometry]).round(4),
            hide_index=True,
            width="stretch",
        )
