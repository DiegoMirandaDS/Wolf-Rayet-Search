"""Case-by-case review of per-source predictions for one model."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from wr_detector.apps.explorer_ui import charts, data, ui


SPLIT_LABELS = {"holdout": "Holdout", "train_oof": "Train (out-of-fold)"}
VIEW_MODES = ["Top candidates", "False positives", "Missed WR", "All WR", "All sources"]
CASE_SOURCE_KEY = "case_source_id"

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
        selected_model = ui.select_model(results)
    with top[1]:
        split = st.selectbox("Split", options=list(SPLIT_LABELS), format_func=SPLIT_LABELS.get)
    with top[2]:
        mode = st.selectbox("View", options=VIEW_MODES)
    with top[3]:
        top_k = int(st.number_input("Budget K", min_value=10, max_value=5000, value=100, step=10))

    cases = data.cases(str(selected_model["result_id"]), split)
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

    subset = _subset(cases, mode, top_k)
    if subset.empty:
        st.info("No cases in this view.")
        return

    table_col, detail_col = st.columns([1.45, 1], gap="large")
    with table_col:
        _case_table(subset)
        chart = charts.color_magnitude(cases, selected_source_id=st.session_state.get(CASE_SOURCE_KEY))
        if chart is not None:
            st.altair_chart(chart)
    with detail_col:
        _case_detail(subset, split, top_k)


def _subset(cases: pd.DataFrame, mode: str, top_k: int) -> pd.DataFrame:
    if mode == "Top candidates":
        return cases.head(top_k)
    if mode == "False positives":
        return cases[cases["target"].eq(0) & cases["rank"].le(top_k)]
    if mode == "Missed WR":
        return cases[cases["target"].eq(1) & cases["rank"].gt(top_k)]
    if mode == "All WR":
        return cases[cases["target"].eq(1)]
    return cases


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
        st.session_state[CASE_SOURCE_KEY] = int(view.iloc[selected_rows[0]]["source_id"])


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
        st.rerun()
    nav_pos.markdown(
        f"<div style='text-align:center;padding-top:6px'>case {idx + 1} of {len(subset)}</div>",
        unsafe_allow_html=True,
    )
    if nav_next.button("Next >", disabled=idx >= len(subset) - 1, width="stretch"):
        st.session_state[CASE_SOURCE_KEY] = int(subset.iloc[idx + 1]["source_id"])
        st.rerun()

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
