"""Aggregate statistics: subtype recovery, contamination and run-wide patterns."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from wr_detector.apps.explorer_ui import charts, data, ui
from wr_detector.modeling.cases import false_positive_composition, subtype_recovery


SPLIT_LABELS = {"holdout": "Holdout", "train_oof": "Train (out-of-fold)"}
RECOVERY_KS = [10, 50, 100, 500, 1000]


def render() -> None:
    st.title("Statistics")
    results = data.results()
    if results.empty:
        st.warning("The selected run has no model results.")
        return

    top = st.columns([2.4, 1.2, 1])
    with top[0]:
        selected_model = ui.select_model(results)
    with top[1]:
        split = st.selectbox("Split", options=list(SPLIT_LABELS), format_func=SPLIT_LABELS.get)
    with top[2]:
        top_k = int(st.number_input("Budget K", min_value=10, max_value=5000, value=100, step=10))

    cases = data.cases(str(selected_model["result_id"]), split)
    if cases.empty:
        st.info("No synchronized predictions for this model and split.")
        return

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("WR recovery by subtype")
        recovery = subtype_recovery(cases, ks=RECOVERY_KS)
        chart = charts.subtype_recovery_lines(recovery, ks=RECOVERY_KS)
        if chart is not None:
            st.altair_chart(chart)
            st.dataframe(
                recovery[["wr_subtype", "total", *[f"recovered_at_{k}" for k in [50, 100, 500] if f"recovered_at_{k}" in recovery.columns]]],
                hide_index=True,
                width="stretch",
            )
        else:
            st.info("No WR subtype information available (WR reference DB not found).")
    with right:
        st.subheader(f"False-positive composition @{top_k}")
        composition = false_positive_composition(cases, top_k=top_k)
        chart = charts.composition_bars(composition)
        if chart is not None:
            st.altair_chart(chart)
            st.caption("SIMBAD main object types of negatives ranked inside the budget.")
        else:
            st.info("No false positives inside the budget, or SIMBAD types unavailable.")

    st.subheader("Score distribution")
    histogram = charts.score_histogram(cases)
    if histogram is not None:
        st.altair_chart(histogram)
        st.caption("Counts use a symlog scale; negatives vastly outnumber WR.")

    st.divider()
    st.subheader("Run-wide patterns across all models")
    st.caption(
        f"Sources flagged consistently by the {len(results)} models of this run, using budget K={top_k} on the {SPLIT_LABELS[split]} split."
    )
    tab_fp, tab_missed = st.tabs(["Recurrent contaminants", "Persistently missed WR"])
    with tab_fp:
        _overlap_table(data.case_overlap(split, "false_positive", top_k), kind="false_positive")
    with tab_missed:
        _overlap_table(data.case_overlap(split, "missed_wr", top_k), kind="missed_wr")


def _overlap_table(overlap: pd.DataFrame, *, kind: str) -> None:
    if overlap.empty:
        st.info("No recurrent cases found.")
        return
    columns = [
        column
        for column in [
            "object_name",
            "model_share",
            "n_models",
            "score_mean",
            "best_rank",
            "worst_rank",
            "spectral_type" if kind == "missed_wr" else "simbad_main_type",
            "simbad_sp_type" if kind == "false_positive" else None,
        ]
        if column is not None and column in overlap.columns
    ]
    st.dataframe(
        overlap[columns].head(50),
        hide_index=True,
        width="stretch",
        column_config={
            "object_name": st.column_config.TextColumn("object"),
            "model_share": st.column_config.ProgressColumn("models flagging", min_value=0.0, max_value=1.0, format="percent"),
            "n_models": st.column_config.NumberColumn("n models"),
            "score_mean": st.column_config.NumberColumn("mean score", format="%.3f"),
            "best_rank": st.column_config.NumberColumn("best rank"),
            "worst_rank": st.column_config.NumberColumn("worst rank"),
            "simbad_main_type": st.column_config.TextColumn("SIMBAD type"),
            "simbad_sp_type": st.column_config.TextColumn("SIMBAD sp. type"),
            "spectral_type": st.column_config.TextColumn("WR spectral type"),
        },
    )
