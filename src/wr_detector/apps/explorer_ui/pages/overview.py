"""Run overview: headline numbers and diverse operational candidates."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from wr_detector.apps.explorer_ui import charts, data, ui
from wr_detector.modeling.explorer import rank_models, select_diverse_top_models


def render() -> None:
    st.title("Run overview")
    results = data.results()
    if results.empty:
        st.warning("The selected run has no model results.")
        return
    results = ui.with_short_labels(results)
    ui.active_model_context(results, widget_key="overview_active_model")

    total = len(results)
    accepted = int(results["selection_status"].eq("accepted").sum()) if "selection_status" in results else 0
    warnings = (
        int(results["overfit_warning_flag"].fillna(False).astype(bool).sum())
        if "overfit_warning_flag" in results
        else 0
    )
    wr_holdout = results.get("wr_holdout", pd.Series(dtype="float64")).dropna()
    wr_holdout_range = (
        f"{int(wr_holdout.min()):,}–{int(wr_holdout.max()):,}"
        if not wr_holdout.empty
        else "-"
    )
    best = rank_models(results, metric="ranking_score", top_n=1)
    best_row = best.iloc[0] if not best.empty else None

    ui.kpi_row(
        [
            ("Configurations", f"{total:,}", "Models evaluated in this run"),
            (
                "Accepted",
                f"{accepted / total:.0%}" if total else "0%",
                f"{accepted:,} of {total:,} configurations",
            ),
            ("Overfit flags", f"{warnings:,}", "overfit_warning_flag set"),
            (
                "WR range",
                wr_holdout_range,
                "Positive denominators vary by dataset variant.",
            ),
            (
                "Best score",
                ui.fmt(best_row.get("ranking_score")) if best_row is not None else "-",
                best_row["short_label"] if best_row is not None else None,
            ),
        ]
    )

    candidates = select_diverse_top_models(results, top_n=10)
    st.subheader("Top 10 operational profiles")
    st.caption(
        "Distinct candidates selected across recall, AP, precision, calibrated-threshold "
        "and low-FPR profiles; WR@100 remains descriptive."
    )
    chart = charts.metric_bar(
        candidates,
        value_col="ranking_score",
        value_title="Ranking score",
        percent=False,
    )
    if chart is not None:
        st.altair_chart(chart)
    else:
        st.info("No chartable values for this run.")

    st.dataframe(
        ui.model_selection_view(candidates),
        hide_index=True,
        width="stretch",
        column_config=ui.model_selection_column_config(),
    )

    runs = data.runs()
    current = runs[runs["run_id"].eq(data.run_id())]
    if not current.empty:
        row = current.iloc[0]
        st.caption(
            f"Run `{row['run_id']}` | imported {row.get('imported_at', '-')} | source `{row.get('source_csv', '-')}`"
        )
