"""Run overview: headline numbers and dataset winners."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from wr_detector.apps.explorer_ui import charts, data, ui
from wr_detector.modeling.explorer import best_models_by_dataset, rank_models


def render() -> None:
    st.title("Run overview")
    results = data.results()
    if results.empty:
        st.warning("The selected run has no model results.")
        return
    results = ui.with_short_labels(results)

    total = len(results)
    accepted = int(results["selection_status"].eq("accepted").sum()) if "selection_status" in results else 0
    warnings = (
        int(results["overfit_warning_flag"].fillna(False).astype(bool).sum())
        if "overfit_warning_flag" in results
        else 0
    )
    wr_holdout = results["wr_holdout"].dropna()
    best = rank_models(results, metric="holdout_wr_at_100", top_n=1)
    best_row = best.iloc[0] if not best.empty else None

    ui.kpi_row(
        [
            ("Configurations", f"{total:,}", "Models evaluated in this run"),
            ("Accepted", f"{accepted:,} ({accepted / total:.0%})" if total else "0", "selection_status = accepted"),
            ("Overfit warnings", f"{warnings:,}", "overfit_warning_flag set"),
            ("WR in holdout", f"{int(wr_holdout.iloc[0]):,}" if not wr_holdout.empty else "-", "Positive holdout sample size"),
            (
                "Best WR@100",
                ui.fmt_count_pct(best_row.get("holdout_wr_at_100"), best_row.get("holdout_wr_at_100_pct")) if best_row is not None else "-",
                best_row["short_label"] if best_row is not None else None,
            ),
        ]
    )

    st.subheader("Top models by WR recovered @100")
    ranked = rank_models(results, metric="holdout_wr_at_100", top_n=10)
    chart = charts.metric_bar(
        ranked,
        value_col="holdout_wr_at_100_pct" if "holdout_wr_at_100_pct" in ranked.columns else "holdout_wr_at_100",
        value_title="WR holdout recovered @100",
        percent="holdout_wr_at_100_pct" in ranked.columns,
    )
    if chart is not None:
        st.altair_chart(chart)
    else:
        st.info("No chartable values for this run.")

    st.subheader("Best model per dataset variant")
    winners = best_models_by_dataset(results, metric="holdout_wr_at_100")
    view = winners[
        [
            column
            for column in [
                "dataset_variant",
                "short_label",
                "selection_status",
                "holdout_wr_at_100",
                "holdout_wr_at_100_pct",
                "holdout_average_precision",
                "overfit_risk_score",
            ]
            if column in winners.columns
        ]
    ]
    st.dataframe(
        view,
        hide_index=True,
        width="stretch",
        column_config={
            "dataset_variant": st.column_config.TextColumn("dataset"),
            "short_label": st.column_config.TextColumn("model"),
            "selection_status": st.column_config.TextColumn("status"),
            "holdout_wr_at_100": st.column_config.NumberColumn("WR@100"),
            "holdout_wr_at_100_pct": st.column_config.ProgressColumn("recovery @100", min_value=0.0, max_value=1.0, format="percent"),
            "holdout_average_precision": st.column_config.NumberColumn("AP", format="%.4f"),
            "overfit_risk_score": st.column_config.NumberColumn("risk", format="%.2f"),
        },
    )

    runs = data.runs()
    current = runs[runs["run_id"].eq(data.run_id())]
    if not current.empty:
        row = current.iloc[0]
        st.caption(
            f"Run `{row['run_id']}` | imported {row.get('imported_at', '-')} | source `{row.get('source_csv', '-')}`"
        )
