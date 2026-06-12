"""Model comparison: filters, ranking table and comparative charts."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from wr_detector.apps.explorer_ui import charts, data, ui
from wr_detector.modeling.explorer import RANKING_METRICS, filter_results, rank_models


TABLE_COLUMNS = [
    ("short_label", "model"),
    ("selection_status", "status"),
    ("holdout_wr_at_10", "WR@10"),
    ("holdout_wr_at_50", "WR@50"),
    ("holdout_wr_at_100", "WR@100"),
    ("holdout_wr_at_100_pct", "recovery @100"),
    ("holdout_average_precision", "AP"),
    ("holdout_recall_at_fpr_0p005", "recall @FPR 0.5%"),
    ("train_cv_gap_average_precision", "AP gap"),
    ("overfit_risk_score", "risk"),
    ("hyperparams_compact", "hyperparameters"),
]


def render() -> None:
    st.title("Compare models")
    results = data.results()
    if results.empty:
        st.warning("The selected run has no model results.")
        return
    results = ui.with_short_labels(results)

    filtered = _filters(results)
    if filtered.empty:
        st.warning("No models match the current filters.")
        return

    control_left, control_right = st.columns([3, 1])
    metric = control_left.selectbox("Ranking metric", options=RANKING_METRICS, format_func=ui.metric_label, index=2)
    top_n = control_right.slider("Top N", min_value=3, max_value=max(3, len(filtered)), value=min(16, len(filtered)))
    ranked = rank_models(filtered, metric=metric, top_n=top_n)

    st.caption(f"{len(filtered):,} of {len(results):,} models match the filters; showing the top {len(ranked)} by {ui.metric_label(metric)}.")

    _ranking_table(ranked, metric)

    st.subheader(ui.metric_label(metric))
    pct_col = f"{metric}_pct" if metric.startswith("holdout_wr_at_") else None
    use_pct = pct_col is not None and pct_col in ranked.columns
    chart = charts.metric_bar(
        ranked,
        value_col=pct_col if use_pct else metric,
        value_title=ui.metric_label(metric),
        percent=use_pct,
    )
    if chart is not None:
        st.altair_chart(chart)

    st.subheader("WR recovery by review budget")
    ks = [k for k in [10, 50, 100, 500, 1000] if f"holdout_wr_at_{k}_pct" in ranked.columns]
    recovery = charts.recovery_lines(ranked.head(6), ks=ks) if ks else None
    if recovery is not None:
        st.altair_chart(recovery)

    tab_matrix, tab_gaps = st.tabs(["Dataset × model matrix", "Stability gaps"])
    with tab_matrix:
        heatmap = charts.dataset_heatmap(
            filtered,
            value_col=pct_col if use_pct else metric,
            value_title=ui.metric_label(metric),
            percent=use_pct,
        )
        if heatmap is not None:
            st.altair_chart(heatmap)
        else:
            st.info("No values available for the matrix.")
    with tab_gaps:
        gaps = charts.stability_gap_bars(ranked.head(12))
        if gaps is not None:
            st.altair_chart(gaps)
        else:
            st.info("No stability gap columns available.")


def _filters(results: pd.DataFrame) -> pd.DataFrame:
    with st.expander("Filters", expanded=False):
        row1 = st.columns(4)
        selections = {
            "dataset_families": _multiselect(row1[0], "Family", results, "dataset_family"),
            "models": _multiselect(row1[1], "Model", results, "model"),
            "samplers": _multiselect(row1[2], "Sampler", results, "sampler"),
            "feature_sets": _multiselect(row1[3], "Feature set", results, "feature_set"),
        }
        row2 = st.columns([2, 1, 1])
        selections["dataset_variants"] = _multiselect(row2[0], "Dataset variant", results, "dataset_variant")
        selections["negative_ratio_labels"] = _multiselect(row2[1], "Negative ratio", results, "negative_ratio_label")
        accepted_only = row2[2].toggle("Accepted only", value=False)
    return filter_results(results, include_warnings=not accepted_only, **selections)


def _multiselect(container, label: str, df: pd.DataFrame, column: str) -> list[str]:
    if column not in df.columns:
        return []
    options = sorted(df[column].fillna("missing").astype(str).unique().tolist())
    return container.multiselect(label, options=options)


def _ranking_table(ranked: pd.DataFrame, metric: str) -> None:
    columns = [column for column, _ in TABLE_COLUMNS if column in ranked.columns]
    if metric not in columns and metric in ranked.columns:
        columns.insert(2, metric)
    view = ranked[["result_id", *columns]].copy()
    view.insert(1, "rank", range(1, len(view) + 1))

    labels = dict(TABLE_COLUMNS)
    column_config: dict[str, object] = {
        "result_id": None,
        "rank": st.column_config.NumberColumn("#", width="small"),
    }
    for column in columns:
        label = labels.get(column, ui.metric_label(column))
        if column.endswith("_pct"):
            column_config[column] = st.column_config.ProgressColumn(label, min_value=0.0, max_value=1.0, format="percent")
        elif pd.api.types.is_float_dtype(view[column]):
            column_config[column] = st.column_config.NumberColumn(label, format="%.4f")
        else:
            column_config[column] = st.column_config.TextColumn(label)

    event = st.dataframe(
        view,
        hide_index=True,
        width="stretch",
        column_config=column_config,
        on_select="rerun",
        selection_mode="single-row",
        key="compare_ranking_table",
    )
    selected_rows = event.selection.rows if event.selection else []
    if selected_rows:
        result_id = str(view.iloc[selected_rows[0]]["result_id"])
        st.session_state[ui.SELECTED_MODEL_KEY] = result_id
        st.caption(
            f"Selected **{view.iloc[selected_rows[0]]['short_label']}** — open *Model detail*, *Case review* or *Statistics* to inspect it."
        )
