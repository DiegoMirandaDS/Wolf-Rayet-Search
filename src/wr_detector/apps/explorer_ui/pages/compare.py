"""Model comparison: filters, ranking table and comparative charts."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from wr_detector.apps.explorer_ui import charts, data, ui
from wr_detector.modeling.explorer import (
    RANKING_METRICS,
    assign_model_profiles,
    filter_results,
    rank_models,
)

MAX_COMPARED_MODELS = 8


def render() -> None:
    st.title("Compare models")
    results = data.results()
    if results.empty:
        st.warning("The selected run has no model results.")
        return
    results = ui.with_short_labels(results)
    ui.active_model_context(results, widget_key="compare_active_model")

    filtered = assign_model_profiles(_filters(results))
    if filtered.empty:
        st.warning("No models match the current filters.")
        return

    control_left, control_right = st.columns([3, 1])
    metric = control_left.selectbox(
        "Ranking metric",
        options=RANKING_METRICS,
        format_func=ui.metric_label,
        index=0,
    )
    top_n = control_right.slider(
        "Top N",
        min_value=1,
        max_value=len(filtered),
        value=min(16, len(filtered)),
    )
    ranked = rank_models(filtered, metric=metric, top_n=top_n)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))

    st.caption(
        f"{len(filtered):,} of {len(results):,} models match the filters; showing "
        f"{len(ranked)} by {ui.metric_label(metric)}. WR@100 is descriptive; relative "
        "recall is used when holdout WR totals differ."
    )

    compared = _ranking_table(ranked, metric)
    if compared.empty:
        st.info("Check one or more rows in the table to populate the comparison charts.")
        return
    if len(compared) > MAX_COMPARED_MODELS:
        st.warning(
            f"{len(compared)} rows are checked. Charts use the first "
            f"{MAX_COMPARED_MODELS} checked models to remain legible."
        )
        compared = compared.head(MAX_COMPARED_MODELS)
    compared["chart_label"] = compared.apply(ui.chart_label, axis=1)

    _active_from_compared(compared)
    with st.expander("How to read the comparison", expanded=False):
        st.markdown(
            """
- **Recall @100/@50** divides recovered WR by the model's own positive holdout denominator. **WR@100** is retained only as a descriptive count.
- **Precision @100** measures candidate-list purity. **Threshold precision/recall** are evaluated on holdout at the calibrated threshold.
- **FP / FPR** use the holdout negatives; threshold-calibration pass rate remains a separate stress-test metric.
- **Ranking score** combines Recall @100, AP, Precision @100, Recall @50 and calibrated-threshold precision/recall, with a soft holdout-FPR penalty.
- **AP gap** is train average precision minus cross-validated average precision. Smaller is better; a large positive gap suggests the fit does not generalize.
- **Risk** combines normalized Train-CV F2 and AP gaps, a low holdout/CV AP ratio, and weak low-budget/low-FPR recovery. It is a diagnostic score, not a probability; lower is better.
- The arrow in a table header only shows the current table sort direction. It does not mean that the metric itself should be minimized or maximized.
            """
        )

    use_pct = metric in {
        "holdout_recall_at_100",
        "holdout_precision_at_100",
        "holdout_recall_at_50",
        "holdout_precision_wr",
        "holdout_recall_wr",
        "holdout_fpr",
        "holdout_recall_at_fpr_0p005",
    }
    ks = [k for k in [10, 50, 100, 500, 1000] if f"holdout_wr_at_{k}_pct" in ranked.columns]

    primary_left, primary_right = st.columns([1, 1.45], gap="large")
    with primary_left:
        st.subheader(ui.metric_label(metric))
        chart = charts.metric_bar(
            compared,
            value_col=metric,
            value_title=ui.metric_label(metric),
            percent=use_pct,
            orientation="vertical",
            label_col="chart_label",
        )
        if chart is not None:
            st.altair_chart(chart)
    with primary_right:
        st.subheader("WR recovery by review budget")
        recovery = charts.recovery_lines(compared, ks=ks) if ks else None
        if recovery is not None:
            st.altair_chart(recovery)

    secondary_left, secondary_right = st.columns(2, gap="large")
    with secondary_left:
        st.subheader("Dataset x model matrix")
        heatmap = charts.dataset_heatmap(
            compared,
            value_col=metric,
            value_title=ui.metric_label(metric),
            percent=use_pct,
        )
        if heatmap is not None:
            st.altair_chart(heatmap)
        else:
            st.info("No values available for the matrix.")
    with secondary_right:
        st.subheader("Stability gaps")
        gaps = charts.stability_gap_bars(compared)
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


def _ranking_table(ranked: pd.DataFrame, metric: str) -> pd.DataFrame:
    view = ui.model_selection_view(ranked)
    if metric not in view.columns and metric in ranked.columns:
        view.insert(min(4, len(view.columns)), metric, ranked[metric])
    view.insert(0, "result_id", ranked["result_id"].astype(str))

    column_config = ui.model_selection_column_config()
    column_config["result_id"] = None
    if metric not in column_config and metric in view.columns:
        if metric in {
            "holdout_recall_at_fpr_0p005",
        }:
            column_config[metric] = st.column_config.ProgressColumn(
                ui.metric_label(metric),
                min_value=0.0,
                max_value=1.0,
                format="percent",
            )
        else:
            column_config[metric] = st.column_config.NumberColumn(
                ui.metric_label(metric),
                format="%.4f",
            )

    event = st.dataframe(
        view,
        hide_index=True,
        width="stretch",
        column_config=column_config,
        on_select="rerun",
        selection_mode="multi-row",
        key="compare_ranking_table",
    )
    selected_rows = event.selection.rows if event.selection else []
    valid_rows = [index for index in selected_rows if 0 <= index < len(ranked)]
    return ranked.iloc[valid_rows].copy()


def _active_from_compared(compared: pd.DataFrame) -> None:
    """Offer an explicit promotion without coupling row checks to active state."""
    options = compared["result_id"].astype(str).tolist()
    labels = {
        str(row.result_id): str(row.short_label)
        for row in compared.itertuples(index=False)
    }
    control, action = st.columns([4, 1])
    chosen = control.selectbox(
        "Active model from checked rows",
        options=options,
        format_func=lambda value: labels.get(value, value),
        key="compare_active_from_checked",
        label_visibility="collapsed",
    )
    if action.button(
        "Set active",
        key="compare_set_active",
        width="stretch",
    ):
        st.session_state[ui.SELECTED_MODEL_KEY] = chosen
        st.rerun()
    st.caption(f"{len(compared)} checked · checks control charts only")
