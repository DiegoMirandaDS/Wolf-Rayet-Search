"""Joint first-stage and validation-layer candidate-stack review."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from wr_detector.apps.explorer_ui import charts, data, ui
from wr_detector.modeling.layers import VALIDATION_LAYERS


CASE_VIEWS = [
    "Top input candidates",
    "WR rejected by validator",
    "Negatives removed",
    "Compatibility passers",
    "All sources",
]


def render() -> None:
    st.title("Candidate stack")
    st.caption(
        "Joint holdout audit of a first-stage ranking and a WN/WC compatibility pair. "
        "The recommended operational mode remains audit-only until a stack improves fixed-budget recovery."
    )
    results = data.results()
    if results.empty:
        st.warning("The selected run has no model results.")
        return
    selected_model = ui.active_model_context(results, widget_key="stack_active_model")
    result_id = str(selected_model["result_id"])

    layer = VALIDATION_LAYERS[0]
    compatible_runs = data.compatible_stack_runs(layer.key, result_id)
    if compatible_runs.empty:
        st.info(
            "No synchronized validation-layer run has a complete WN/WC pair with matching "
            "dataset and config hashes for this model."
        )
        return

    run_labels = {
        row.run_id: f"{row.run_id} | {int(row.pair_count)} pairs"
        for row in compatible_runs.itertuples(index=False)
    }
    controls = st.columns([1.8, 1.8, 1, 1])
    layer_run_id = controls[0].selectbox(
        "Validation run",
        options=list(run_labels),
        format_func=lambda value: run_labels.get(value, value),
        key="stack_layer_run",
    )
    pairs = data.stack_validator_pairs(layer.key, layer_run_id, result_id)
    if pairs.empty:
        st.warning("The selected validation run has no complete compatible WN/WC pair.")
        return
    pair_labels = dict(zip(pairs["pair_key"], pairs["pair_label"], strict=False))
    pair_key = controls[1].selectbox(
        "WN/WC validator pair",
        options=pairs["pair_key"].tolist(),
        format_func=lambda value: pair_labels.get(value, value),
        index=_pair_index(pairs["pair_key"].tolist()),
        key="stack_pair",
    )
    pair = pairs[pairs["pair_key"].eq(pair_key)].iloc[0]

    try:
        evaluation = data.candidate_stack(
            layer.key,
            layer_run_id,
            result_id,
            str(pair["feature_set"]),
            str(pair["method"]),
        )
    except FileNotFoundError:
        st.info(
            "Candidate Stack requires optional second-layer model binaries and "
            "reduced datasets. They are not included in the reviewer bundle "
            "because this layer was retained only as an aggregate audit. "
            "Its stored metrics remain available under Validation Layers."
        )
        return
    except ValueError as exc:
        st.warning(f"Candidate-stack evaluation is unavailable: {exc}")
        return
    recovery = evaluation["recovery"]
    tradeoff = evaluation["tradeoff"]
    if recovery.empty or tradeoff.empty:
        st.info("No synchronized holdout predictions are available for this stack.")
        return

    budgets = tradeoff["input_budget"].astype(int).tolist()
    input_budget = int(
        controls[2].selectbox(
            "Input top M",
            options=budgets,
            index=_budget_index(budgets, 500),
            key="stack_input_budget",
        )
    )
    review_budgets = sorted(recovery["budget"].astype(int).unique().tolist())
    review_budget = int(
        controls[3].selectbox(
            "Review budget K",
            options=review_budgets,
            index=_budget_index(review_budgets, 100),
            key="stack_review_budget",
        )
    )

    trade_row = tradeoff[tradeoff["input_budget"].eq(input_budget)].iloc[0]
    baseline = recovery[
        recovery["policy"].eq("First stage") & recovery["budget"].eq(review_budget)
    ].iloc[0]
    stacked = recovery[
        recovery["policy"].eq("Pass-first") & recovery["budget"].eq(review_budget)
    ].iloc[0]
    delta = int(stacked["wr_recovered"] - baseline["wr_recovered"])

    ui.kpi_row(
        [
            (f"First-stage WR@{review_budget}", f"{int(baseline['wr_recovered']):,}", None),
            (
                f"Pass-first WR@{review_budget}",
                f"{int(stacked['wr_recovered']):,}",
                f"delta {delta:+d}",
            ),
            (
                f"WR retained in top {input_budget}",
                ui.fmt_pct(trade_row["wr_retention"]),
                f"{int(trade_row['wr_pass'])} of {int(trade_row['wr_input'])}",
            ),
            (
                f"Negatives removed in top {input_budget}",
                ui.fmt_pct(trade_row["negative_removal_rate"]),
                f"{int(trade_row['negatives_input'] - trade_row['negatives_pass'])} removed",
            ),
        ]
    )
    _assessment(delta, trade_row, review_budget)

    tab_joint, tab_subtypes, tab_cases, tab_lineage = st.tabs(
        ["Joint performance", "Subtype retention", "Candidate transitions", "Data lineage"]
    )
    with tab_joint:
        left, right = st.columns(2, gap="large")
        with left:
            st.subheader("Recovery at fixed review budget")
            chart = charts.stack_recovery_lines(recovery)
            if chart is not None:
                st.altair_chart(chart)
        with right:
            st.subheader("Compatibility trade-off")
            chart = charts.stack_tradeoff_bars(tradeoff)
            if chart is not None:
                st.altair_chart(chart)
        st.dataframe(
            tradeoff,
            hide_index=True,
            width="stretch",
            column_config={
                "input_budget": st.column_config.NumberColumn("input top M"),
                "wr_input": st.column_config.NumberColumn("WR input"),
                "wr_pass": st.column_config.NumberColumn("WR pass"),
                "wr_retention": st.column_config.ProgressColumn(
                    "WR retained", min_value=0.0, max_value=1.0, format="percent"
                ),
                "negatives_input": st.column_config.NumberColumn("negatives input"),
                "negatives_pass": st.column_config.NumberColumn("negatives pass"),
                "negative_pass_rate": st.column_config.ProgressColumn(
                    "negative pass", min_value=0.0, max_value=1.0, format="percent"
                ),
                "negative_removal_rate": st.column_config.ProgressColumn(
                    "negative removal", min_value=0.0, max_value=1.0, format="percent"
                ),
            },
        )
        st.caption(
            "Pass-first places candidates compatible with either WN or WC first, preserving the "
            "first-stage score order within pass/fail groups. It is an audit policy, not a selected production policy."
        )
    with tab_subtypes:
        _subtype_table(evaluation["subtypes"])
    with tab_cases:
        _case_table(evaluation["cases"], input_budget)
    with tab_lineage:
        _lineage(evaluation["lineage"], pair)


def _assessment(delta: int, trade_row: pd.Series, review_budget: int) -> None:
    retention = float(trade_row["wr_retention"])
    removal = float(trade_row["negative_removal_rate"])
    if delta > 0 and retention >= 0.95 and removal >= 0.30:
        st.success(
            f"Promising audit result: WR@{review_budget} improves by {delta}, with "
            f"{retention:.1%} WR retention and {removal:.1%} negative removal. "
            "Confirm with paired uncertainty before operational use."
        )
    elif delta <= 0:
        st.info(
            f"Audit-only recommended: pass-first changes WR@{review_budget} by {delta:+d}. "
            "Keep compatibility flags for review, but do not reject prediction-pool candidates."
        )
    else:
        st.warning(
            "The fixed-budget ranking improves, but WR retention or negative removal is too weak "
            "to justify an operational gate."
        )


def _subtype_table(subtypes: pd.DataFrame) -> None:
    if subtypes.empty:
        st.info("WR subtype information is unavailable from the reference database.")
        return
    st.dataframe(
        subtypes,
        hide_index=True,
        width="stretch",
        column_config={
            "input_budget": st.column_config.NumberColumn("input top M"),
            "wr_subtype": st.column_config.TextColumn("subtype"),
            "wr_input": st.column_config.NumberColumn("WR input"),
            "wr_pass": st.column_config.NumberColumn("WR pass"),
            "wr_retention": st.column_config.ProgressColumn(
                "WR retained", min_value=0.0, max_value=1.0, format="percent"
            ),
        },
    )
    st.caption("WO is reported as a stress-test group; no WO validator is trained.")


def _case_table(cases: pd.DataFrame, input_budget: int) -> None:
    if cases.empty:
        st.info("No per-source stack rows are available.")
        return
    controls = st.columns([2, 1])
    view_mode = controls[0].selectbox("Transition view", options=CASE_VIEWS)
    top_only = controls[1].toggle(f"Limit to input top {input_budget}", value=True)
    view = cases.copy()
    if top_only:
        view = view[view["rank_before"].le(input_budget)]
    if view_mode == "WR rejected by validator":
        view = view[view["target"].eq(1) & ~view["second_layer_pass"]]
    elif view_mode == "Negatives removed":
        view = view[view["target"].eq(0) & ~view["second_layer_pass"]]
    elif view_mode == "Compatibility passers":
        view = view[view["second_layer_pass"]]
    elif view_mode == "Top input candidates":
        view = view.sort_values("rank_before")
    columns = [
        column
        for column in [
            "object_name",
            "target",
            "wr_subtype",
            "simbad_main_type",
            "score",
            "rank_before",
            "rank_after",
            "rank_delta",
            "wn_pass",
            "wc_pass",
            "second_layer_pass",
        ]
        if column in view.columns
    ]
    st.dataframe(
        view[columns].head(500),
        hide_index=True,
        width="stretch",
        height=460,
        column_config={
            "object_name": st.column_config.TextColumn("object"),
            "target": st.column_config.NumberColumn("target"),
            "wr_subtype": st.column_config.TextColumn("WR subtype"),
            "simbad_main_type": st.column_config.TextColumn("SIMBAD type"),
            "score": st.column_config.NumberColumn("first score", format="%.4f"),
            "rank_before": st.column_config.NumberColumn("rank before"),
            "rank_after": st.column_config.NumberColumn("rank after"),
            "rank_delta": st.column_config.NumberColumn("rank gain"),
            "wn_pass": st.column_config.CheckboxColumn("WN pass"),
            "wc_pass": st.column_config.CheckboxColumn("WC pass"),
            "second_layer_pass": st.column_config.CheckboxColumn("either pass"),
        },
    )


def _lineage(lineage: pd.DataFrame, pair: pd.Series) -> None:
    if lineage.empty:
        st.info("No lineage fields are available.")
        return
    display = lineage.copy()
    for column in ["first_result_id", "dataset_sha256", "models_config_sha256"]:
        if column in display.columns:
            display[column] = display[column].astype(str).str.slice(0, 12)
    st.dataframe(display, hide_index=True, width="stretch")
    st.caption(
        f"Pair: {pair['method']} / {pair['feature_set']}. Full identifiers remain in DuckDB; "
        "the interface uses compact labels."
    )


def _budget_index(options: list[int], preferred: int) -> int:
    if preferred in options:
        return options.index(preferred)
    return max(0, len(options) - 1)


def _pair_index(options: list[str]) -> int:
    preferred = "enriched_tabular::gaussian_mixture"
    return options.index(preferred) if preferred in options else 0
