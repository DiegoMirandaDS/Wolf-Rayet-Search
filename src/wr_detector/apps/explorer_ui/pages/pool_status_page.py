"""Operational status of the exact-union pool and mass scoring."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from wr_detector.apps.explorer_ui import candidate_plots, data, ui


def render() -> None:
    st.title("Prediction pool status")
    st.caption(
        "Read-only integrity, coverage and lineage for the exact-union pool, "
        "five-model scoring run and persisted candidate review."
    )
    try:
        availability = data.candidate_availability()
    except Exception as exc:
        st.info(f"Candidate configuration is unavailable: {exc}")
        return
    if not availability["ready"]:
        st.info(
            "Prediction-pool review artifacts are incomplete. Models and "
            "validation pages remain available."
        )
        with st.expander("Expected local artifacts", expanded=True):
            st.json(availability)
        return

    configured_pool = data.configured_pool_build_id()
    configured_scoring = data.configured_scoring_run_id()
    pool_options = data.available_pool_build_ids()
    scoring_options = data.available_scoring_run_ids()
    pool_labels = {
        value: (
            f"{value} (configured)"
            if configured_pool and value == configured_pool
            else f"{value} (historical)"
        )
        for value in pool_options
    }
    scoring_labels = {
        value: (
            f"{value} (configured)"
            if configured_scoring and value == configured_scoring
            else f"{value} (historical)"
        )
        for value in scoring_options
    }
    controls = st.columns([1.4, 1.4])
    with controls[0]:
        selected_pool = st.selectbox(
            "Pool build",
            options=pool_options,
            format_func=lambda value: pool_labels.get(value, value),
            index=0,
            help="Configured operational build first; historical builds are selectable but never mixed.",
        )
    with controls[1]:
        selected_scoring = st.selectbox(
            "Scoring run",
            options=scoring_options,
            format_func=lambda value: scoring_labels.get(value, value),
            index=0,
            help="Configured operational scoring run first; historical runs are selectable but never mixed.",
        )

    pool_runs, tiles = data.pool_status(selected_pool)
    scoring_runs, scored_models = data.scoring_status(selected_scoring)
    review_runs = data.candidate_review_runs()
    summary = data.candidate_summary()
    dispositions = data.candidate_dispositions()
    models = data.candidate_models()
    if pool_runs.empty or scoring_runs.empty or not summary:
        st.warning("Prediction-pool databases exist but lack synchronized status tables.")
        return

    pool = pool_runs.iloc[0]
    scoring = scoring_runs.iloc[0]
    operational_pool = bool(configured_pool) and str(pool.get("pool_build_id", "")) == str(configured_pool)
    operational_scoring = (
        bool(configured_scoring)
        and str(scoring.get("scoring_run_id", "")) == str(configured_scoring)
    )
    scope_caption = (
        "Operational scope: configured pool build and scoring run. "
        "Historical selections above replace the operational view for this session."
    )
    if not operational_pool or not operational_scoring:
        scope_caption = (
            "Historical scope selected. Candidate-review summary and model "
            "metrics below still reflect the configured operational review."
        )
    st.caption(scope_caption)
    ui.kpi_row(
        [
            (
                "Pool build",
                str(pool.get("status", "unknown")),
                str(pool.get("pool_build_id", "")),
            ),
            (
                "Completed tiles",
                f"{int(pool.get('completed_tiles', 0)):,} / "
                f"{int(pool.get('registered_tiles', 0)):,}",
                None,
            ),
            (
                "Operational sources",
                f"{int(pool.get('operational_rows', 0)):,}",
                "Known sources already excluded",
            ),
        ]
    )
    ui.kpi_row(
        [
            (
                "Scoring units",
                f"{int(scored_models['completed_units'].sum()):,} / "
                f"{int(scored_models['work_units'].sum()):,}",
                str(scoring.get("scoring_run_id", "")),
            ),
            (
                "Review candidates",
                f"{int(summary.get('review_rows', 0)):,}",
                f"{int(summary.get('consensus_rows', 0)):,} unique in RRF union",
            ),
        ]
    )

    flow = pd.DataFrame(
        [
            {"stage": "Acquired", "sources": pool.get("acquired_rows")},
            {"stage": "Accepted union", "sources": pool.get("accepted_union_rows")},
            {"stage": "Known excluded", "sources": pool.get("known_excluded_rows")},
            {"stage": "Operational", "sources": pool.get("operational_rows")},
        ]
    )
    left, right = st.columns([1.55, 1], gap="large")
    with left:
        st.subheader("Exact-union sky coverage")
        chart = candidate_plots.pool_tile_map(tiles)
        if chart is None:
            st.info("Tile geometry is unavailable.")
        else:
            st.altair_chart(chart)
            st.caption(
                "Color is the operational-to-acquired row fraction per tile; "
                "coverage status remains authoritative in DuckDB."
            )
    with right:
        st.subheader("Source reconciliation")
        st.dataframe(
            flow,
            hide_index=True,
            width="stretch",
            column_config={
                "stage": st.column_config.TextColumn("Stage"),
                "sources": st.column_config.NumberColumn("Sources", format="%d"),
            },
        )
        st.subheader("SIMBAD review composition")
        chart = candidate_plots.disposition_bar(dispositions)
        if chart is None:
            st.info("No candidate disposition rows.")
        else:
            st.altair_chart(chart)

    st.subheader("Scoring coverage by model")
    coverage = candidate_plots.scoring_coverage(scored_models)
    if coverage is None:
        st.info("Per-model scoring rows are unavailable.")
    else:
        st.altair_chart(coverage)
    scoring_table = scored_models[
        [
            column
            for column in [
                "model_label",
                "dataset_variant",
                "work_units",
                "completed_units",
                "eligible_rows",
                "scored_rows",
                "missing_feature_rows",
                "scored_fraction",
                "missing_feature_fraction",
            ]
            if column in scored_models.columns
        ]
    ]
    st.dataframe(
        scoring_table,
        hide_index=True,
        width="stretch",
        column_config={
            "model_label": st.column_config.TextColumn("Model role"),
            "dataset_variant": st.column_config.TextColumn("Variant"),
            "work_units": st.column_config.NumberColumn("Units", format="%d"),
            "completed_units": st.column_config.NumberColumn("Complete", format="%d"),
            "eligible_rows": st.column_config.NumberColumn("Eligible", format="%d"),
            "scored_rows": st.column_config.NumberColumn("Scored", format="%d"),
            "missing_feature_rows": st.column_config.NumberColumn("Missing", format="%d"),
            "scored_fraction": st.column_config.ProgressColumn(
                "Scored / eligible",
                min_value=0.0,
                max_value=1.0,
                format="percent",
            ),
            "missing_feature_fraction": st.column_config.ProgressColumn(
                "Missing features",
                min_value=0.0,
                max_value=1.0,
                format="percent",
            ),
        },
    )

    st.subheader("Five-model operating set")
    _model_metrics(models)

    with st.expander("Data lineage", expanded=False):
        st.markdown("**Candidate review run**")
        st.dataframe(review_runs, hide_index=True, width="stretch")
        st.markdown("**Scoring run**")
        st.dataframe(scoring_runs, hide_index=True, width="stretch")
        st.markdown("**Pool build**")
        st.dataframe(pool_runs, hide_index=True, width="stretch")
        st.caption(
            f"Candidate config: `{data.candidate_config_path()}`  \n"
            f"Candidate DB: `{availability['candidate_db']}`  \n"
            f"Scoring DB: `{availability['scoring_db']}`  \n"
            f"Pool DB: `{availability['pool_db']}`"
        )


def _model_metrics(models: pd.DataFrame) -> None:
    if models.empty:
        st.info("Candidate model metrics are unavailable.")
        return
    columns = [
        column
        for column in [
            "role",
            "dataset_variant",
            "model",
            "sampler",
            "wr_holdout",
            "holdout_average_precision",
            "holdout_precision_at_100",
            "holdout_recall_at_100",
            "holdout_recall_at_fpr_0p001",
            "threshold_calibration_negative_pass_rate",
            "overfit_risk_score",
            "selection_status",
        ]
        if column in models.columns
    ]
    st.dataframe(
        models[columns],
        hide_index=True,
        width="stretch",
        column_config={
            "role": st.column_config.TextColumn("Role"),
            "dataset_variant": st.column_config.TextColumn("Variant"),
            "model": st.column_config.TextColumn("Estimator"),
            "sampler": st.column_config.TextColumn("Sampler"),
            "wr_holdout": st.column_config.NumberColumn("WR holdout", format="%d"),
            "holdout_average_precision": st.column_config.NumberColumn(
                "AP",
                format="%.4f",
            ),
            "holdout_precision_at_100": st.column_config.ProgressColumn(
                "Precision @100",
                min_value=0.0,
                max_value=1.0,
                format="percent",
            ),
            "holdout_recall_at_100": st.column_config.ProgressColumn(
                "Recall @100",
                min_value=0.0,
                max_value=1.0,
                format="percent",
            ),
            "holdout_recall_at_fpr_0p001": st.column_config.ProgressColumn(
                "Recall @FPR 0.1%",
                min_value=0.0,
                max_value=1.0,
                format="percent",
            ),
            "threshold_calibration_negative_pass_rate": st.column_config.ProgressColumn(
                "Calibration negatives passing",
                min_value=0.0,
                max_value=1.0,
                format="percent",
            ),
            "overfit_risk_score": st.column_config.NumberColumn(
                "Overfit risk",
                format="%.2f",
            ),
            "selection_status": st.column_config.TextColumn("Status"),
        },
    )
