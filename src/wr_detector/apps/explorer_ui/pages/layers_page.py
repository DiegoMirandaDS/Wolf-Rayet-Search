"""Validation layers: second-layer (and future layers) run review."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from wr_detector.apps.explorer_ui import charts, data, ui
from wr_detector.modeling.layers import VALIDATION_LAYERS


RESULT_COLUMNS = [
    ("dataset_variant", "dataset"),
    ("feature_set", "features"),
    ("method", "method"),
    ("subtype", "subtype"),
    ("status", "status"),
    ("holdout_positive_retention", "WR retained"),
    ("holdout_negative_pass_rate", "holdout neg pass"),
    ("threshold_calibration_negative_pass_rate", "calib neg pass"),
    ("holdout_average_precision", "AP"),
    ("holdout_roc_auc", "ROC AUC"),
    ("threshold", "threshold"),
    ("train_positive_count", "train WR"),
    ("holdout_positive_count", "holdout WR"),
]


def render() -> None:
    st.title("Validation layers")
    st.caption(
        "Post-first-stage validators re-rank or sanity-check top candidates. "
        "They are compatibility scorers, not hard rejection gates. "
        "This audit did not replace the operational first-stage ranking."
    )

    layer = st.selectbox(
        "Layer",
        options=VALIDATION_LAYERS,
        format_func=lambda value: value.title,
    )
    runs = data.layer_runs(layer.key)
    if runs.empty:
        st.info(
            f"No {layer.title} runs found yet. Train one with:\n\n```\n{layer.train_command}\n```"
        )
        return

    labels = {
        row.run_id: f"{row.run_id} | {row.modified_at:%Y-%m-%d %H:%M} UTC | {row.source}"
        for row in runs.itertuples(index=False)
    }
    run_id = st.selectbox("Layer run", options=list(labels), format_func=lambda value: labels.get(value, value))
    selected_run = runs[runs["run_id"].eq(run_id)].iloc[0]
    if selected_run["source"] == "csv":
        st.warning(
            "This run exists only as CSV and is not in the canonical history DB. "
            f"Sync it with `wr-detector sync-second-layer-history --run-id {run_id}`."
        )
    results = data.layer_results(layer.key, run_id)
    if results.empty:
        st.warning("The selected run has no result rows.")
        return
    _lineage_section(results)

    total = len(results)
    fitted = int(results["status"].eq("accepted").sum()) if "status" in results else 0
    retention = results.get("holdout_positive_retention")
    calib_pass = results.get("threshold_calibration_negative_pass_rate")
    ui.kpi_row(
        [
            ("Validators", f"{total:,}", "variant x features x method x subtype"),
            (
                "Fitted",
                f"{fitted:,} ({fitted / total:.0%})" if total else "0",
                "accepted means fit completed; it is not an operational recommendation",
            ),
            ("Best WR retention", ui.fmt_pct(retention.max()) if retention is not None else "-", "Holdout positives kept at threshold"),
            ("Lowest calib pass", ui.fmt_pct(calib_pass.min()) if calib_pass is not None else "-", "Calibration negatives passing (lower is better)"),
        ]
    )

    filtered = _filters(results)
    if filtered.empty:
        st.warning("No validators match the current filters.")
        return

    st.subheader("Retention vs negative pass-rate")
    scatter = charts.retention_tradeoff_scatter(filtered)
    if scatter is not None:
        st.altair_chart(scatter)
        st.caption("The ideal validator sits top-left: keeps WR while passing few calibration negatives.")
    else:
        st.info("No retention/pass-rate columns available for this run.")

    st.subheader("Validator results")
    _results_table(filtered)


def _lineage_section(results: pd.DataFrame) -> None:
    lineage_columns = [
        "dataset_path",
        "dataset_sha256",
        "models_config_sha256",
        "holdout_fraction",
        "require_color_locus_keep",
        "color_locus_excluded_rows",
    ]
    present = [
        column
        for column in lineage_columns
        if column in results.columns and results[column].notna().any()
    ]
    if not present:
        st.caption(
            "Warning: this run predates data-lineage tracking. It does not record which reduced datasets "
            "or split policy it was trained against. Prefer retraining before pairing it with a first-layer run."
        )
        return
    with st.expander("Data lineage", expanded=False):
        lineage = results[present].drop_duplicates().copy()
        for column in ["dataset_sha256", "models_config_sha256"]:
            if column in lineage.columns:
                lineage[column] = lineage[column].astype(str).str.slice(0, 12)
        st.dataframe(lineage, hide_index=True, width="stretch")
        st.caption(
            "Hashes of the reduced datasets and split policy this run consumed. "
            "Pair this layer with a first-layer run only when both used the same dataset hashes."
        )


def _filters(results: pd.DataFrame) -> pd.DataFrame:
    with st.expander("Filters", expanded=False):
        columns = st.columns(4)
        filtered = results
        for container, column, label in [
            (columns[0], "dataset_variant", "Dataset variant"),
            (columns[1], "feature_set", "Feature set"),
            (columns[2], "method", "Method"),
            (columns[3], "subtype", "Subtype"),
        ]:
            if column not in filtered.columns:
                continue
            options = sorted(filtered[column].fillna("missing").astype(str).unique().tolist())
            selected = container.multiselect(label, options=options)
            if selected:
                filtered = filtered[filtered[column].fillna("missing").astype(str).isin(selected)]
    return filtered.reset_index(drop=True)


def _results_table(filtered: pd.DataFrame) -> None:
    columns = [column for column, _ in RESULT_COLUMNS if column in filtered.columns]
    view = filtered[columns].copy()
    labels = dict(RESULT_COLUMNS)
    column_config: dict[str, object] = {}
    for column in columns:
        label = labels.get(column, column)
        if column in {"holdout_positive_retention", "holdout_negative_pass_rate", "threshold_calibration_negative_pass_rate"}:
            column_config[column] = st.column_config.ProgressColumn(label, min_value=0.0, max_value=1.0, format="percent")
        elif pd.api.types.is_float_dtype(view[column]):
            column_config[column] = st.column_config.NumberColumn(label, format="%.4f")
        else:
            view[column] = view[column].astype("string")
            column_config[column] = st.column_config.TextColumn(label)
    st.dataframe(view, hide_index=True, width="stretch", column_config=column_config)
