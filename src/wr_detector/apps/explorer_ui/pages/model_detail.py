"""Single-model inspection: metrics, stability, importance and live curves."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from wr_detector.apps.explorer_ui import charts, data, ui
from wr_detector.modeling.cases import case_confusion, precision_recall_points, roc_points
from wr_detector.modeling.explorer import parse_best_params, resolve_artifact_path


ARTIFACT_PATH_COLUMNS = [
    ("Model", "model_path"),
    ("PR curve figure", "holdout_pr_curve_path"),
    ("ROC curve figure", "holdout_roc_curve_path"),
    ("Confusion matrix figure", "holdout_confusion_matrix_path"),
    ("Feature importance figure", "feature_importance_figure_path"),
]


def render() -> None:
    st.title("Model detail")
    results = data.results()
    if results.empty:
        st.warning("The selected run has no model results.")
        return

    selected = ui.select_model(results)
    st.markdown(f"**{selected['short_label']}** &nbsp; {ui.status_badge(selected.get('selection_status'))}")
    st.caption(str(selected.get("model_label", "")))

    ui.kpi_row(
        [
            ("WR@10", ui.fmt_count_pct(selected.get("holdout_wr_at_10"), selected.get("holdout_wr_at_10_pct")), None),
            ("WR@50", ui.fmt_count_pct(selected.get("holdout_wr_at_50"), selected.get("holdout_wr_at_50_pct")), None),
            ("WR@100", ui.fmt_count_pct(selected.get("holdout_wr_at_100"), selected.get("holdout_wr_at_100_pct")), None),
            ("AP", ui.fmt(selected.get("holdout_average_precision")), "Holdout average precision"),
            ("Recall @FPR 0.5%", ui.fmt(selected.get("holdout_recall_at_fpr_0p005")), None),
            ("Overfit risk", ui.fmt(selected.get("overfit_risk_score"), digits=2), "Composite overfit risk score"),
        ]
    )

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Train / CV / holdout stability")
        split_chart = charts.split_metric_bars(selected)
        if split_chart is not None:
            st.altair_chart(split_chart)
        else:
            st.info("No per-split metric columns available.")
    with right:
        st.subheader("Feature importance")
        importance = data.feature_importance(str(selected["result_id"]))
        importance_chart = charts.importance_bars(importance)
        if importance_chart is not None:
            st.altair_chart(importance_chart)
        else:
            st.info("No feature importance rows are synchronized for this model.")

    _curves_section(selected)

    st.subheader("Hyperparameters")
    params = parse_best_params(selected.get("bayes_best_params"))
    if params:
        st.code(json.dumps(params, indent=2, default=str), language="json")
    else:
        st.info("No tuned hyperparameters recorded.")

    st.subheader("Prediction score summary")
    summary = data.prediction_summary(str(selected["result_id"]))
    if summary.empty:
        st.info("No synchronized prediction rows are available for this model.")
    else:
        st.dataframe(summary, hide_index=True, width="stretch")

    _artifact_section(selected)


def _curves_section(selected: pd.Series) -> None:
    st.subheader("Holdout ranking curves")
    cases = data.cases(str(selected["result_id"]), "holdout")
    if cases.empty:
        st.info("No synchronized holdout predictions; curves are unavailable.")
        return
    threshold = cases["threshold"].dropna().iloc[0] if cases["threshold"].notna().any() else None
    left, mid, right = st.columns([1.2, 1.2, 1], gap="large")
    with left:
        st.caption(f"Precision-recall · AP {ui.fmt(selected.get('holdout_average_precision'))}")
        pr_chart = charts.precision_recall_chart(precision_recall_points(cases), threshold=threshold)
        if pr_chart is not None:
            st.altair_chart(pr_chart)
    with mid:
        st.caption(f"ROC · AUC {ui.fmt(selected.get('holdout_roc_auc'))}")
        roc = charts.roc_chart(roc_points(cases), threshold=threshold)
        if roc is not None:
            st.altair_chart(roc)
    with right:
        st.caption(f"Confusion at threshold {ui.fmt(threshold, digits=4)}")
        st.altair_chart(charts.confusion_matrix_chart(case_confusion(cases)))
    st.caption("Curves are computed live from synchronized per-source predictions; the red point marks the selected operating threshold.")


def _artifact_section(selected: pd.Series) -> None:
    with st.expander("Artifact files on disk", expanded=False):
        for label, column in ARTIFACT_PATH_COLUMNS:
            raw = selected.get(column)
            path = resolve_artifact_path(raw)
            if path is None:
                continue
            status = "✓" if path.exists() else "missing —"
            st.markdown(f"**{label}** {status} `{raw}`")
