"""Shared widgets and formatting helpers for the Model Explorer."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from wr_detector.modeling.explorer import rank_models


METRIC_LABELS = {
    "holdout_wr_at_10": "WR recovered @10",
    "holdout_wr_at_50": "WR recovered @50",
    "holdout_wr_at_100": "WR recovered @100",
    "holdout_average_precision": "Average precision",
    "holdout_recall_at_fpr_0p005": "Recall at FPR 0.5%",
    "cv_train_gap_f2": "Train-CV F2 gap",
    "train_cv_gap_average_precision": "Train-CV AP gap",
    "overfit_risk_score": "Overfit risk score",
    "holdout_cv_gap_f2": "Holdout-CV F2 gap",
}

MODEL_ABBREVIATIONS = {
    "random_forest": "RF",
    "hist_gradient_boosting": "HGB",
    "xgboost": "XGB",
}

STATUS_BADGES = {
    "accepted": ":green-badge[accepted]",
    "overfit_warning": ":orange-badge[overfit warning]",
}

SELECTED_MODEL_KEY = "selected_model_result_id"


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric)


def with_short_labels(results: pd.DataFrame) -> pd.DataFrame:
    enriched = results.copy()
    enriched["short_label"] = enriched.apply(short_label, axis=1)
    duplicated = enriched["short_label"].duplicated(keep=False)
    if duplicated.any() and "negative_ratio_label" in enriched.columns:
        enriched.loc[duplicated, "short_label"] = (
            enriched.loc[duplicated, "short_label"]
            + " | "
            + enriched.loc[duplicated, "negative_ratio_label"].astype(str)
        )
    return enriched


def short_label(row: pd.Series) -> str:
    model = MODEL_ABBREVIATIONS.get(str(row.get("model")), str(row.get("model")))
    sampler = str(row.get("sampler", "")).replace("smote_enn", "SMOTE-ENN").replace("smote", "SMOTE")
    variant = str(row.get("dataset_variant", ""))
    feat = " +err" if bool(row.get("includes_parallax_error")) else ""
    return f"{model}/{sampler} | {variant}{feat}"


def status_badge(status: object) -> str:
    text = str(status)
    if text in STATUS_BADGES:
        return STATUS_BADGES[text]
    if "floor_not_met" in text:
        return f":red-badge[{text}]"
    return f":gray-badge[{text}]"


def select_model(results: pd.DataFrame, *, label: str = "Model") -> pd.Series:
    """Model selectbox synchronized with the cross-page selected model."""
    ranked = rank_models(with_short_labels(results), metric="holdout_wr_at_100")
    options = ranked["result_id"].tolist()
    labels = dict(zip(options, ranked["short_label"], strict=False))
    current = st.session_state.get(SELECTED_MODEL_KEY)
    index = options.index(current) if current in options else 0
    selected_id = st.selectbox(label, options=options, index=index, format_func=lambda value: labels.get(value, value))
    st.session_state[SELECTED_MODEL_KEY] = selected_id
    return ranked[ranked["result_id"].eq(selected_id)].iloc[0]


def kpi_row(items: list[tuple[str, str, str | None]]) -> None:
    columns = st.columns(len(items))
    for column, (label, value, help_text) in zip(columns, items, strict=False):
        column.metric(label, value, help=help_text, border=True)


def fmt(value: object, *, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def fmt_pct(value: object) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.1%}"


def fmt_count_pct(count: object, pct: object) -> str:
    if count is None or pd.isna(count):
        return "-"
    text = str(int(count))
    if pct is not None and pd.notna(pct):
        text += f" | {float(pct):.0%}"
    return text
