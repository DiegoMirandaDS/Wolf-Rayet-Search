"""Shared widgets and formatting helpers for the Model Explorer."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from wr_detector.modeling.explorer import rank_models


METRIC_LABELS = {
    "holdout_wr_at_10": "WR recovered @10",
    "holdout_wr_at_50": "WR recovered @50",
    "holdout_wr_at_100": "WR recovered @100",
    "holdout_recall_at_50": "Recall @50",
    "holdout_recall_at_100": "Recall @100",
    "holdout_precision_at_100": "Precision @100",
    "holdout_average_precision": "Average precision",
    "holdout_precision_wr": "Threshold precision",
    "holdout_recall_wr": "Threshold recall",
    "holdout_fp": "False positives",
    "holdout_fpr": "False-positive rate",
    "ranking_score": "Ranking score",
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

MODEL_SELECTION_COLUMNS = [
    "rank",
    "profile",
    "short_label",
    "dataset_variant",
    "selection_status",
    "wr_holdout",
    "negative_holdout",
    "holdout_average_precision",
    "holdout_wr_at_100",
    "holdout_recall_at_100",
    "holdout_precision_at_100",
    "holdout_recall_at_50",
    "holdout_precision_wr",
    "holdout_recall_wr",
    "holdout_fp",
    "holdout_fpr",
    "ranking_score",
]


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


def active_model(results: pd.DataFrame) -> pd.Series:
    """Return the synchronized active model, repairing stale run state."""
    ranked = rank_models(with_short_labels(results), metric="ranking_score")
    options = ranked["result_id"].astype(str).tolist()
    current = st.session_state.get(SELECTED_MODEL_KEY)
    if current not in options:
        current = options[0]
        st.session_state[SELECTED_MODEL_KEY] = current
    return ranked[ranked["result_id"].astype(str).eq(str(current))].iloc[0]


def active_model_control(
    results: pd.DataFrame,
    *,
    widget_key: str,
) -> pd.Series:
    """Show the active model and a compact two-step picker."""
    selected = active_model(results)
    st.markdown(
        f"**{selected['short_label']}**  \n"
        f"{status_badge(selected.get('selection_status'))}"
    )
    with st.popover("Switch model", icon=":material/swap_horiz:"):
        _active_model_picker(results, selected=selected, widget_key=widget_key)
    return selected


def active_model_context(
    results: pd.DataFrame,
    *,
    widget_key: str,
    selector_ratio: float = 2.6,
) -> pd.Series:
    """Render a compact, ID-free model selector with operational metrics."""
    columns = st.columns([selector_ratio, 1, 1, 1])
    with columns[0]:
        selected = active_model_control(results, widget_key=widget_key)
    columns[1].metric(
        "Recall @100",
        fmt_pct(selected.get("holdout_recall_at_100", selected.get("holdout_wr_at_100_pct"))),
        help=f"WR recovered @100: {fmt(selected.get('holdout_wr_at_100'))}",
        border=True,
    )
    columns[2].metric(
        "AP",
        fmt(selected.get("holdout_average_precision")),
        border=True,
    )
    columns[3].metric(
        "Score",
        fmt(selected.get("ranking_score"), digits=3),
        help="Relative operating score with a soft false-positive-rate penalty.",
        border=True,
    )
    return selected


def _active_model_picker(
    results: pd.DataFrame,
    *,
    selected: pd.Series,
    widget_key: str,
) -> None:
    candidates = with_short_labels(results)
    variant = _filtered_selectbox(
        "Dataset",
        sorted(candidates["dataset_variant"].astype(str).unique().tolist()),
        default=str(selected["dataset_variant"]),
        key=f"{widget_key}_variant",
    )
    candidates = candidates[candidates["dataset_variant"].astype(str).eq(variant)]

    options = candidates["result_id"].astype(str).tolist()
    labels = {
        str(row.result_id): model_configuration_label(pd.Series(row._asdict()))
        for row in candidates.itertuples(index=False)
    }
    chosen = _filtered_selectbox(
        "Configuration",
        options,
        default=str(selected["result_id"]),
        key=f"{widget_key}_result",
        format_func=lambda value: labels.get(value, value),
    )
    if st.button(
        "Use as active model",
        key=f"{widget_key}_apply",
        type="primary",
        width="stretch",
    ):
        st.session_state[SELECTED_MODEL_KEY] = chosen
        st.rerun()


def model_configuration_label(row: pd.Series) -> str:
    """Compact model label for a dataset-scoped picker."""
    model = MODEL_ABBREVIATIONS.get(str(row.get("model")), str(row.get("model")))
    sampler = (
        str(row.get("sampler", ""))
        .replace("smote_enn", "SMOTE-ENN")
        .replace("smote", "SMOTE")
    )
    features = "+err" if bool(row.get("includes_parallax_error")) else "colors"
    ratio_value = str(row.get("negative_ratio_label", ""))
    ratio_display = f" · {ratio_value}" if ratio_value and ratio_value != "nan" else ""
    return (
        f"{model}/{sampler} · {features}{ratio_display} · "
        f"R@100 {fmt_pct(row.get('holdout_recall_at_100', row.get('holdout_wr_at_100_pct')))} "
        f"({fmt(row.get('holdout_wr_at_100'))} WR) · "
        f"AP {fmt(row.get('holdout_average_precision'))}"
    )


def chart_label(row: pd.Series) -> str:
    """Short but distinguishable label for compact comparison charts."""
    model = MODEL_ABBREVIATIONS.get(str(row.get("model")), str(row.get("model")))
    sampler = (
        str(row.get("sampler", ""))
        .replace("smote_enn", "SMOTE-ENN")
        .replace("smote", "SMOTE")
    )
    variant = str(row.get("dataset_variant", ""))
    family = "R" if variant.startswith("relaxed_") else "S"
    subset = variant.removeprefix("relaxed_").removeprefix("strict_")
    subset = {"photometry": "photo", "parallax_soft": "px-soft"}.get(
        subset,
        subset,
    )
    features = " +err" if bool(row.get("includes_parallax_error")) else ""
    return f"{model}/{sampler} · {family}/{subset}{features}"


def _filtered_selectbox(
    label: str,
    options: list[str],
    *,
    default: str,
    key: str,
    format_func=None,
) -> str:
    if not options:
        raise ValueError(f"No options available for {label}.")
    current = st.session_state.get(key)
    if current not in options:
        st.session_state[key] = default if default in options else options[0]
    kwargs = {"key": key}
    if format_func is not None:
        kwargs["format_func"] = format_func
    return st.selectbox(label, options=options, **kwargs)


def kpi_row(items: list[tuple[str, str, str | None]]) -> None:
    columns = st.columns(len(items))
    for column, (label, value, help_text) in zip(columns, items, strict=False):
        column.metric(label, value, help=help_text, border=True)


def model_selection_view(results: pd.DataFrame) -> pd.DataFrame:
    """Return the shared operational comparison columns in display order."""
    columns = [column for column in MODEL_SELECTION_COLUMNS if column in results.columns]
    return results[columns].copy()


def model_selection_column_config() -> dict[str, object]:
    """Shared formatting for Overview and Compare model-selection tables."""
    percentage_help = {
        "holdout_recall_at_100": "WR recovered in the first 100 candidates divided by holdout WR.",
        "holdout_precision_at_100": "WR recovered in the first 100 candidates divided by candidates reviewed.",
        "holdout_recall_at_50": "WR recovered in the first 50 candidates divided by holdout WR.",
        "holdout_precision_wr": "Holdout precision at the calibrated operating threshold.",
        "holdout_recall_wr": "Holdout recall at the calibrated operating threshold.",
    }
    config: dict[str, object] = {
        "rank": st.column_config.NumberColumn("Rank", width="small", format="%d"),
        "profile": st.column_config.TextColumn("Profile"),
        "short_label": st.column_config.TextColumn("Model"),
        "dataset_variant": st.column_config.TextColumn("Dataset variant"),
        "selection_status": st.column_config.TextColumn("Status"),
        "wr_holdout": st.column_config.NumberColumn(
            "Holdout WR",
            format="%d",
            help="Positive denominator for Recall @50 and Recall @100.",
        ),
        "negative_holdout": st.column_config.NumberColumn(
            "Holdout negatives",
            format="%d",
            help="Negative denominator for FPR.",
        ),
        "holdout_average_precision": st.column_config.NumberColumn("AP", format="%.4f"),
        "holdout_wr_at_100": st.column_config.NumberColumn(
            "WR@100",
            format="%d",
            help="Descriptive count only; not comparable when holdout WR totals differ.",
        ),
        "holdout_fp": st.column_config.NumberColumn("FP", format="%d"),
        "holdout_fpr": st.column_config.NumberColumn(
            "FPR",
            format="%.4f",
            help="Holdout false positives divided by holdout negatives.",
        ),
        "ranking_score": st.column_config.NumberColumn(
            "Ranking score",
            format="%.4f",
            help="Weighted relative score with a soft penalty for high holdout FPR.",
        ),
    }
    for column, help_text in percentage_help.items():
        config[column] = st.column_config.ProgressColumn(
            metric_label(column),
            min_value=0.0,
            max_value=1.0,
            format="percent",
            help=help_text,
        )
    return config


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
