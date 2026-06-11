from __future__ import annotations

import argparse
import html
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from wr_detector.modeling.explorer import (
    RANKING_METRICS,
    best_models_by_dataset,
    explorer_db_path,
    filter_results,
    list_explorer_runs,
    load_feature_importance,
    load_prediction_summary,
    load_run_results,
    rank_models,
    resolve_artifact_path,
)


DISPLAY_COLUMNS = [
    "rank",
    "result_id",
    "dataset_variant",
    "feature_set",
    "model",
    "sampler",
    "negative_ratio_label",
    "model_label",
    "selection_status",
    "holdout_wr_at_10",
    "holdout_wr_at_10_pct",
    "holdout_wr_at_50",
    "holdout_wr_at_50_pct",
    "holdout_wr_at_100",
    "holdout_wr_at_100_pct",
    "holdout_average_precision",
    "holdout_recall_at_fpr_0p005",
    "cv_train_gap_f2",
    "train_cv_gap_average_precision",
    "overfit_risk_score",
    "holdout_cv_gap_f2",
    "hyperparams_compact",
]

METRIC_LABELS = {
    "holdout_wr_at_10": "WR recovered @10",
    "holdout_wr_at_50": "WR recovered @50",
    "holdout_wr_at_100": "WR recovered @100",
    "holdout_average_precision": "Average precision",
    "holdout_recall_at_fpr_0p005": "Recall at FPR 0.005",
    "cv_train_gap_f2": "Train-CV F2 gap",
    "train_cv_gap_average_precision": "Train-CV AP gap",
    "overfit_risk_score": "Overfit risk score",
    "holdout_cv_gap_f2": "Holdout-CV F2 gap",
}


def main() -> None:
    args = _parse_args()
    st.set_page_config(page_title="WR Model Explorer", layout="wide")
    _inject_style()

    config_path = Path(args.config)
    _app_header(explorer_db_path(config_path))

    runs = _cached_runs(str(config_path))
    if runs.empty:
        st.error("No training runs found in the configured DuckDB history.")
        return

    _sidebar_brand()
    run_id = _run_selector(runs)
    results = _cached_results(str(config_path), run_id)
    if results.empty:
        st.warning("The selected run has no model results.")
        return

    filtered, metric, top_n = _sidebar_filters(results)
    if filtered.empty:
        _summary(results, pd.DataFrame())
        st.warning("No models match the current filters.")
        return

    ranked = rank_models(filtered, metric=metric, top_n=top_n)
    best_by_dataset = best_models_by_dataset(filtered, metric=metric)

    _section_marker("Model comparison", f"{len(filtered):,} filtered models")
    _summary(results, filtered)
    _comparison_section(ranked, metric)
    left, right = st.columns(2, gap="medium")
    with left:
        _best_by_dataset_section(best_by_dataset, metric)
    with right:
        clicked_result_id = _ranked_models_section(ranked, metric)
    if clicked_result_id:
        st.session_state["selected_model_result_id"] = clicked_result_id

    _recovery_budget_section(ranked)
    _additional_run_diagnostics(filtered, ranked, metric)

    _section_marker("Selected model", "single-model inspection")
    selected = _selected_model_picker(ranked)
    _selected_model_panel(config_path, run_id, selected)


@st.cache_data(show_spinner=False)
def _cached_runs(config_path: str) -> pd.DataFrame:
    return list_explorer_runs(config_path)


@st.cache_data(show_spinner=False)
def _cached_results(config_path: str, run_id: str) -> pd.DataFrame:
    return load_run_results(config_path, run_id=run_id)


@st.cache_data(show_spinner=False)
def _cached_feature_importance(config_path: str, run_id: str, result_id: str) -> pd.DataFrame:
    return load_feature_importance(config_path, run_id=run_id, result_id=result_id)


@st.cache_data(show_spinner=False)
def _cached_prediction_summary(config_path: str, run_id: str, result_id: str) -> pd.DataFrame:
    return load_prediction_summary(config_path, run_id=run_id, result_id=result_id)


def _run_selector(runs: pd.DataFrame) -> str:
    labels = {
        row.run_id: f"{row.run_id} ({int(row.row_count)} models)"
        for row in runs.itertuples(index=False)
    }
    return st.sidebar.selectbox(
        "Training run",
        options=list(labels.keys()),
        format_func=lambda value: labels.get(value, value),
    )


def _sidebar_filters(results: pd.DataFrame) -> tuple[pd.DataFrame, str, int]:
    st.sidebar.markdown('<div class="sidebar-title">Filters</div>', unsafe_allow_html=True)
    include_warnings = st.sidebar.toggle("Include warnings and failed floors", value=True)
    metric = st.sidebar.selectbox("Primary metric", options=RANKING_METRICS, format_func=lambda value: METRIC_LABELS.get(value, value), index=2)
    top_n = st.sidebar.number_input("Top N", min_value=1, max_value=max(1, len(results)), value=min(16, len(results)), step=1)

    selections = {
        "dataset_families": _multiselect("Family", results, "dataset_family"),
        "models": _multiselect("Model", results, "model"),
    }
    if st.sidebar.checkbox("Dataset filters", value=False, key="show_dataset_filters_v2"):
        selections.update(
            {
                "dataset_variants": _multiselect("Dataset", results, "dataset_variant"),
                "astrometric_subsets": _multiselect("Astrometric subset", results, "astrometric_subset"),
                "feature_sets": _multiselect("Feature set", results, "feature_set"),
            }
        )
    if st.sidebar.checkbox("Training filters", value=False, key="show_training_filters_v2"):
        selections.update(
            {
                "samplers": _multiselect("Sampler", results, "sampler"),
                "negative_ratio_labels": _multiselect("Negative ratio", results, "negative_ratio_label"),
                "selection_statuses": _multiselect("Selection status", results, "selection_status"),
            }
        )
    filtered = filter_results(results, include_warnings=include_warnings, **selections)
    return filtered, metric, int(top_n)


def _sidebar_brand() -> None:
    st.sidebar.markdown(
        '<div class="sidebar-brand"><div class="brand-mark">WR</div><div><strong>Model Explorer</strong><p>Training runs</p></div></div>',
        unsafe_allow_html=True,
    )


def _summary(results: pd.DataFrame, filtered: pd.DataFrame) -> None:
    total = len(results)
    accepted = _status_count(results, "accepted")
    warnings = _warning_count(results)
    floors = int(results["selection_status"].astype(str).str.contains("floor_not_met", na=False).sum()) if "selection_status" in results else 0
    visible = len(filtered)

    st.markdown(
        f"""
        <div class="metric-grid">
          {_metric_card("Configurations", f"{total:,}", "Total evaluated models", "cube", "cyan")}
          {_metric_card("Accepted", f"{accepted:,}", f"{accepted / total:.1%} of total" if total else "0.0% of total", "check", "teal")}
          {_metric_card("Warnings", f"{warnings:,}", f"{warnings / total:.1%} of total" if total else "0.0% of total", "warn", "pink")}
          {_metric_card("Visible", f"{visible:,}", f"{floors:,} floor flags", "eye", "yellow")}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _section_marker(title: str, detail: str) -> None:
    st.markdown(
        f"""
        <div class="section-marker">
          <strong>{html.escape(title)}</strong>
          <span>{html.escape(detail)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _best_by_dataset_section(best: pd.DataFrame, metric: str) -> None:
    table = _display_table(best)
    if table.empty:
        st.info("No dataset winners for the current filters.")
        return
    st.markdown(
        f"""
        <div class="panel table-panel compact-table-panel">
          <div class="panel-title">Best model by dataset</div>
          {_model_table_html(table.head(8), metric_col=metric, compact=True)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _ranked_table(ranked: pd.DataFrame, metric: str) -> pd.DataFrame:
    columns = [
        "result_id",
        "dataset_variant",
        "feature_set",
        "model",
        "sampler",
        "selection_status",
        metric,
    ]
    pct_col = _pct_column_for_metric(metric)
    if pct_col in ranked.columns:
        columns.append(pct_col)
    for optional in [
        "holdout_average_precision",
        "cv_train_gap_f2",
        "train_cv_gap_average_precision",
        "overfit_risk_score",
        "holdout_cv_gap_f2",
        "hyperparams_compact",
    ]:
        if optional not in columns:
            columns.append(optional)

    view = ranked[[column for column in columns if column in ranked.columns]].copy()
    view.insert(0, "rank", range(1, len(view) + 1))
    rename = {
        "dataset_variant": "dataset",
        "feature_set": "features",
        "selection_status": "status",
        "holdout_average_precision": "AP",
        "cv_train_gap_f2": "train-CV gap",
        "train_cv_gap_average_precision": "AP gap",
        "overfit_risk_score": "risk",
        "holdout_cv_gap_f2": "holdout-CV gap",
        "hyperparams_compact": "params",
    }
    rename[metric] = _table_header(metric)
    if pct_col in view.columns:
        rename[pct_col] = "recovery"
    view = view.rename(columns=rename)
    return view


def _comparison_section(ranked: pd.DataFrame, metric: str) -> None:
    chart_data = ranked.dropna(subset=[metric])
    if chart_data.empty:
        st.info("No values available for this metric.")
        return
    st.markdown(f'<div class="panel-title primary-title">Metric comparison <span>{html.escape(METRIC_LABELS.get(metric, metric))}</span></div>', unsafe_allow_html=True)
    chart_rows = chart_data.head(18)
    chart_height = min(7.0, max(5.2, 1.0 + 0.36 * len(chart_rows)))
    _metric_comparison_chart(chart_rows, metric=metric, height=chart_height)


def _recovery_budget_section(ranked: pd.DataFrame) -> None:
    ks = [k for k in [10, 50, 100, 500, 1000] if f"holdout_wr_at_{k}_pct" in ranked.columns]
    if not ks:
        return
    st.markdown('<div class="panel-title outside">Recovery by review budget</div>', unsafe_allow_html=True)
    _topk_recovery_chart(ranked.head(6), ks=ks, height=4.8)


def _additional_run_diagnostics(filtered: pd.DataFrame, ranked: pd.DataFrame, metric: str) -> None:
    with st.expander("Additional run diagnostics", expanded=False):
        tab_budget, tab_dataset, tab_gaps, tab_split, tab_complexity = st.tabs(
            ["Dataset matrix", "Stability gaps", "Train/CV/Holdout", "Complexity", "Prediction audit"]
        )
        with tab_budget:
            _dataset_matrix_chart(filtered, metric)
        with tab_dataset:
            _stability_gaps_chart(ranked.head(12))
        with tab_gaps:
            _split_metrics_chart(ranked.head(8))
        with tab_split:
            _model_complexity_table(filtered)
        with tab_complexity:
            st.info("SIMBAD false-positive types and WR subtype recall require joining holdout predictions with reference/source tables. The Explorer keeps prediction loading aggregated by default to avoid pulling the full prediction table into memory.")


def _ranked_models_section(ranked: pd.DataFrame, metric: str) -> str | None:
    if ranked.empty:
        st.info("No ranked models available.")
        return None

    view = _ranked_table(ranked, metric)
    current = st.session_state.get("selected_model_result_id")
    view.insert(0, "inspect", view["result_id"].eq(current) if current in set(view["result_id"]) else False)
    display_view = _format_dataframe(view.drop(columns=["result_id"], errors="ignore"))
    st.markdown('<div class="panel-title outside">Ranked models</div>', unsafe_allow_html=True)
    edited = st.data_editor(
        display_view,
        width="stretch",
        hide_index=True,
        disabled=[column for column in display_view.columns if column != "inspect"],
        column_config={"inspect": st.column_config.CheckboxColumn("inspect")},
        key="ranked_models_editor_v1",
    )
    checked = edited.index[edited["inspect"].fillna(False)].tolist() if "inspect" in edited.columns else []
    if checked:
        preferred = [idx for idx in checked if str(view.iloc[idx]["result_id"]) != str(current)]
        row_idx = preferred[0] if preferred else checked[0]
        return str(view.iloc[row_idx]["result_id"])
    return None


def _selected_model_picker(ranked: pd.DataFrame) -> pd.Series:
    options = ranked["result_id"].tolist()
    labels = dict(zip(ranked["result_id"], ranked["model_label"], strict=False))
    current = st.session_state.get("selected_model_result_id")
    index = options.index(current) if current in options else 0
    selected_id = st.selectbox(
        "Selected model",
        options=options,
        index=index,
        format_func=lambda value: labels.get(value, value),
        key="selected_model_dropdown_v2",
    )
    st.session_state["selected_model_result_id"] = selected_id
    return ranked[ranked["result_id"].eq(selected_id)].iloc[0]


def _selected_model_panel(config_path: Path, run_id: str, selected: pd.Series) -> None:
    metrics = [
        "holdout_wr_at_10",
        "holdout_wr_at_50",
        "holdout_wr_at_100",
        "holdout_average_precision",
        "holdout_recall_at_fpr_0p005",
        "train_cv_gap_average_precision",
        "overfit_risk_score",
        "holdout_cv_gap_f2",
    ]
    metric_values = {metric: selected.get(metric) for metric in metrics if metric in selected.index}
    model_label_value = html.escape(str(selected.get("model_label", "")))
    status = html.escape(str(selected.get("selection_status", "")))
    st.markdown(
        f"""
        <div class="panel selected-panel">
          <div class="selected-head">
            <div>
              <div class="panel-title">Selected model</div>
              <div class="selected-label">{model_label_value}</div>
            </div>
            <div class="status-chip">{status}</div>
          </div>
          <div class="selected-metrics">
            {_selected_metric("WR@10", metric_values.get("holdout_wr_at_10"), selected.get("holdout_wr_at_10_pct"))}
            {_selected_metric("WR@50", metric_values.get("holdout_wr_at_50"), selected.get("holdout_wr_at_50_pct"))}
            {_selected_metric("WR@100", metric_values.get("holdout_wr_at_100"), selected.get("holdout_wr_at_100_pct"), active=True)}
            {_selected_metric("AP", metric_values.get("holdout_average_precision"))}
            {_selected_metric("Recall @ FPR 0.005", metric_values.get("holdout_recall_at_fpr_0p005"))}
            {_selected_metric("AP gap", metric_values.get("train_cv_gap_average_precision"))}
            {_selected_metric("Risk", metric_values.get("overfit_risk_score"))}
            {_selected_metric("F2 gap", metric_values.get("holdout_cv_gap_f2"))}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _feature_importance_section(config_path, run_id, str(selected["result_id"]))
    _artifact_buttons(selected)

    with st.expander("Diagnostics and previews", expanded=False):
        st.markdown('<div class="subpanel-title">Hyperparameters</div>', unsafe_allow_html=True)
        st.code(str(selected.get("bayes_best_params", "")))
        _prediction_summary_section(config_path, run_id, str(selected["result_id"]))
        _artifact_images(selected)


def _artifact_images(selected: pd.Series) -> None:
    image_columns = [
        ("Confusion matrix", "holdout_confusion_matrix_path"),
        ("ROC curve", "holdout_roc_curve_path"),
        ("Precision-recall curve", "holdout_pr_curve_path"),
        ("Feature importance figure", "feature_importance_figure_path"),
    ]
    rendered = 0
    for idx, (label, column) in enumerate(image_columns):
        path = resolve_artifact_path(selected.get(column))
        if path is None or not path.exists():
            continue
        st.image(str(path), caption=label, width="stretch")
        rendered += 1
    if rendered == 0:
        st.info("No image artifacts are available for the selected model.")


def _artifact_buttons(selected: pd.Series) -> None:
    rows = [
        ("PR curve", "holdout_pr_curve_path", "cyan"),
        ("ROC curve", "holdout_roc_curve_path", "pink"),
        ("Confusion matrix", "holdout_confusion_matrix_path", "yellow"),
    ]
    rows_html = []
    for label, column, tone in rows:
        path = selected.get(column)
        resolved = resolve_artifact_path(path)
        exists = bool(resolved and resolved.exists())
        state = "available" if exists else "missing"
        short_path = html.escape(str(path or "not recorded"))
        rows_html.append(
            f'<div class="artifact-row {tone}">'
            f'<span>{html.escape(label)}</span>'
            f'<em>{state}</em>'
            f'<code>{short_path}</code>'
            f'</div>'
        )
    st.markdown(
        f"""
        <div class="artifact-panel">
          <div class="subpanel-title">Artifact files</div>
          {''.join(rows_html)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _feature_importance_section(config_path: Path, run_id: str, result_id: str) -> None:
    importance = _cached_feature_importance(str(config_path), run_id, result_id)
    if importance.empty:
        st.info("No feature importance rows are synchronized for this model.")
        return
    top = importance.sort_values("importance_mean", ascending=False).head(20)
    with st.container(border=True):
        st.markdown('<div class="subpanel-title">Feature importance</div>', unsafe_allow_html=True)
        _horizontal_bar_chart(top[["feature", "importance_mean"]].head(16), label_col="feature", value_col="importance_mean", height=5.4)


def _prediction_summary_section(config_path: Path, run_id: str, result_id: str) -> None:
    st.subheader("Prediction score summary")
    summary = _cached_prediction_summary(str(config_path), run_id, result_id)
    if summary.empty:
        st.info("No synchronized prediction rows are available for this model.")
        return
    st.dataframe(summary, width="stretch", hide_index=True)


def _app_header(db_path: Path) -> None:
    st.markdown(
        f"""
        <div class="app-header">
          <div>
            <h1>WR Model Explorer</h1>
            <p class="app-source">{db_path}</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _metric_comparison_chart(df: pd.DataFrame, *, metric: str, height: float) -> None:
    pct_col = _pct_column_for_metric(metric)
    plot_data = df.copy()
    use_pct_axis = pct_col in plot_data.columns and metric != "holdout_wr_at_10"
    value_col = pct_col if use_pct_axis else metric
    plot_data = plot_data.dropna(subset=[value_col]).sort_values(value_col, ascending=True)
    if plot_data.empty:
        st.info("No chartable values available.")
        return

    labels = plot_data["model_label"].astype(str).map(lambda value: _chart_label(value, compact=False))
    values = plot_data[value_col].astype(float)
    if use_pct_axis:
        values = values * 100

    fig, ax = plt.subplots(figsize=(12.8, height), facecolor="#14001f")
    ax.set_facecolor("#180629")
    colors = ["#25f4d0", "#ff4fab", "#ffe66d", "#7c5cff", "#48a7ff"]
    bars = ax.barh(range(len(values)), values, color=[colors[idx % len(colors)] for idx in range(len(values))], alpha=0.94, height=0.62)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, color="#f6edff", fontsize=8)
    ax.tick_params(axis="x", colors="#c9a8de", labelsize=8)
    ax.grid(axis="x", color="#3c1850", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#361348")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if pct_col in plot_data.columns:
        label_values = []
        for row in plot_data.itertuples(index=False):
            count = getattr(row, metric, None)
            pct = getattr(row, pct_col, None)
            total = getattr(row, "wr_holdout", None)
            if pd.notna(count) and pd.notna(pct):
                total_text = f"/{int(total)}" if total is not None and pd.notna(total) else ""
                label_values.append(f"{int(count)}{total_text} · {100 * float(pct):.1f}%")
            else:
                label_values.append(_format_metric_value(count))
    else:
        label_values = [_format_metric_value(value) for value in plot_data[metric]]

    offset = max(values.max() * 0.012, 0.02) if len(values) else 0.02
    for bar, text in zip(bars, label_values, strict=False):
        ax.text(bar.get_width() + offset, bar.get_y() + bar.get_height() / 2, text, va="center", color="#f6edff", fontsize=8)

    axis_label = f"{METRIC_LABELS.get(metric, metric)} (%)" if use_pct_axis else METRIC_LABELS.get(metric, metric)
    ax.set_xlabel(axis_label, color="#c9a8de", fontsize=9)
    ax.margins(x=0.12)
    fig.tight_layout(pad=1.2)
    st.pyplot(fig, clear_figure=True, use_container_width=True)


def _topk_recovery_chart(df: pd.DataFrame, *, ks: list[int], height: float) -> None:
    top = df.head(6).copy()
    if top.empty:
        return
    fig, ax = plt.subplots(figsize=(12.4, height), facecolor="#14001f")
    ax.set_facecolor("#180629")
    colors = ["#25f4d0", "#ff4fab", "#ffe66d", "#7c5cff", "#48a7ff", "#c084fc"]
    for idx, row in top.reset_index(drop=True).iterrows():
        y = [100 * row.get(f"holdout_wr_at_{k}_pct", pd.NA) for k in ks]
        ax.plot(ks, y, marker="o", lw=2, color=colors[idx % len(colors)], label=_chart_label(str(row["model_label"]), compact=True))
        for k, pct_value in zip(ks, y, strict=False):
            count = row.get(f"holdout_wr_at_{k}", pd.NA)
            if pd.notna(count) and pd.notna(pct_value):
                ax.text(k, pct_value + 1.0, str(int(count)), color="#f6edff", fontsize=7, ha="center")
    ax.set_xscale("log")
    ax.set_xticks(ks, labels=[str(k) for k in ks])
    ax.set_ylabel("WR holdout recovered (%)", color="#c9a8de")
    ax.set_xlabel("Candidates reviewed", color="#c9a8de")
    ax.tick_params(colors="#c9a8de", labelsize=8)
    ax.grid(color="#3c1850", linewidth=0.8, alpha=0.7)
    ax.legend(frameon=False, fontsize=7, loc="lower right", labelcolor="#f6edff")
    for spine in ax.spines.values():
        spine.set_color("#361348")
    fig.tight_layout(pad=1.2)
    st.pyplot(fig, clear_figure=True, use_container_width=True)


def _dataset_matrix_chart(df: pd.DataFrame, metric: str) -> None:
    pct_col = _pct_column_for_metric(metric)
    value_col = pct_col if pct_col in df.columns else metric
    if value_col not in df.columns or df.empty:
        st.info("No dataset matrix values available.")
        return
    matrix = df.pivot_table(index="dataset_variant", columns=["model", "sampler"], values=value_col, aggfunc="max")
    if matrix.empty:
        st.info("No dataset matrix values available.")
        return
    values = matrix.astype(float)
    if value_col == pct_col:
        values = values * 100
    fig, ax = plt.subplots(figsize=(12.2, max(3.2, 0.42 * len(values) + 1.4)), facecolor="#14001f")
    ax.set_facecolor("#180629")
    image = ax.imshow(values.fillna(0), aspect="auto", cmap="magma")
    ax.set_xticks(range(len(values.columns)), [" / ".join(map(str, col)) for col in values.columns], rotation=35, ha="right", color="#f6edff", fontsize=8)
    ax.set_yticks(range(len(values.index)), values.index.astype(str), color="#f6edff", fontsize=8)
    for y_idx, (_, row) in enumerate(values.iterrows()):
        for x_idx, value in enumerate(row):
            if pd.notna(value):
                ax.text(x_idx, y_idx, f"{value:.1f}" if value_col == pct_col else f"{value:.3g}", ha="center", va="center", color="#f8efff", fontsize=7)
    cbar = fig.colorbar(image, ax=ax, fraction=0.024, pad=0.02)
    cbar.ax.tick_params(colors="#c9a8de", labelsize=8)
    cbar.set_label("% recovered" if value_col == pct_col else METRIC_LABELS.get(metric, metric), color="#c9a8de")
    for spine in ax.spines.values():
        spine.set_color("#361348")
    fig.tight_layout(pad=1.2)
    st.pyplot(fig, clear_figure=True, use_container_width=True)


def _stability_gaps_chart(df: pd.DataFrame) -> None:
    required = {"model_label", "cv_train_gap_f2", "holdout_cv_gap_f2"}
    if not required.issubset(df.columns):
        st.info("No stability gap columns available.")
        return
    plot = df.dropna(subset=["cv_train_gap_f2", "holdout_cv_gap_f2"]).head(12).copy()
    if plot.empty:
        st.info("No stability gap values available.")
        return
    labels = plot["model_label"].astype(str).map(lambda value: _chart_label(value, compact=True))
    x = range(len(plot))
    fig, ax = plt.subplots(figsize=(12.2, 4.8), facecolor="#14001f")
    ax.set_facecolor("#180629")
    width = 0.36
    ax.bar([idx - width / 2 for idx in x], plot["cv_train_gap_f2"], width=width, color="#ffe66d", label="train - CV")
    ax.bar([idx + width / 2 for idx in x], plot["holdout_cv_gap_f2"], width=width, color="#25f4d0", label="holdout - CV")
    ax.axhline(0, color="#c9a8de", linewidth=0.8)
    ax.axhline(0.15, color="#ff4fab", linewidth=0.9, linestyle="--", alpha=0.75)
    ax.set_xticks(list(x), labels, rotation=30, ha="right", color="#f6edff", fontsize=8)
    ax.set_ylabel("F2 gap", color="#c9a8de")
    ax.tick_params(axis="y", colors="#c9a8de", labelsize=8)
    ax.legend(frameon=False, fontsize=8, labelcolor="#f6edff")
    ax.grid(axis="y", color="#3c1850", linewidth=0.8, alpha=0.7)
    for spine in ax.spines.values():
        spine.set_color("#361348")
    fig.tight_layout(pad=1.2)
    st.pyplot(fig, clear_figure=True, use_container_width=True)


def _split_metrics_chart(df: pd.DataFrame) -> None:
    metric_specs = [
        ("average_precision", "Average precision"),
        ("f2_wr", "F2"),
        ("recall_wr", "Recall"),
        ("precision_wr", "Precision"),
    ]
    top = df.head(8).copy()
    available = [spec for spec in metric_specs if all(f"{split}_{spec[0]}" in top.columns for split in ["train", "cv", "holdout"])]
    if not available:
        st.info("No train/CV/holdout metric columns available.")
        return
    labels = top["model_label"].astype(str).map(lambda value: _chart_label(value, compact=True))
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.0), facecolor="#14001f")
    colors = {"train": "#25f4d0", "cv": "#ffe66d", "holdout": "#ff4fab"}
    for ax, (metric, title) in zip(axes.ravel(), metric_specs, strict=False):
        ax.set_facecolor("#180629")
        if (metric, title) not in available:
            ax.axis("off")
            continue
        x = list(range(len(top)))
        for split in ["train", "cv", "holdout"]:
            ax.plot(x, top[f"{split}_{metric}"], marker="o", linewidth=1.8, label=split, color=colors[split])
        ax.set_title(title, color="#f8efff", fontsize=10)
        ax.set_xticks(x, labels, rotation=32, ha="right", color="#f6edff", fontsize=7)
        ax.tick_params(axis="y", colors="#c9a8de", labelsize=8)
        ax.grid(color="#3c1850", linewidth=0.8, alpha=0.7)
        for spine in ax.spines.values():
            spine.set_color("#361348")
    axes[0, 0].legend(frameon=False, fontsize=8, labelcolor="#f6edff")
    fig.tight_layout(pad=1.3)
    st.pyplot(fig, clear_figure=True, use_container_width=True)


def _model_complexity_table(df: pd.DataFrame) -> None:
    if "model" not in df.columns or df.empty:
        st.info("No model complexity data available.")
        return
    agg_spec = {"configs": ("model_label", "size")}
    for column in ["n_estimators", "max_depth", "cv_train_gap_f2", "holdout_cv_gap_f2", "holdout_average_precision", "overfit_warning_flag"]:
        if column in df.columns:
            if column == "overfit_warning_flag":
                agg_spec["overfit_warning_rate"] = (column, "mean")
            else:
                agg_spec[f"{column}_mean"] = (column, "mean")
                if column in {"n_estimators", "max_depth"}:
                    agg_spec[f"{column}_median"] = (column, "median")
    summary = df.groupby("model", dropna=False).agg(**agg_spec).reset_index()
    if summary.empty:
        st.info("No model complexity data available.")
        return
    st.dataframe(_format_dataframe(summary), width="stretch", hide_index=True)


def _horizontal_bar_chart(df: pd.DataFrame, *, label_col: str, value_col: str, height: float, compact: bool = False) -> None:
    plot_data = df[[label_col, value_col]].dropna().tail(30).copy()
    if plot_data.empty:
        st.info("No chartable values available.")
        return
    plot_data = plot_data.sort_values(value_col, ascending=True)
    labels = plot_data[label_col].astype(str).map(lambda value: _chart_label(value, compact=compact))
    values = plot_data[value_col].astype(float)

    fig_width = 7.8 if compact else 12
    fig, ax = plt.subplots(figsize=(fig_width, height), facecolor="#14001f")
    ax.set_facecolor("#180629")
    colors = ["#25f4d0", "#ff4fab", "#ffe66d", "#7c5cff", "#48a7ff"]
    ax.barh(range(len(values)), values, color=[colors[idx % len(colors)] for idx in range(len(values))], alpha=0.92, height=0.62)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, color="#f6edff", fontsize=7 if compact else 8)
    ax.tick_params(axis="x", colors="#c9a8de", labelsize=7 if compact else 8)
    ax.grid(axis="x", color="#3c1850", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#361348")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel(value_col, color="#c9a8de", fontsize=8 if compact else 9)
    fig.tight_layout(pad=1.2)
    st.pyplot(fig, clear_figure=True, use_container_width=True)


def _metric_card(label: str, value: str, detail: str, icon: str, tone: str) -> str:
    icons = {
        "cube": "M",
        "check": "OK",
        "warn": "!",
        "eye": "V",
    }
    icon_text = icons.get(icon, "*")
    return (
        f'<div class="metric-card {tone}">'
        f'<div class="metric-icon">{html.escape(icon_text)}</div>'
        f'<div><p>{html.escape(label)}</p><strong>{html.escape(value)}</strong><span>{html.escape(detail)}</span></div>'
        "</div>"
    )


def _chart_label(value: str, *, compact: bool) -> str:
    if " / " in value:
        parts = value.split(" / ")
        if len(parts) >= 3:
            value = f"{parts[0]} / {parts[2]}"
    limit = 30 if compact else 48
    return value if len(value) <= limit else value[: limit - 3] + "..."


def _selected_metric(label: str, value: object, pct: object | None = None, *, active: bool = False) -> str:
    formatted = _format_metric_value(value)
    pct_text = ""
    if pct is not None and pd.notna(pct):
        pct_text = f"<span>{float(pct):.1%}</span>"
    active_class = " active" if active else ""
    return f'<div class="selected-metric{active_class}"><p>{html.escape(label)}</p><strong>{formatted}</strong>{pct_text}</div>'


def _pct_column_for_metric(metric: str) -> str:
    return f"{metric}_pct" if metric.startswith("holdout_wr_at_") else ""


def _format_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    formatted = df.copy()
    for column in formatted.columns:
        if column == "rank":
            continue
        if column == "recovery" or str(column).endswith("_pct"):
            formatted[column] = formatted[column].map(lambda value: f"{float(value):.1%}" if pd.notna(value) else "")
        elif pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(lambda value: f"{float(value):.4f}" if pd.notna(value) else "")
    return formatted


def _model_table_html(df: pd.DataFrame, *, metric_col: str, compact: bool = False) -> str:
    preferred = (
        ["dataset_variant", "model", "selection_status", metric_col]
        if compact
        else [
            "model_label",
            "selection_status",
            "holdout_wr_at_10",
            "holdout_wr_at_50",
            "holdout_wr_at_100",
            "holdout_average_precision",
        ]
    )
    columns = [column for column in preferred if column in df.columns]
    header = "".join(f"<th>{_table_header(column)}</th>" for column in columns)
    body_rows = []
    for idx, row in df[columns].iterrows():
        cells = []
        for column in columns:
            value = row[column]
            text = "" if pd.isna(value) else str(value)
            if column == "model_label":
                text = text if len(text) <= 86 else text[:83] + "..."
                cells.append(f'<td class="model-cell"><i class="{_dot_class(idx)}"></i>{html.escape(text)}</td>')
            elif column == "selection_status":
                status_text = _short_status(text) if compact else text
                cells.append(f'<td><span class="status-badge {_status_class(text)}">{html.escape(status_text)}</span></td>')
            elif column in {"dataset_variant", "model"}:
                compact_limit = 24 if column == "dataset_variant" else 18
                text = text if len(text) <= compact_limit else text[: compact_limit - 3] + "..."
                cells.append(f'<td class="text-cell">{html.escape(text)}</td>')
            else:
                metric_class = " metric-focus" if column == metric_col else ""
                cells.append(f'<td class="numeric{metric_class}">{html.escape(text)}</td>')
        body_rows.append(f"<tr>{''.join(cells)}</tr>")
    return f'<div class="model-table-wrap"><table class="model-table"><thead><tr>{header}</tr></thead><tbody>{"".join(body_rows)}</tbody></table></div>'


def _table_header(column: str) -> str:
    labels = {
        "dataset_variant": "dataset",
        "model": "model",
        "model_label": "model_label",
        "selection_status": "status",
        "holdout_wr_at_10": "WR@10",
        "holdout_wr_at_50": "WR@50",
        "holdout_wr_at_100": "WR@100",
        "holdout_average_precision": "AP",
        "holdout_recall_at_fpr_0p005": "recall",
        "cv_train_gap_f2": "train gap",
        "train_cv_gap_average_precision": "AP gap",
        "overfit_risk_score": "risk",
        "holdout_cv_gap_f2": "holdout gap",
    }
    return labels.get(column, column)


def _short_status(status: str) -> str:
    if status == "accepted":
        return "accepted"
    if status == "overfit_warning":
        return "overfit"
    if "floor_not_met" in status:
        return "floor"
    return status if len(status) <= 18 else status[:15] + "..."


def _status_class(status: str) -> str:
    if status == "accepted":
        return "accepted"
    if "overfit" in status:
        return "warning"
    if "floor" in status:
        return "floor"
    return "neutral"


def _dot_class(idx: object) -> str:
    try:
        return ["cyan", "pink", "yellow", "violet"][int(idx) % 4]
    except (TypeError, ValueError):
        return "cyan"


def _format_metric_value(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return html.escape(str(int(value) if isinstance(value, int) else value))


def _display_table(df: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in DISPLAY_COLUMNS if column in df.columns]
    table = df[columns].copy()
    for column in table.columns:
        if column.endswith("_pct"):
            table[column] = table[column].map(lambda value: f"{value:.1%}" if pd.notna(value) else "")
        elif pd.api.types.is_float_dtype(table[column]):
            table[column] = table[column].map(lambda value: f"{value:.4f}" if pd.notna(value) else "")
    return table


def _multiselect(label: str, df: pd.DataFrame, column: str) -> list[str]:
    if column not in df.columns:
        return []
    options = _options(df[column])
    return st.sidebar.multiselect(label, options=options)


def _options(values: Iterable[object]) -> list[str]:
    series = pd.Series(values).fillna("missing").astype(str)
    return sorted(series.unique().tolist())


def _status_count(df: pd.DataFrame, status: str) -> int:
    if "selection_status" not in df.columns:
        return 0
    return int(df["selection_status"].eq(status).sum())


def _warning_count(df: pd.DataFrame) -> int:
    if "overfit_warning_flag" in df.columns:
        return int(df["overfit_warning_flag"].fillna(False).astype(bool).sum())
    return _status_count(df, "overfit_warning")


def _inject_style() -> None:
    st.markdown(
        """
        <style>
        :root {
            --wr-bg: #15001f;
            --wr-bg-2: #250034;
            --wr-surface: #170820;
            --wr-surface-2: #21102e;
            --wr-surface-3: #2d1041;
            --wr-line: rgba(255, 255, 255, 0.10);
            --wr-line-strong: rgba(255, 255, 255, 0.22);
            --wr-text: #f8efff;
            --wr-muted: #c59bd9;
            --wr-dim: #8f69a5;
            --wr-cyan: #25f4d0;
            --wr-pink: #ff4fab;
            --wr-yellow: #ffe66d;
            --wr-violet: #7c5cff;
        }

        header[data-testid="stHeader"] {
            background: transparent;
            height: 0;
        }

        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        #MainMenu,
        footer {
            visibility: hidden;
            height: 0;
        }

        .stApp {
            color: var(--wr-text);
            background:
                radial-gradient(circle at 12% 8%, rgba(128, 31, 190, 0.34), transparent 34%),
                radial-gradient(circle at 82% 18%, rgba(255, 79, 171, 0.16), transparent 32%),
                linear-gradient(135deg, #0d0016 0%, #1b0029 42%, #30003f 100%);
            overflow-x: hidden;
        }

        html, body {
            overflow-x: hidden;
        }

        .block-container {
            max-width: 1500px;
            padding: 0.95rem 1.15rem 2.5rem;
        }

        .sidebar-brand {
            display: flex;
            align-items: center;
            gap: 12px;
            margin: 2px 0 12px;
            padding: 8px 2px;
            color: var(--wr-text);
        }

        .brand-mark {
            display: grid;
            place-items: center;
            width: 36px;
            height: 36px;
            border: 1px solid rgba(255, 79, 171, 0.46);
            border-radius: 10px;
            color: var(--wr-cyan);
            background: linear-gradient(135deg, rgba(255, 79, 171, 0.18), rgba(37, 244, 208, 0.12));
            font-size: 0.78rem;
            font-weight: 780;
        }

        .sidebar-brand strong {
            color: var(--wr-text);
            font-size: 1rem;
            line-height: 1;
        }

        .sidebar-brand p {
            margin: 3px 0 0;
            color: var(--wr-dim) !important;
            font-size: 0.74rem;
        }

        .section-marker {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            margin: 0.95rem 0 0.65rem;
            padding: 10px 0 8px;
            border-bottom: 1px solid var(--wr-line);
        }

        .section-marker strong {
            color: var(--wr-text);
            font-size: 1.08rem;
            font-weight: 760;
        }

        .section-marker span {
            color: var(--wr-muted);
            font-size: 0.82rem;
        }

        .app-header {
            position: relative;
            margin-bottom: 0.75rem;
            padding: 12px 16px;
            overflow: hidden;
            border: 1px solid var(--wr-line);
            border-radius: 10px;
            background:
                linear-gradient(115deg, rgba(42, 9, 62, 0.96) 0%, rgba(28, 4, 42, 0.96) 54%, rgba(68, 7, 91, 0.86) 100%);
            box-shadow: 0 8px 22px rgba(0, 0, 0, 0.22);
        }

        .app-header::before {
            content: "";
            position: absolute;
            inset: 0 0 auto 0;
            height: 3px;
            background: linear-gradient(90deg, var(--wr-cyan) 0%, var(--wr-pink) 47%, var(--wr-yellow) 100%);
        }

        .app-header h1 {
            margin: 0;
            color: var(--wr-text);
            font-size: 1.45rem;
            line-height: 1.15;
            letter-spacing: 0;
            font-weight: 760;
        }

        .app-source {
            margin: 6px 0 0;
            color: var(--wr-muted);
            font-size: 0.78rem;
            word-break: break-word;
        }

        section[data-testid="stSidebar"] {
            width: 232px !important;
            min-width: 232px !important;
            border-right: 1px solid var(--wr-line);
            background:
                linear-gradient(180deg, #13001e 0%, #21002f 52%, #16001f 100%);
        }

        section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
            padding: 0.8rem 0.65rem 1.6rem 1.05rem;
        }

        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: var(--wr-text);
            letter-spacing: 0;
        }

        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span {
            color: var(--wr-muted);
        }

        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
            margin-top: 0.2rem;
            margin-bottom: 0.7rem;
            font-size: 1.05rem;
        }

        .sidebar-title {
            margin: 4px 0 8px;
            color: var(--wr-text);
            font-size: 0.95rem;
            font-weight: 760;
        }

        section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            gap: 0.55rem;
        }

        section[data-testid="stSidebar"] [data-testid="stExpander"] {
            margin-top: 0.2rem;
            border-radius: 8px;
            box-shadow: none;
        }

        .metric-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.8rem;
            margin: 0.75rem 0 0.9rem;
        }

        .metric-card {
            position: relative;
            display: flex;
            gap: 12px;
            min-height: 92px;
            padding: 14px 16px;
            overflow: hidden;
            border: 1px solid var(--wr-line);
            border-radius: 10px;
            background: linear-gradient(180deg, rgba(29, 13, 45, 0.98), rgba(16, 5, 26, 0.98));
            box-shadow: 0 8px 22px rgba(0, 0, 0, 0.22);
        }

        .metric-card::before {
            content: "";
            position: absolute;
            inset: 0 0 auto 0;
            height: 3px;
            background: var(--metric-color);
        }

        .metric-card.cyan { --metric-color: var(--wr-violet); }
        .metric-card.teal { --metric-color: var(--wr-cyan); }
        .metric-card.pink { --metric-color: var(--wr-pink); }
        .metric-card.yellow { --metric-color: var(--wr-yellow); }

        .metric-icon {
            display: grid;
            place-items: center;
            flex: 0 0 46px;
            width: 42px;
            height: 42px;
            margin-top: 2px;
            border-radius: 50%;
            color: var(--metric-color);
            background: color-mix(in srgb, var(--metric-color) 18%, transparent);
            border: 1px solid color-mix(in srgb, var(--metric-color) 38%, transparent);
            font-weight: 800;
        }

        .metric-card p {
            margin: 0;
            color: var(--wr-muted);
            font-size: 0.88rem;
        }

        .metric-card strong {
            display: block;
            margin-top: 4px;
            color: var(--wr-text);
            font-size: 1.65rem;
            line-height: 1;
        }

        .metric-card span {
            display: block;
            margin-top: 8px;
            color: var(--wr-dim);
            font-size: 0.8rem;
        }

        .panel {
            border: 1px solid var(--wr-line);
            border-radius: 10px;
            background: linear-gradient(180deg, rgba(28, 13, 43, 0.97), rgba(17, 7, 28, 0.97));
            box-shadow: 0 8px 22px rgba(0, 0, 0, 0.22);
        }

        .table-panel {
            margin: 0;
            padding: 16px;
        }

        .primary-title {
            margin: 0.9rem 0 0.4rem;
            font-size: 1.22rem;
        }

        .compact-table-panel {
            min-height: 380px;
        }

        .panel-title {
            margin: 0 0 12px;
            color: var(--wr-text);
            font-size: 1.12rem;
            font-weight: 760;
            letter-spacing: 0;
        }

        .panel-title.outside {
            margin: 0 0 8px;
        }

        .panel-title span {
            color: var(--wr-muted);
            font-size: 0.78rem;
            font-weight: 520;
        }

        .subpanel-title {
            margin: 0 0 10px;
            color: var(--wr-text);
            font-size: 0.92rem;
            font-weight: 680;
        }

        .model-table-wrap {
            overflow-x: auto;
            border: 1px solid var(--wr-line);
            border-radius: 8px;
            background: rgba(13, 4, 22, 0.72);
        }

        .model-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.82rem;
        }

        .compact-table-panel .model-table {
            font-size: 0.75rem;
        }

        .model-table th {
            padding: 11px 12px;
            color: #d8c5e9;
            text-align: left;
            font-weight: 650;
            background: rgba(255, 255, 255, 0.045);
            border-bottom: 1px solid var(--wr-line);
            white-space: nowrap;
        }

        .model-table td {
            padding: 10px 12px;
            color: var(--wr-text);
            border-bottom: 1px solid rgba(255, 255, 255, 0.07);
            white-space: nowrap;
        }

        .model-table tr:last-child td {
            border-bottom: 0;
        }

        .model-cell {
            min-width: 440px;
        }

        .compact-table-panel .model-cell {
            min-width: 210px;
            max-width: 280px;
            white-space: normal;
            line-height: 1.25;
        }

        .text-cell {
            max-width: 150px;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .compact-table-panel .text-cell {
            max-width: 118px;
        }

        .model-cell i {
            display: inline-block;
            width: 9px;
            height: 9px;
            margin-right: 11px;
            border-radius: 50%;
            vertical-align: 1px;
        }

        .model-cell i.cyan { background: var(--wr-cyan); box-shadow: 0 0 10px rgba(37, 244, 208, 0.4); }
        .model-cell i.pink { background: var(--wr-pink); box-shadow: 0 0 10px rgba(255, 79, 171, 0.4); }
        .model-cell i.yellow { background: var(--wr-yellow); box-shadow: 0 0 10px rgba(255, 230, 109, 0.36); }
        .model-cell i.violet { background: var(--wr-violet); box-shadow: 0 0 10px rgba(124, 92, 255, 0.42); }

        .numeric {
            text-align: right;
            font-variant-numeric: tabular-nums;
        }

        .metric-focus {
            color: var(--wr-cyan) !important;
            font-weight: 760;
        }

        .status-badge {
            display: inline-flex;
            align-items: center;
            min-height: 22px;
            padding: 2px 8px;
            border: 1px solid var(--badge-color);
            border-radius: 6px;
            color: var(--badge-color);
            background: color-mix(in srgb, var(--badge-color) 13%, transparent);
            font-size: 0.76rem;
        }

        .status-badge.accepted { --badge-color: var(--wr-cyan); }
        .status-badge.warning { --badge-color: #f59e0b; }
        .status-badge.floor { --badge-color: var(--wr-pink); }
        .status-badge.neutral { --badge-color: var(--wr-muted); }

        .selected-panel {
            padding: 16px;
            margin-bottom: 0.8rem;
        }

        .selected-head {
            display: flex;
            justify-content: space-between;
            gap: 12px;
            align-items: flex-start;
            margin-bottom: 12px;
        }

        .selected-label {
            color: var(--wr-cyan);
            font-size: 0.92rem;
            font-weight: 760;
            line-height: 1.35;
            word-break: break-word;
        }

        .status-chip {
            flex: 0 0 auto;
            padding: 5px 8px;
            border: 1px solid rgba(37, 244, 208, 0.36);
            border-radius: 8px;
            color: var(--wr-cyan);
            background: rgba(37, 244, 208, 0.08);
            font-size: 0.72rem;
        }

        .selected-metrics {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            border: 1px solid var(--wr-line);
            border-radius: 8px;
            overflow: hidden;
            background: rgba(255, 255, 255, 0.03);
        }

        .selected-metric {
            padding: 11px 10px;
            border-right: 1px solid var(--wr-line);
            border-bottom: 1px solid var(--wr-line);
            text-align: center;
        }

        .selected-metric:nth-child(3n) {
            border-right: 0;
        }

        .selected-metric:nth-last-child(-n + 3) {
            border-bottom: 0;
        }

        .selected-metric p {
            margin: 0 0 6px;
            color: var(--wr-muted);
            font-size: 0.7rem;
        }

        .selected-metric strong {
            display: block;
            color: var(--wr-text);
            font-size: 1.25rem;
            line-height: 1;
            font-variant-numeric: tabular-nums;
        }

        .selected-metric span {
            display: block;
            margin-top: 4px;
            color: var(--wr-muted);
            font-size: 0.72rem;
        }

        .selected-metric.active strong,
        .selected-metric.active span {
            color: var(--wr-cyan);
        }

        .artifact-panel {
            min-height: 0;
            margin-top: 0.85rem;
            padding: 14px;
            border: 1px solid var(--wr-line);
            border-radius: 10px;
            background: rgba(22, 7, 34, 0.94);
        }

        .artifact-panel .artifact-row {
            grid-template-columns: 160px 90px minmax(0, 1fr);
            align-items: baseline;
        }

        .artifact-row {
            display: grid;
            grid-template-columns: auto auto;
            gap: 4px 10px;
            padding: 9px 0;
            border-top: 1px solid rgba(255, 255, 255, 0.10);
            color: var(--wr-text);
            background: transparent;
        }

        .artifact-row.cyan { --artifact-color: var(--wr-cyan); }
        .artifact-row.pink { --artifact-color: var(--wr-pink); }
        .artifact-row.yellow { --artifact-color: var(--wr-yellow); }

        .artifact-row span {
            color: var(--artifact-color);
            font-weight: 720;
            font-size: 0.8rem;
        }

        .artifact-row em {
            justify-self: end;
            color: var(--wr-muted);
            font-size: 0.72rem;
            font-style: normal;
        }

        .artifact-row code {
            grid-column: auto;
            max-width: 100%;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            color: var(--wr-dim);
            background: transparent;
            font-size: 0.66rem;
        }

        div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stMetric"]) {
            gap: 0.8rem;
        }

        div[data-testid="stMetric"] {
            min-height: 104px;
            border: 1px solid var(--wr-line);
            border-radius: 12px;
            padding: 14px 16px;
            background:
                linear-gradient(180deg, rgba(43, 14, 62, 0.98) 0%, rgba(22, 8, 32, 0.98) 100%);
            box-shadow: 0 8px 22px rgba(0, 0, 0, 0.22);
        }

        div[data-testid="stMetricLabel"] p {
            color: var(--wr-muted);
            font-size: 0.86rem;
            font-weight: 620;
        }

        div[data-testid="stMetricValue"] {
            color: var(--wr-text);
            font-weight: 760;
        }

        div[data-testid="stMetricDelta"] {
            color: var(--wr-cyan);
        }

        h2, h3 {
            color: var(--wr-text);
            letter-spacing: 0;
        }

        div[data-testid="stDataFrame"],
        div[data-testid="stTable"] {
            border: 1px solid var(--wr-line);
            border-radius: 10px;
            overflow: hidden;
            background: var(--wr-surface);
            box-shadow: 0 8px 22px rgba(0, 0, 0, 0.22);
        }

        div[data-testid="stAlert"] {
            border-radius: 10px;
            border-color: var(--wr-line);
            background: rgba(33, 16, 46, 0.92);
            color: var(--wr-text);
        }

        div[data-baseweb="select"] > div,
        div[data-testid="stNumberInput"] input,
        div[data-testid="stTextInput"] input {
            border-radius: 10px;
            border-color: var(--wr-line-strong);
            background: rgba(20, 4, 31, 0.92);
            color: var(--wr-text);
        }

        div[data-baseweb="select"] > div:focus-within,
        div[data-testid="stNumberInput"] input:focus,
        div[data-testid="stTextInput"] input:focus {
            border-color: var(--wr-cyan);
            box-shadow: 0 0 0 3px rgba(37, 244, 208, 0.16);
        }

        div[data-baseweb="select"] input,
        div[data-baseweb="select"] span,
        div[data-testid="stNumberInput"] input {
            color: var(--wr-text) !important;
        }

        div[data-testid="stNumberInput"] button {
            background: #0f0018;
            border-color: var(--wr-line-strong);
            color: var(--wr-text);
        }

        button[kind="secondary"],
        button[data-testid="baseButton-secondary"] {
            border-radius: 10px;
            border-color: var(--wr-line-strong);
            color: var(--wr-text);
            background: linear-gradient(180deg, #2a0a3d 0%, #170820 100%);
        }

        [data-testid="stExpander"] {
            border: 1px solid var(--wr-line);
            border-radius: 10px;
            background: var(--wr-surface);
            box-shadow: 0 8px 22px rgba(0, 0, 0, 0.22);
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
            border-color: var(--wr-line);
            border-radius: 10px;
            background: rgba(22, 7, 34, 0.94);
            box-shadow: 0 8px 22px rgba(0, 0, 0, 0.22);
        }

        div[data-testid="stImage"] {
            padding: 10px;
            border: 1px solid var(--wr-line);
            border-radius: 12px;
            background: linear-gradient(180deg, #21102e 0%, #160820 100%);
        }

        [data-testid="stExpander"] div[data-testid="stImage"] {
            margin-bottom: 1rem;
        }

        div[data-testid="stMarkdownContainer"] p,
        div[data-testid="stCaptionContainer"],
        .stCaptionContainer {
            color: var(--wr-muted);
        }

        .st-emotion-cache-1r6slb0,
        .st-emotion-cache-12w0qpk {
            background: transparent;
        }

        code {
            border-radius: 8px;
            color: #f6edff;
            background: #100018;
        }

        canvas {
            border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default="configs/models.yaml")
    return parser.parse_known_args()[0]


if __name__ == "__main__":
    main()
