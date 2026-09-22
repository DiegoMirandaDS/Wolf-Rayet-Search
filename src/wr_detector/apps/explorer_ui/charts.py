"""Altair chart builders for the Model Explorer.

All charts use ``width="container"`` and bounded heights so they scale
with the page instead of overflowing it, and they inherit the Streamlit
theme (transparent backgrounds, theme fonts and grid colors).
"""

from __future__ import annotations

from typing import Literal

import altair as alt
import numpy as np
import pandas as pd

CLASS_COLORS = alt.Scale(domain=["WR", "negative"], range=["#f28e2b", "#4e79a7"])
SPLIT_COLORS = alt.Scale(domain=["train", "cv", "holdout"], range=["#59a14f", "#edc948", "#e15759"])
CATEGORY_SCHEME = "tableau10"

DIAGNOSTIC_ORDER = [
    "Background",
    "True negative",
    "Contaminant @K",
    "False positive",
    "WR outside @K",
    "False negative",
    "WR recovered @K",
    "True positive",
]

DIAGNOSTIC_COLORS = {
    "Background": "#4e79a7",
    "True negative": "#4e79a7",
    "Contaminant @K": "#ff79c6",
    "False positive": "#ff79c6",
    "WR outside @K": "#edc948",
    "False negative": "#edc948",
    "WR recovered @K": "#f28e2b",
    "True positive": "#f28e2b",
}

DIAGNOSTIC_MARKS = {
    "Background": {"shape": "circle", "size": 18, "opacity": 0.18, "filled": True},
    "True negative": {"shape": "circle", "size": 18, "opacity": 0.18, "filled": True},
    "Contaminant @K": {"shape": "triangle-up", "size": 46, "opacity": 0.72, "filled": True},
    "False positive": {"shape": "triangle-up", "size": 46, "opacity": 0.72, "filled": True},
    "WR outside @K": {"shape": "diamond", "size": 68, "opacity": 0.9, "filled": False},
    "False negative": {"shape": "diamond", "size": 68, "opacity": 0.9, "filled": False},
    "WR recovered @K": {"shape": "circle", "size": 78, "opacity": 0.95, "filled": True},
    "True positive": {"shape": "circle", "size": 78, "opacity": 0.95, "filled": True},
}


def metric_bar(
    df: pd.DataFrame,
    *,
    value_col: str,
    value_title: str,
    label_col: str = "short_label",
    percent: bool = False,
    orientation: Literal["horizontal", "vertical"] = "horizontal",
) -> alt.Chart | None:
    columns = [column for column in {label_col, value_col, "model", "selection_status"} if column in df.columns]
    data = df[columns].dropna(subset=[value_col]).copy()
    if data.empty:
        return None
    value_axis = alt.Axis(format=".0%") if percent else alt.Axis()
    value_format = ".1%" if percent else ".4f"
    tooltip = [
        alt.Tooltip(f"{label_col}:N", title="model"),
        alt.Tooltip(f"{value_col}:Q", title=value_title, format=value_format),
    ]
    if "selection_status" in data.columns:
        tooltip.append(alt.Tooltip("selection_status:N", title="status"))

    if orientation == "vertical":
        height = 300
        bars = (
            alt.Chart(data)
            .mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
            .encode(
                x=alt.X(
                    f"{label_col}:N",
                    sort="-y",
                    title=None,
                    axis=alt.Axis(
                        labelAngle=-35,
                        labelLimit=140,
                        labelOverlap=False,
                    ),
                    scale=alt.Scale(paddingInner=0.45, paddingOuter=0.2),
                ),
                y=alt.Y(
                    f"{value_col}:Q",
                    title=value_title,
                    axis=value_axis,
                    scale=alt.Scale(zero=True),
                ),
                color=alt.Color(
                    "model:N",
                    scale=alt.Scale(scheme=CATEGORY_SCHEME),
                    legend=None,
                ),
                tooltip=tooltip,
            )
        )
        labels = bars.mark_text(dy=-7, fontSize=11).encode(
            text=alt.Text(f"{value_col}:Q", format=value_format),
            color=alt.value("#e6e9ef"),
        )
        return (bars + labels).properties(width="container", height=height)

    height = min(560, max(180, 32 * len(data) + 36))
    return (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=2, size=18)
        .encode(
            y=alt.Y(f"{label_col}:N", sort="-x", title=None, axis=alt.Axis(labelLimit=320)),
            x=alt.X(f"{value_col}:Q", title=value_title, axis=value_axis),
            color=alt.Color("model:N", scale=alt.Scale(scheme=CATEGORY_SCHEME), legend=alt.Legend(title=None, orient="bottom")),
            tooltip=tooltip,
        )
        .properties(width="container", height=height)
    )


def recovery_lines(ranked: pd.DataFrame, *, ks: list[int]) -> alt.Chart | None:
    label_col = "chart_label" if "chart_label" in ranked.columns else "short_label"
    rows = []
    for _, row in ranked.iterrows():
        for k in ks:
            pct = row.get(f"holdout_wr_at_{k}_pct")
            if pd.isna(pct):
                continue
            rows.append(
                {
                    "series_label": row[label_col],
                    "model_label": row["short_label"],
                    "budget": k,
                    "recovered_pct": float(pct),
                    "recovered": row.get(f"holdout_wr_at_{k}"),
                }
            )
    if not rows:
        return None
    data = pd.DataFrame(rows)
    return (
        alt.Chart(data)
        .mark_line(point=True)
        .encode(
            x=alt.X("budget:Q", scale=alt.Scale(type="log"), title="candidates reviewed", axis=alt.Axis(values=ks)),
            y=alt.Y("recovered_pct:Q", title="WR holdout recovered", axis=alt.Axis(format=".0%")),
            color=alt.Color(
                "series_label:N",
                scale=alt.Scale(scheme=CATEGORY_SCHEME),
                legend=alt.Legend(
                    title=None,
                    orient="bottom",
                    columns=2,
                    labelLimit=180,
                ),
            ),
            tooltip=[
                alt.Tooltip("model_label:N", title="model"),
                alt.Tooltip("budget:Q", title="budget"),
                alt.Tooltip("recovered:Q", title="WR recovered"),
                alt.Tooltip("recovered_pct:Q", title="recovered", format=".1%"),
            ],
        )
        .properties(width="container", height=300)
    )


def dataset_heatmap(df: pd.DataFrame, *, value_col: str, value_title: str, percent: bool = False) -> alt.Chart | None:
    data = df.dropna(subset=[value_col]).copy()
    if data.empty:
        return None
    model_labels = {
        "random_forest": "RF",
        "hist_gradient_boosting": "HGB",
        "xgboost": "XGB",
    }
    sampler_labels = {
        "none": "none",
        "smote": "SMOTE",
        "smote_enn": "SMOTE-ENN",
    }
    if "includes_parallax_error" in data.columns:
        feature_suffix = data["includes_parallax_error"].fillna(False).map(
            {True: "+err", False: "colors"}
        )
    elif "feature_set" in data.columns:
        feature_suffix = data["feature_set"].astype(str).map(
            lambda value: "+err" if "error" in value else "colors"
        )
    else:
        feature_suffix = pd.Series("colors", index=data.index)
    data["column_label"] = (
        data["model"].astype(str).replace(model_labels)
        + "/"
        + data["sampler"].astype(str).replace(sampler_labels)
        + " "
        + feature_suffix
    )
    if (
        "negative_ratio_label" in data.columns
        and data["negative_ratio_label"].astype(str).nunique() > 1
    ):
        data["column_label"] += " " + data["negative_ratio_label"].astype(str)
    aggregated = (
        data.groupby(["dataset_variant", "column_label"], as_index=False)[value_col].max()
    )
    height = min(360, max(210, 38 * aggregated["dataset_variant"].nunique() + 70))
    value_format = ".0%" if percent else ".3f"
    base = alt.Chart(aggregated).encode(
        x=alt.X(
            "column_label:N",
            title=None,
            axis=alt.Axis(labelAngle=-25, labelLimit=130),
        ),
        y=alt.Y(
            "dataset_variant:N",
            title=None,
            axis=alt.Axis(labelLimit=160),
        ),
    )
    legend = alt.Legend(title=value_title, format=".0%") if percent else alt.Legend(title=value_title)
    rect = base.mark_rect().encode(
        color=alt.Color(f"{value_col}:Q", scale=alt.Scale(scheme="viridis"), legend=legend),
        tooltip=[
            alt.Tooltip("dataset_variant:N", title="dataset"),
            alt.Tooltip("column_label:N", title="model"),
            alt.Tooltip(f"{value_col}:Q", title=value_title, format=value_format),
        ],
    )
    bright_cutoff = aggregated[value_col].min() + 0.6 * (
        aggregated[value_col].max() - aggregated[value_col].min() or 1.0
    )
    text = base.mark_text(fontSize=11).encode(
        text=alt.Text(f"{value_col}:Q", format=value_format),
        color=alt.condition(alt.datum[value_col] > bright_cutoff, alt.value("black"), alt.value("white")),
    )
    return (rect + text).properties(width="container", height=height)


def stability_gap_bars(ranked: pd.DataFrame) -> alt.Chart | None:
    label_col = "chart_label" if "chart_label" in ranked.columns else "short_label"
    required = {label_col, "cv_train_gap_f2", "holdout_cv_gap_f2"}
    if not required.issubset(ranked.columns):
        return None
    data = ranked.dropna(subset=["cv_train_gap_f2", "holdout_cv_gap_f2"]).copy()
    if data.empty:
        return None
    long = data.melt(
        id_vars=[label_col],
        value_vars=["cv_train_gap_f2", "holdout_cv_gap_f2"],
        var_name="gap",
        value_name="value",
    )
    long["gap"] = long["gap"].map(
        {
            "cv_train_gap_f2": "Train - CV",
            "holdout_cv_gap_f2": "Holdout - CV",
        }
    )
    bars = (
        alt.Chart(long)
        .mark_bar(size=10)
        .encode(
            y=alt.Y(
                f"{label_col}:N",
                title=None,
                sort="-x",
                axis=alt.Axis(labelLimit=210),
            ),
            yOffset=alt.YOffset("gap:N"),
            x=alt.X("value:Q", title="F2 gap"),
            color=alt.Color(
                "gap:N",
                legend=alt.Legend(
                    title=None,
                    orient="bottom",
                    columns=2,
                    labelLimit=100,
                ),
            ),
            tooltip=[
                alt.Tooltip(f"{label_col}:N", title="model"),
                alt.Tooltip("gap:N"),
                alt.Tooltip("value:Q", format=".3f"),
            ],
        )
    )
    threshold = (
        alt.Chart(pd.DataFrame({"x": [0.15]}))
        .mark_rule(strokeDash=[5, 4], color="#e15759")
        .encode(x="x:Q")
    )
    height = min(420, max(220, 36 * data[label_col].nunique() + 60))
    return (bars + threshold).properties(width="container", height=height)


def split_metric_bars(row: pd.Series) -> alt.Chart | None:
    metric_specs = [
        ("average_precision", "AP"),
        ("roc_auc", "ROC AUC"),
        ("precision_wr", "Precision"),
        ("recall_wr", "Recall"),
        ("f2_wr", "F2"),
    ]
    rows = []
    for split in ["train", "cv", "holdout"]:
        for metric, label in metric_specs:
            value = row.get(f"{split}_{metric}")
            if value is not None and pd.notna(value):
                rows.append({"split": split, "metric": label, "value": float(value)})
    if not rows:
        return None
    data = pd.DataFrame(rows)
    return (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X("metric:N", title=None, sort=[label for _, label in metric_specs]),
            xOffset=alt.XOffset("split:N", sort=["train", "cv", "holdout"]),
            y=alt.Y("value:Q", title=None, scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("split:N", scale=SPLIT_COLORS, sort=["train", "cv", "holdout"], legend=alt.Legend(title=None, orient="bottom")),
            tooltip=[
                alt.Tooltip("split:N"),
                alt.Tooltip("metric:N"),
                alt.Tooltip("value:Q", format=".4f"),
            ],
        )
        .properties(width="container", height=300)
    )


def importance_bars(importance: pd.DataFrame, *, top_n: int = 16) -> alt.Chart | None:
    if importance.empty:
        return None
    top = importance.sort_values("importance_mean", ascending=False).head(top_n).copy()
    height = max(160, 26 * len(top) + 30)
    bars = (
        alt.Chart(top)
        .mark_bar(cornerRadiusEnd=2)
        .encode(
            y=alt.Y("feature:N", sort="-x", title=None),
            x=alt.X("importance_mean:Q", title="importance"),
            color=alt.value("#4e79a7"),
            tooltip=[
                alt.Tooltip("feature:N"),
                alt.Tooltip("importance_mean:Q", format=".4f"),
                alt.Tooltip("importance_std:Q", format=".4f"),
            ],
        )
    )
    layers = [bars]
    if "importance_std" in top.columns and top["importance_std"].notna().any():
        spread = top.assign(
            low=top["importance_mean"] - top["importance_std"],
            high=top["importance_mean"] + top["importance_std"],
        )
        layers.append(
            alt.Chart(spread)
            .mark_rule(color="#e6e9ef", opacity=0.8)
            .encode(y=alt.Y("feature:N", sort="-x", title=None), x="low:Q", x2="high:Q")
        )
    return alt.layer(*layers).properties(width="container", height=height)


def score_histogram(cases: pd.DataFrame) -> alt.Chart | None:
    data = cases.dropna(subset=["score"]).copy()
    if data.empty:
        return None
    data["class"] = data["target"].map({1: "WR", 0: "negative"})
    return (
        alt.Chart(data[["score", "class"]])
        .mark_bar(opacity=0.75, binSpacing=0)
        .encode(
            x=alt.X("score:Q", bin=alt.Bin(maxbins=40), title="model score"),
            y=alt.Y("count():Q", title="sources", scale=alt.Scale(type="symlog")),
            color=alt.Color("class:N", scale=CLASS_COLORS, legend=alt.Legend(title=None, orient="bottom")),
            tooltip=[alt.Tooltip("class:N"), alt.Tooltip("count():Q", title="sources")],
        )
        .properties(width="container", height=280)
    )


def subtype_recovery_lines(recovery: pd.DataFrame, *, ks: list[int]) -> alt.Chart | None:
    if recovery.empty:
        return None
    rows = []
    for _, row in recovery.iterrows():
        for k in ks:
            pct = row.get(f"recovered_at_{k}_pct")
            if pd.isna(pct):
                continue
            rows.append(
                {
                    "wr_subtype": row["wr_subtype"],
                    "budget": k,
                    "recovered_pct": float(pct),
                    "recovered": row.get(f"recovered_at_{k}"),
                    "total": row["total"],
                }
            )
    if not rows:
        return None
    data = pd.DataFrame(rows)
    return (
        alt.Chart(data)
        .mark_line(point=True)
        .encode(
            x=alt.X("budget:Q", scale=alt.Scale(type="log"), title="candidates reviewed", axis=alt.Axis(values=ks)),
            y=alt.Y("recovered_pct:Q", title="recovered", axis=alt.Axis(format=".0%")),
            color=alt.Color("wr_subtype:N", scale=alt.Scale(scheme=CATEGORY_SCHEME), legend=alt.Legend(title="subtype", orient="bottom")),
            tooltip=[
                alt.Tooltip("wr_subtype:N", title="subtype"),
                alt.Tooltip("budget:Q"),
                alt.Tooltip("recovered:Q"),
                alt.Tooltip("total:Q"),
                alt.Tooltip("recovered_pct:Q", format=".1%"),
            ],
        )
        .properties(width="container", height=300)
    )


def composition_bars(composition: pd.DataFrame) -> alt.Chart | None:
    if composition.empty:
        return None
    height = max(140, 26 * len(composition) + 30)
    return (
        alt.Chart(composition)
        .mark_bar(cornerRadiusEnd=2, color="#4e79a7")
        .encode(
            y=alt.Y("simbad_main_type:N", sort="-x", title=None),
            x=alt.X("count:Q", title="false positives"),
            tooltip=[
                alt.Tooltip("simbad_main_type:N", title="SIMBAD type"),
                alt.Tooltip("count:Q"),
                alt.Tooltip("share:Q", format=".1%"),
            ],
        )
        .properties(width="container", height=height)
    )


def precision_recall_chart(
    curve: pd.DataFrame, *, operating: pd.DataFrame | None = None
) -> alt.Chart | None:
    if curve.empty:
        return None
    line = (
        alt.Chart(curve)
        .mark_line(color="#f28e2b", interpolate="step-after")
        .encode(
            x=alt.X("recall:Q", title="recall", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("precision:Q", title="precision", scale=alt.Scale(domain=[0, 1])),
            tooltip=[
                alt.Tooltip("recall:Q", format=".3f"),
                alt.Tooltip("precision:Q", format=".3f"),
                alt.Tooltip("score:Q", title="score cutoff", format=".4f"),
            ],
        )
    )
    layers = [line]
    if operating is not None and not operating.empty:
        layers.append(
            alt.Chart(operating)
            .mark_point(size=140, filled=True, color="#e15759")
            .encode(
                x="recall:Q",
                y="precision:Q",
                tooltip=[
                    alt.Tooltip("recall:Q", format=".3f"),
                    alt.Tooltip("precision:Q", format=".3f"),
                    alt.Tooltip("score:Q", title="threshold", format=".4f"),
                ],
            )
        )
    return alt.layer(*layers).properties(width="container", height=320)


def roc_chart(
    curve: pd.DataFrame, *, operating: pd.DataFrame | None = None
) -> alt.Chart | None:
    if curve.empty:
        return None
    diagonal = (
        alt.Chart(pd.DataFrame({"fpr": [0, 1], "tpr": [0, 1]}))
        .mark_line(strokeDash=[5, 4], color="#888", opacity=0.6)
        .encode(x="fpr:Q", y="tpr:Q")
    )
    line = (
        alt.Chart(curve)
        .mark_line(color="#4e79a7")
        .encode(
            x=alt.X("fpr:Q", title="false-positive rate", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("tpr:Q", title="true-positive rate", scale=alt.Scale(domain=[0, 1])),
            tooltip=[
                alt.Tooltip("fpr:Q", format=".4f"),
                alt.Tooltip("tpr:Q", format=".3f"),
                alt.Tooltip("score:Q", title="score cutoff", format=".4f"),
            ],
        )
    )
    layers = [diagonal, line]
    if operating is not None and not operating.empty:
        layers.append(
            alt.Chart(operating)
            .mark_point(size=140, filled=True, color="#e15759")
            .encode(x="fpr:Q", y="tpr:Q", tooltip=[alt.Tooltip("score:Q", title="threshold", format=".4f")])
        )
    return alt.layer(*layers).properties(width="container", height=320)

def confusion_matrix_chart(confusion: dict[str, int]) -> alt.Chart:
    data = pd.DataFrame(
        [
            {"actual": "WR", "predicted": "WR", "count": confusion["true_positive"]},
            {"actual": "WR", "predicted": "negative", "count": confusion["false_negative"]},
            {"actual": "negative", "predicted": "WR", "count": confusion["false_positive"]},
            {"actual": "negative", "predicted": "negative", "count": confusion["true_negative"]},
        ]
    )
    base = alt.Chart(data).encode(
        x=alt.X("predicted:N", title="predicted", sort=["WR", "negative"]),
        y=alt.Y("actual:N", title="actual", sort=["WR", "negative"]),
    )
    rect = base.mark_rect().encode(
        color=alt.Color(
            "count:Q",
            scale=alt.Scale(scheme="viridis", type="symlog"),
            legend=None,
        ),
        tooltip=[alt.Tooltip("actual:N"), alt.Tooltip("predicted:N"), alt.Tooltip("count:Q", format=",")],
    )
    text = base.mark_text(fontSize=15, fontWeight="bold", color="white").encode(
        text=alt.Text("count:Q", format=","),
    )
    return (rect + text).properties(width="container", height=220)


def retention_tradeoff_scatter(results: pd.DataFrame) -> alt.Chart | None:
    required = {"holdout_positive_retention", "threshold_calibration_negative_pass_rate"}
    if not required.issubset(results.columns):
        return None
    data = results.dropna(subset=list(required)).copy()
    if data.empty:
        return None
    rate_columns = {*required, "holdout_negative_pass_rate"}
    tooltip = [
        alt.Tooltip(column, format=".1%") if column in rate_columns else alt.Tooltip(column)
        for column in [
            "dataset_variant",
            "feature_set",
            "method",
            "subtype",
            "holdout_positive_retention",
            "threshold_calibration_negative_pass_rate",
            "holdout_negative_pass_rate",
        ]
        if column in data.columns
    ]
    return (
        alt.Chart(data)
        .mark_point(size=110, filled=True, opacity=0.85)
        .encode(
            x=alt.X(
                "threshold_calibration_negative_pass_rate:Q",
                title="calibration negatives passing (lower is better)",
                axis=alt.Axis(format=".0%"),
            ),
            y=alt.Y(
                "holdout_positive_retention:Q",
                title="holdout WR retained (higher is better)",
                axis=alt.Axis(format=".0%"),
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.Color("method:N", scale=alt.Scale(scheme=CATEGORY_SCHEME), legend=alt.Legend(title="method", orient="bottom")),
            shape=alt.Shape("subtype:N", legend=alt.Legend(title="subtype", orient="bottom")),
            tooltip=tooltip,
        )
        .interactive()
        .properties(width="container", height=380)
    )


def stack_recovery_lines(recovery: pd.DataFrame) -> alt.Chart | None:
    required = {"policy", "budget", "wr_recovered", "wr_recovered_pct"}
    if recovery.empty or not required.issubset(recovery.columns):
        return None
    data = recovery.dropna(subset=["budget", "wr_recovered"]).copy()
    if data.empty:
        return None
    budgets = sorted(data["budget"].astype(int).unique().tolist())
    return (
        alt.Chart(data)
        .mark_line(point=True, strokeWidth=2.5)
        .encode(
            x=alt.X(
                "budget:Q",
                scale=alt.Scale(type="log"),
                axis=alt.Axis(values=budgets),
                title="candidates reviewed",
            ),
            y=alt.Y("wr_recovered:Q", title="WR recovered"),
            color=alt.Color(
                "policy:N",
                scale=alt.Scale(
                    domain=["First stage", "Pass-first"],
                    range=["#4e79a7", "#f28e2b"],
                ),
                legend=alt.Legend(title=None, orient="bottom"),
            ),
            strokeDash=alt.StrokeDash(
                "policy:N",
                scale=alt.Scale(
                    domain=["First stage", "Pass-first"],
                    range=[[1, 0], [6, 3]],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("policy:N"),
                alt.Tooltip("budget:Q"),
                alt.Tooltip("wr_recovered:Q", title="WR recovered"),
                alt.Tooltip("wr_recovered_pct:Q", title="holdout recovery", format=".1%"),
                alt.Tooltip("negatives_reviewed:Q", title="negatives reviewed"),
                alt.Tooltip("candidates_per_wr:Q", title="candidates / WR", format=".2f"),
            ],
        )
        .properties(width="container", height=320)
    )


def stack_tradeoff_bars(tradeoff: pd.DataFrame) -> alt.Chart | None:
    required = {"input_budget", "wr_retention", "negative_removal_rate"}
    if tradeoff.empty or not required.issubset(tradeoff.columns):
        return None
    long = tradeoff[["input_budget", "wr_retention", "negative_removal_rate"]].melt(
        id_vars=["input_budget"],
        value_vars=["wr_retention", "negative_removal_rate"],
        var_name="metric",
        value_name="value",
    )
    long["metric"] = long["metric"].map(
        {
            "wr_retention": "WR retained",
            "negative_removal_rate": "negatives removed",
        }
    )
    return (
        alt.Chart(long)
        .mark_bar()
        .encode(
            x=alt.X("input_budget:N", title="first-stage input budget"),
            xOffset=alt.XOffset("metric:N"),
            y=alt.Y(
                "value:Q",
                title=None,
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format=".0%"),
            ),
            color=alt.Color(
                "metric:N",
                scale=alt.Scale(
                    domain=["WR retained", "negatives removed"],
                    range=["#f28e2b", "#4e79a7"],
                ),
                legend=alt.Legend(title=None, orient="bottom"),
            ),
            tooltip=[
                alt.Tooltip("input_budget:N", title="input budget"),
                alt.Tooltip("metric:N"),
                alt.Tooltip("value:Q", format=".1%"),
            ],
        )
        .properties(width="container", height=300)
    )


def color_magnitude(
    cases: pd.DataFrame,
    *,
    selected_source_id: int | None = None,
    x: str = "BP_RP",
    y: str = "G",
    color_color: bool = False,
) -> alt.Chart | None:
    """Photometric case scatter with explicit diagnostic draw order."""
    columns = [
        column
        for column in {
            x,
            y,
            "target",
            "diagnostic_state",
            "object_name",
            "rank",
            "score",
            "source_id",
            "simbad_main_type",
            "spectral_type",
        }
        if column in cases.columns
    ]
    data = cases[columns].dropna(subset=[x, y]).copy()
    if data.empty:
        return None
    if "diagnostic_state" not in data.columns:
        data["diagnostic_state"] = data["target"].map(
            {1: "WR recovered @K", 0: "Background"}
        )
    tooltip = _case_tooltip(data, extra_numeric=[x, y])
    layers = _diagnostic_point_layers(
        data,
        x=x,
        y=y,
        x_title=_axis_label(x),
        y_title=_axis_label(y),
        y_reverse=not color_color,
        tooltip=tooltip,
    )
    selected = _selected_case_layer(
        data,
        selected_source_id=selected_source_id,
        x=x,
        y=y,
        y_reverse=not color_color,
        tooltip=tooltip,
    )
    if selected is not None:
        layers.append(selected)
    return (
        alt.layer(*layers)
        .resolve_scale(color="shared")
        .interactive()
        .properties(width="container", height=420)
    )


def galactic_polar_chart(
    cases: pd.DataFrame,
    *,
    selected_source_id: int | None = None,
) -> alt.Chart | None:
    """North-polar projection of Galactic longitude and latitude."""
    required = {"polar_x", "polar_y", "galactic_l", "galactic_b"}
    if not required.issubset(cases.columns):
        return None
    data = cases[
        _case_chart_columns(cases, list(required))
    ].dropna(subset=["polar_x", "polar_y"]).copy()
    if data.empty:
        return None
    tooltip = _case_tooltip(
        data,
        extra_numeric=["galactic_l", "galactic_b"],
    )
    guide_rows = []
    for radius, latitude in [(45.0, 45), (90.0, 0), (135.0, -45)]:
        for angle in np.linspace(0, 2 * np.pi, 181):
            guide_rows.append(
                {
                    "guide": f"b={latitude}°",
                    "x": radius * np.sin(angle),
                    "y": radius * np.cos(angle),
                    "order": angle,
                }
            )
    guides = (
        alt.Chart(pd.DataFrame(guide_rows))
        .mark_line(color="#888", opacity=0.22, strokeWidth=1)
        .encode(
            x=alt.X(
                "x:Q",
                axis=_hidden_axis(),
                scale=alt.Scale(domain=[-180, 180]),
            ),
            y=alt.Y(
                "y:Q",
                axis=_hidden_axis(),
                scale=alt.Scale(domain=[-180, 180]),
            ),
            detail="guide:N",
            order="order:Q",
        )
    )
    labels = alt.Chart(
        pd.DataFrame(
            [
                {"x": 0, "y": 173, "label": "l=0°"},
                {"x": 173, "y": 0, "label": "l=90°"},
                {"x": 0, "y": -173, "label": "l=180°"},
                {"x": -173, "y": 0, "label": "l=270°"},
            ]
        )
    ).mark_text(color="#aaa", fontSize=10).encode(
        x=alt.X("x:Q", axis=_hidden_axis()),
        y=alt.Y("y:Q", axis=_hidden_axis()),
        text="label:N",
    )
    points = _diagnostic_point_layers(
        data,
        x="polar_x",
        y="polar_y",
        x_title=None,
        y_title=None,
        x_domain=[-180, 180],
        y_domain=[-180, 180],
        hide_axes=True,
        legend_columns=2,
        tooltip=tooltip,
    )
    selected = _selected_case_layer(
        data,
        selected_source_id=selected_source_id,
        x="polar_x",
        y="polar_y",
        tooltip=tooltip,
    )
    layers: list[alt.Chart] = [guides, labels, *points]
    if selected is not None:
        layers.append(selected)
    return (
        alt.layer(*layers)
        .resolve_scale(color="shared")
        .properties(width="container", height=430)
    )


def galactic_plane_map(
    cases: pd.DataFrame,
    *,
    selected_source_id: int | None = None,
    sun_distance_kpc: float = 8.122,
) -> alt.Chart | None:
    """Schematic Milky Way plane with parallax-qualified source positions."""
    required = {
        "galactocentric_x_kpc",
        "galactocentric_y_kpc",
        "distance_plotted",
    }
    if not required.issubset(cases.columns):
        return None
    data = cases[
        _case_chart_columns(
            cases,
            [
                "galactocentric_x_kpc",
                "galactocentric_y_kpc",
                "distance_plotted",
                "distance_kpc",
                "galactic_l",
                "galactic_b",
            ],
        )
    ]
    data = data[data["distance_plotted"].fillna(False)].dropna(
        subset=["galactocentric_x_kpc", "galactocentric_y_kpc"]
    )
    if data.empty:
        return None

    observed_extent = float(
        np.nanmax(
            np.abs(
                data[
                    ["galactocentric_x_kpc", "galactocentric_y_kpc"]
                ].to_numpy(dtype=float)
            )
        )
    )
    extent = min(30.0, max(17.0, np.ceil(observed_extent + 1.0)))
    rings = []
    for radius in [4.0, 8.0, 12.0, 16.0]:
        for angle in np.linspace(0, 2 * np.pi, 181):
            rings.append(
                {
                    "ring": radius,
                    "x": radius * np.cos(angle),
                    "y": radius * np.sin(angle),
                    "order": angle,
                }
            )
    ring_chart = (
        alt.Chart(pd.DataFrame(rings))
        .mark_line(color="#6272a4", opacity=0.18, strokeWidth=1)
        .encode(
            x=alt.X(
                "x:Q",
                title="Galactocentric X (kpc)",
                scale=alt.Scale(domain=[-extent, extent]),
            ),
            y=alt.Y(
                "y:Q",
                title="Galactocentric Y (kpc)",
                scale=alt.Scale(domain=[-extent, extent]),
            ),
            detail="ring:N",
            order="order:Q",
        )
    )
    arm_rows = []
    pitch = np.deg2rad(18.5)
    for arm in range(4):
        for order, radius in enumerate(np.linspace(2.6, 16.0, 180)):
            angle = np.log(radius / 2.6) / np.tan(pitch) + arm * np.pi / 2
            arm_rows.append(
                {
                    "arm": arm,
                    "x": radius * np.cos(angle),
                    "y": radius * np.sin(angle),
                    "order": order,
                }
            )
    arms = (
        alt.Chart(pd.DataFrame(arm_rows))
        .mark_line(color="#bd93f9", opacity=0.16, strokeWidth=8)
        .encode(x="x:Q", y="y:Q", detail="arm:N", order="order:Q")
    )
    landmarks = (
        alt.Chart(
            pd.DataFrame(
                [
                    {"x": 0.0, "y": 0.0, "label": "Galactic center", "kind": "center"},
                    {
                        "x": float(sun_distance_kpc),
                        "y": 0.0,
                        "label": "Sun",
                        "kind": "sun",
                    },
                ]
            )
        )
        .mark_point(size=120, filled=True)
        .encode(
            x="x:Q",
            y="y:Q",
            shape=alt.Shape(
                "kind:N",
                scale=alt.Scale(
                    domain=["center", "sun"],
                    range=["circle", "diamond"],
                ),
                legend=None,
            ),
            color=alt.Color(
                "kind:N",
                scale=alt.Scale(
                    domain=["center", "sun"],
                    range=["#f8f8f2", "#edc948"],
                ),
                legend=None,
            ),
            tooltip=alt.Tooltip("label:N"),
        )
    )
    tooltip = _case_tooltip(
        data,
        extra_numeric=[
            "galactic_l",
            "galactic_b",
            "distance_kpc",
            "galactocentric_x_kpc",
            "galactocentric_y_kpc",
        ],
    )
    points = _diagnostic_point_layers(
        data,
        x="galactocentric_x_kpc",
        y="galactocentric_y_kpc",
        x_title="Galactocentric X (kpc)",
        y_title="Galactocentric Y (kpc)",
        x_domain=[-extent, extent],
        y_domain=[-extent, extent],
        legend_columns=1,
        tooltip=tooltip,
    )
    selected = _selected_case_layer(
        data,
        selected_source_id=selected_source_id,
        x="galactocentric_x_kpc",
        y="galactocentric_y_kpc",
        tooltip=tooltip,
    )
    layers: list[alt.Chart] = [ring_chart, arms, landmarks, *points]
    if selected is not None:
        layers.append(selected)
    return (
        alt.layer(*layers)
        .resolve_scale(color="independent", shape="independent")
        .interactive()
        .properties(width="container", height=450)
    )


def _case_tooltip(
    data: pd.DataFrame,
    *,
    extra_numeric: list[str] | None = None,
) -> list[alt.Tooltip]:
    tooltip = [
        alt.Tooltip("object_name:N", title="object"),
        alt.Tooltip("diagnostic_state:N", title="state"),
        alt.Tooltip("rank:Q"),
        alt.Tooltip("score:Q", format=".4f"),
    ]
    for column in extra_numeric or []:
        if column in data.columns:
            tooltip.append(
                alt.Tooltip(
                    f"{column}:Q",
                    title=_axis_label(column),
                    format=".3f",
                )
            )
    if "simbad_main_type" in data.columns:
        tooltip.append(alt.Tooltip("simbad_main_type:N", title="SIMBAD type"))
    if "spectral_type" in data.columns:
        tooltip.append(alt.Tooltip("spectral_type:N", title="spectral type"))
    return tooltip


def _case_chart_columns(
    data: pd.DataFrame,
    extras: list[str],
) -> list[str]:
    desired = [
        "source_id",
        "target",
        "diagnostic_state",
        "object_name",
        "rank",
        "score",
        "simbad_main_type",
        "spectral_type",
        *extras,
    ]
    return list(dict.fromkeys(column for column in desired if column in data.columns))


def _diagnostic_point_layers(
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    x_title: str | None,
    y_title: str | None,
    tooltip: list[alt.Tooltip],
    y_reverse: bool = False,
    x_domain: list[float] | None = None,
    y_domain: list[float] | None = None,
    hide_axes: bool = False,
    legend_columns: int = 2,
) -> list[alt.Chart]:
    states = [
        state
        for state in DIAGNOSTIC_ORDER
        if state in set(data["diagnostic_state"].dropna().astype(str))
    ]
    colors = [DIAGNOSTIC_COLORS[state] for state in states]
    x_scale = (
        alt.Scale(zero=False, domain=x_domain)
        if x_domain is not None
        else alt.Scale(zero=False)
    )
    y_scale = (
        alt.Scale(zero=False, reverse=y_reverse, domain=y_domain)
        if y_domain is not None
        else alt.Scale(zero=False, reverse=y_reverse)
    )
    layers = []
    for index, state in enumerate(states):
        style = DIAGNOSTIC_MARKS[state]
        state_data = data[data["diagnostic_state"].eq(state)]
        layers.append(
            alt.Chart(state_data)
            .mark_point(
                shape=style["shape"],
                size=style["size"],
                opacity=style["opacity"],
                filled=style["filled"],
                strokeWidth=2 if not style["filled"] else 0.5,
            )
            .encode(
                x=alt.X(
                    f"{x}:Q",
                    title=None if hide_axes else x_title,
                    axis=_hidden_axis() if hide_axes else alt.Axis(),
                    scale=x_scale,
                ),
                y=alt.Y(
                    f"{y}:Q",
                    title=None if hide_axes else y_title,
                    axis=_hidden_axis() if hide_axes else alt.Axis(),
                    scale=y_scale,
                ),
                color=alt.Color(
                    "diagnostic_state:N",
                    scale=alt.Scale(domain=states, range=colors),
                    legend=(
                        alt.Legend(
                            title=None,
                            orient="bottom",
                            columns=legend_columns,
                            labelLimit=160,
                            symbolLimit=8,
                        )
                        if index == 0
                        else None
                    ),
                ),
                tooltip=tooltip,
            )
        )
    return layers


def _selected_case_layer(
    data: pd.DataFrame,
    *,
    selected_source_id: int | None,
    x: str,
    y: str,
    tooltip: list[alt.Tooltip],
    y_reverse: bool = False,
) -> alt.Chart | None:
    if selected_source_id is None or "source_id" not in data.columns:
        return None
    selected = data[data["source_id"].eq(selected_source_id)]
    if selected.empty:
        return None
    return (
        alt.Chart(selected)
        .mark_point(
            shape="diamond",
            size=360,
            filled=False,
            color="#e15759",
            strokeWidth=3,
        )
        .encode(
            x=alt.X(f"{x}:Q"),
            y=alt.Y(
                f"{y}:Q",
                scale=alt.Scale(zero=False, reverse=y_reverse),
            ),
            tooltip=tooltip,
        )
    )


def _axis_label(column: str) -> str:
    labels = {
        "BP_RP": "BP - RP",
        "G_BP": "G - BP",
        "G_RP": "G - RP",
        "J_H": "J - H",
        "J_K": "J - Ks",
        "H_K": "H - Ks",
        "W1_W2": "W1 - W2",
        "galactic_l": "Galactic longitude (deg)",
        "galactic_b": "Galactic latitude (deg)",
        "distance_kpc": "Distance (kpc)",
        "galactocentric_x_kpc": "Galactocentric X (kpc)",
        "galactocentric_y_kpc": "Galactocentric Y (kpc)",
    }
    if column in labels:
        return labels[column]
    if column in {"G", "BP", "RP", "J", "H", "Ks", "W1", "W2"}:
        return f"{column} (mag)"
    return column.replace("_", " ")


def _hidden_axis() -> alt.Axis:
    return alt.Axis(
        labels=False,
        ticks=False,
        domain=False,
        grid=False,
        title=None,
    )
