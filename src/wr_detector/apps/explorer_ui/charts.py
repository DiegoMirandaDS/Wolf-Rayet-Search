"""Altair chart builders for the Model Explorer.

All charts use ``width="container"`` and bounded heights so they scale
with the page instead of overflowing it, and they inherit the Streamlit
theme (transparent backgrounds, theme fonts and grid colors).
"""

from __future__ import annotations

import altair as alt
import pandas as pd

CLASS_COLORS = alt.Scale(domain=["WR", "negative"], range=["#f28e2b", "#4e79a7"])
SPLIT_COLORS = alt.Scale(domain=["train", "cv", "holdout"], range=["#59a14f", "#edc948", "#e15759"])
CATEGORY_SCHEME = "tableau10"


def metric_bar(
    df: pd.DataFrame,
    *,
    value_col: str,
    value_title: str,
    label_col: str = "short_label",
    percent: bool = False,
) -> alt.Chart | None:
    columns = [column for column in {label_col, value_col, "model", "selection_status"} if column in df.columns]
    data = df[columns].dropna(subset=[value_col]).copy()
    if data.empty:
        return None
    height = min(560, max(160, 26 * len(data) + 30))
    axis_format = ".0%" if percent else None
    tooltip = [
        alt.Tooltip(f"{label_col}:N", title="model"),
        alt.Tooltip(f"{value_col}:Q", title=value_title, format=".1%" if percent else ".4f"),
    ]
    if "selection_status" in data.columns:
        tooltip.append(alt.Tooltip("selection_status:N", title="status"))
    return (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=2)
        .encode(
            y=alt.Y(f"{label_col}:N", sort="-x", title=None, axis=alt.Axis(labelLimit=320)),
            x=alt.X(f"{value_col}:Q", title=value_title, axis=alt.Axis(format=axis_format)),
            color=alt.Color("model:N", scale=alt.Scale(scheme=CATEGORY_SCHEME), legend=alt.Legend(title=None, orient="bottom")),
            tooltip=tooltip,
        )
        .properties(width="container", height=height)
    )


def recovery_lines(ranked: pd.DataFrame, *, ks: list[int]) -> alt.Chart | None:
    rows = []
    for _, row in ranked.iterrows():
        for k in ks:
            pct = row.get(f"holdout_wr_at_{k}_pct")
            if pd.isna(pct):
                continue
            rows.append(
                {
                    "short_label": row["short_label"],
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
            color=alt.Color("short_label:N", scale=alt.Scale(scheme=CATEGORY_SCHEME), legend=alt.Legend(title=None, orient="bottom", columns=2, labelLimit=320)),
            tooltip=[
                alt.Tooltip("short_label:N", title="model"),
                alt.Tooltip("budget:Q", title="budget"),
                alt.Tooltip("recovered:Q", title="WR recovered"),
                alt.Tooltip("recovered_pct:Q", title="recovered", format=".1%"),
            ],
        )
        .properties(width="container", height=340)
    )


def dataset_heatmap(df: pd.DataFrame, *, value_col: str, value_title: str, percent: bool = False) -> alt.Chart | None:
    data = df.dropna(subset=[value_col]).copy()
    if data.empty:
        return None
    data["column_label"] = data["model"].astype(str) + " / " + data["sampler"].astype(str)
    aggregated = (
        data.groupby(["dataset_variant", "column_label"], as_index=False)[value_col].max()
    )
    height = max(180, 34 * aggregated["dataset_variant"].nunique() + 60)
    value_format = ".0%" if percent else ".3f"
    base = alt.Chart(aggregated).encode(
        x=alt.X("column_label:N", title=None, axis=alt.Axis(labelAngle=-30)),
        y=alt.Y("dataset_variant:N", title=None),
    )
    rect = base.mark_rect().encode(
        color=alt.Color(f"{value_col}:Q", scale=alt.Scale(scheme="viridis"), legend=alt.Legend(title=value_title, format=".0%" if percent else None)),
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
    required = {"short_label", "cv_train_gap_f2", "holdout_cv_gap_f2"}
    if not required.issubset(ranked.columns):
        return None
    data = ranked.dropna(subset=["cv_train_gap_f2", "holdout_cv_gap_f2"]).copy()
    if data.empty:
        return None
    long = data.melt(
        id_vars=["short_label"],
        value_vars=["cv_train_gap_f2", "holdout_cv_gap_f2"],
        var_name="gap",
        value_name="value",
    )
    long["gap"] = long["gap"].map({"cv_train_gap_f2": "train - CV", "holdout_cv_gap_f2": "holdout - CV"})
    bars = (
        alt.Chart(long)
        .mark_bar()
        .encode(
            x=alt.X("short_label:N", title=None, axis=alt.Axis(labelAngle=-30, labelLimit=200)),
            xOffset=alt.XOffset("gap:N"),
            y=alt.Y("value:Q", title="F2 gap"),
            color=alt.Color("gap:N", legend=alt.Legend(title=None, orient="bottom")),
            tooltip=[
                alt.Tooltip("short_label:N", title="model"),
                alt.Tooltip("gap:N"),
                alt.Tooltip("value:Q", format=".3f"),
            ],
        )
    )
    threshold = alt.Chart(pd.DataFrame({"y": [0.15]})).mark_rule(strokeDash=[5, 4], color="#e15759").encode(y="y:Q")
    return (bars + threshold).properties(width="container", height=300)


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


def precision_recall_chart(curve: pd.DataFrame, *, threshold: float | None = None) -> alt.Chart | None:
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
    operating = _operating_point(curve, threshold)
    if operating is not None:
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


def roc_chart(curve: pd.DataFrame, *, threshold: float | None = None) -> alt.Chart | None:
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
    operating = _operating_point(curve, threshold)
    if operating is not None:
        layers.append(
            alt.Chart(operating)
            .mark_point(size=140, filled=True, color="#e15759")
            .encode(x="fpr:Q", y="tpr:Q", tooltip=[alt.Tooltip("score:Q", title="threshold", format=".4f")])
        )
    return alt.layer(*layers).properties(width="container", height=320)


def _operating_point(curve: pd.DataFrame, threshold: float | None) -> pd.DataFrame | None:
    if threshold is None or curve.empty or "score" not in curve.columns:
        return None
    above = curve[curve["score"] >= threshold]
    return above.tail(1) if not above.empty else None


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


def color_magnitude(cases: pd.DataFrame, *, selected_source_id: int | None = None, x: str = "BP_RP", y: str = "G") -> alt.Chart | None:
    columns = [
        column
        for column in {x, y, "target", "object_name", "rank", "score", "source_id", "simbad_main_type", "spectral_type"}
        if column in cases.columns
    ]
    data = cases[columns].dropna(subset=[x, y]).copy()
    if data.empty:
        return None
    data["class"] = data["target"].map({1: "WR", 0: "negative"})
    tooltip = [
        alt.Tooltip("object_name:N", title="object"),
        alt.Tooltip("rank:Q"),
        alt.Tooltip("score:Q", format=".4f"),
        alt.Tooltip(f"{x}:Q", format=".3f"),
        alt.Tooltip(f"{y}:Q", format=".3f"),
    ]
    if "simbad_main_type" in data.columns:
        tooltip.append(alt.Tooltip("simbad_main_type:N", title="SIMBAD type"))
    if "spectral_type" in data.columns:
        tooltip.append(alt.Tooltip("spectral_type:N", title="spectral type"))
    points = (
        alt.Chart(data)
        .mark_circle()
        .encode(
            x=alt.X(f"{x}:Q", title=x, scale=alt.Scale(zero=False)),
            y=alt.Y(f"{y}:Q", title=f"{y} (mag)", scale=alt.Scale(zero=False, reverse=True)),
            color=alt.Color("class:N", scale=CLASS_COLORS, legend=alt.Legend(title=None, orient="bottom")),
            size=alt.condition("datum.class == 'WR'", alt.value(60), alt.value(22)),
            opacity=alt.condition("datum.class == 'WR'", alt.value(0.9), alt.value(0.45)),
            tooltip=tooltip,
        )
    )
    layers = [points]
    if selected_source_id is not None:
        selected = data[data["source_id"].eq(selected_source_id)]
        if not selected.empty:
            layers.append(
                alt.Chart(selected)
                .mark_point(shape="diamond", size=420, filled=False, color="#e15759", strokeWidth=3)
                .encode(x=alt.X(f"{x}:Q"), y=alt.Y(f"{y}:Q", scale=alt.Scale(zero=False, reverse=True)), tooltip=tooltip)
            )
    return alt.layer(*layers).interactive().properties(width="container", height=420)
