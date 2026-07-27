"""Threshold, binary, and ranking metrics for rare-object candidate recovery."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def select_threshold(
    y_true,
    y_score,
    *,
    beta: float = 2.0,
    min_precision: float = 0.5,
    min_accuracy: float = 0.0,
    min_balanced_accuracy: float = 0.0,
) -> dict[str, float | bool]:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype="float64")
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    rows = []
    for idx, threshold in enumerate(thresholds):
        y_pred = (y_score >= float(threshold)).astype("int8")
        p = float(precision[idx])
        r = float(recall[idx])
        score = _fbeta_from_precision_recall(p, r, beta=beta)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": p,
                "recall": r,
                "f_beta": score,
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            }
        )
    if not rows:
        return {
            "threshold": 0.5,
            "precision": 0.0,
            "recall": 0.0,
            "f_beta": 0.0,
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "precision_floor_met": False,
            "accuracy_floor_met": False,
            "balanced_accuracy_floor_met": False,
        }
    table = pd.DataFrame(rows)
    eligible = table[
        table["precision"].ge(float(min_precision))
        & table["accuracy"].ge(float(min_accuracy))
        & table["balanced_accuracy"].ge(float(min_balanced_accuracy))
    ]
    if not eligible.empty:
        selected = eligible.sort_values(["f_beta", "recall", "precision"], ascending=False).iloc[0]
    else:
        metric_eligible = table[
            table["accuracy"].ge(float(min_accuracy))
            & table["balanced_accuracy"].ge(float(min_balanced_accuracy))
        ]
        if not metric_eligible.empty:
            selected = metric_eligible.sort_values(["precision", "f_beta", "recall"], ascending=False).iloc[0]
        else:
            selected = table.sort_values(["precision", "accuracy", "balanced_accuracy", "f_beta"], ascending=False).iloc[0]
    return {
        "threshold": float(selected["threshold"]),
        "precision": float(selected["precision"]),
        "recall": float(selected["recall"]),
        "f_beta": float(selected["f_beta"]),
        "accuracy": float(selected["accuracy"]),
        "balanced_accuracy": float(selected["balanced_accuracy"]),
        "precision_floor_met": bool(selected["precision"] >= float(min_precision)),
        "accuracy_floor_met": bool(selected["accuracy"] >= float(min_accuracy)),
        "balanced_accuracy_floor_met": bool(selected["balanced_accuracy"] >= float(min_balanced_accuracy)),
    }


def compute_binary_metrics(y_true, y_score, *, threshold: float) -> dict[str, float | int]:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype="float64")
    y_pred = (y_score >= float(threshold)).astype("int8")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "average_precision": _safe_average_precision(y_true, y_score),
        "roc_auc": _safe_roc_auc(y_true, y_score),
        "precision_wr": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_wr": float(recall_score(y_true, y_pred, zero_division=0)),
        "f2_wr": float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": float(threshold),
    }


def compute_ranking_metrics(
    y_true,
    y_score,
    *,
    top_k: list[int],
    fpr_levels: list[float],
) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype="int8")
    y_score = np.asarray(y_score, dtype="float64")
    order = np.argsort(-y_score)
    positives = int(y_true.sum())
    negatives = int((y_true == 0).sum())
    metrics: dict[str, float | int] = {}
    for k in top_k:
        effective_k = min(int(k), len(y_true))
        top_idx = order[:effective_k]
        tp = int(y_true[top_idx].sum())
        metrics[f"precision_at_{k}"] = float(tp / effective_k) if effective_k else 0.0
        metrics[f"recall_at_{k}"] = float(tp / positives) if positives else 0.0
        metrics[f"wr_at_{k}"] = tp
        metrics[f"candidates_per_wr_at_{k}"] = (
            float(effective_k / tp) if tp else float("nan")
        )
    for fpr in fpr_levels:
        label = _fpr_label(fpr)
        allowed_fp = int(np.floor(float(fpr) * negatives))
        best_recall = 0.0
        best_threshold = np.inf
        for threshold in np.unique(y_score)[::-1]:
            pred = y_score >= threshold
            fp = int(((pred == 1) & (y_true == 0)).sum())
            if fp <= allowed_fp:
                tp = int(((pred == 1) & (y_true == 1)).sum())
                best_recall = float(tp / positives) if positives else 0.0
                best_threshold = float(threshold)
            else:
                break
        metrics[f"recall_at_fpr_{label}"] = best_recall
        metrics[f"threshold_at_fpr_{label}"] = best_threshold
    return metrics


def _fpr_label(value: float) -> str:
    return str(value).replace(".", "p")


def _fbeta_from_precision_recall(precision: float, recall: float, *, beta: float) -> float:
    beta2 = beta**2
    denominator = beta2 * precision + recall
    if denominator == 0:
        return 0.0
    return float((1 + beta2) * precision * recall / denominator)


def _safe_average_precision(y_true, y_score) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def _safe_roc_auc(y_true, y_score) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))
