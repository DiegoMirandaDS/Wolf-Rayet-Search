"""Model complexity and permutation-importance diagnostics for trained estimators."""

from __future__ import annotations

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_predict

from wr_detector.modeling.evaluation import compute_binary_metrics
from wr_detector.modeling.splits import make_cv


def evaluate_random_forest_complexity(
    x: pd.DataFrame,
    y: pd.Series,
    *,
    random_state: int = 42,
    n_estimators_grid: list[int] | None = None,
    max_depth_grid: list[int | None] | None = None,
    min_samples_leaf_grid: list[int] | None = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    n_estimators_grid = n_estimators_grid or [50, 100, 200, 300]
    max_depth_grid = max_depth_grid or [3, 5, 8, None]
    min_samples_leaf_grid = min_samples_leaf_grid or [1, 5, 10]
    cv = make_cv(y, n_splits=5, n_repeats=1, random_state=random_state)
    rows = []
    for n_estimators in n_estimators_grid:
        for max_depth in max_depth_grid:
            for min_samples_leaf in min_samples_leaf_grid:
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=random_state,
                )
                oof = cross_val_predict(model, x, y, cv=cv, method="predict_proba")[:, 1]
                fitted = clone(model).fit(x, y)
                train = fitted.predict_proba(x)[:, 1]
                cv_metrics = compute_binary_metrics(y, oof, threshold=threshold)
                train_metrics = compute_binary_metrics(y, train, threshold=threshold)
                rows.append(
                    {
                        "n_estimators": n_estimators,
                        "max_depth": max_depth if max_depth is not None else "none",
                        "min_samples_leaf": min_samples_leaf,
                        "cv_f2_wr": cv_metrics["f2_wr"],
                        "train_f2_wr": train_metrics["f2_wr"],
                        "train_cv_gap_f2": train_metrics["f2_wr"] - cv_metrics["f2_wr"],
                        "cv_average_precision": cv_metrics["average_precision"],
                    }
                )
    return pd.DataFrame(rows)


def evaluate_boosting_iterations(
    x: pd.DataFrame,
    y: pd.Series,
    *,
    random_state: int = 42,
    max_iter_grid: list[int] | None = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    max_iter_grid = max_iter_grid or [25, 50, 100, 200, 300]
    cv = make_cv(y, n_splits=5, n_repeats=1, random_state=random_state)
    rows = []
    for max_iter in max_iter_grid:
        model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=max_iter,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=0.1,
            early_stopping=False,
            class_weight="balanced",
            random_state=random_state,
        )
        oof = cross_val_predict(model, x, y, cv=cv, method="predict_proba")[:, 1]
        fitted = clone(model).fit(x, y)
        train = fitted.predict_proba(x)[:, 1]
        cv_metrics = compute_binary_metrics(y, oof, threshold=threshold)
        train_metrics = compute_binary_metrics(y, train, threshold=threshold)
        rows.append(
            {
                "max_iter": max_iter,
                "cv_f2_wr": cv_metrics["f2_wr"],
                "train_f2_wr": train_metrics["f2_wr"],
                "train_cv_gap_f2": train_metrics["f2_wr"] - cv_metrics["f2_wr"],
                "cv_average_precision": cv_metrics["average_precision"],
            }
        )
    return pd.DataFrame(rows)


def compute_permutation_importance_table(model, x: pd.DataFrame, y: pd.Series, *, scoring: str = "average_precision") -> pd.DataFrame:
    result = permutation_importance(model, x, y, scoring=scoring, n_repeats=10, random_state=42, n_jobs=-1)
    return (
        pd.DataFrame(
            {
                "feature": x.columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
