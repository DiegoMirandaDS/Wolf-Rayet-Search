"""Statistical assumption checks for robust color-locus regression audits."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures

from wr_detector.features import robust_sigma_mad


def residual_assumption_summary(df: pd.DataFrame, residual_col: str, fitted_col: str, x_col: str) -> dict[str, float]:
    residual = df[residual_col].dropna()
    aligned = df[[residual_col, fitted_col, x_col]].dropna()
    return {
        "n": int(len(aligned)),
        "residual_mean": float(residual.mean()),
        "residual_median": float(residual.median()),
        "residual_sigma_mad": robust_sigma_mad(residual),
        "abs_residual_fitted_spearman": float(aligned[residual_col].abs().corr(aligned[fitted_col], method="spearman")),
        "abs_residual_x_spearman": float(aligned[residual_col].abs().corr(aligned[x_col], method="spearman")),
        "residual_fitted_spearman": float(aligned[residual_col].corr(aligned[fitted_col], method="spearman")),
        "residual_x_spearman": float(aligned[residual_col].corr(aligned[x_col], method="spearman")),
    }


def compare_linear_quadratic_fit(x: pd.Series, y: pd.Series) -> dict[str, float]:
    valid = x.notna() & y.notna()
    x_values = x.loc[valid].to_numpy().reshape(-1, 1)
    y_values = y.loc[valid].to_numpy()
    linear = LinearRegression().fit(x_values, y_values)
    quadratic_x = PolynomialFeatures(degree=2, include_bias=False).fit_transform(x_values)
    quadratic = LinearRegression().fit(quadratic_x, y_values)
    linear_pred = linear.predict(x_values)
    quadratic_pred = quadratic.predict(quadratic_x)
    return {
        "n": int(valid.sum()),
        "linear_r2": float(r2_score(y_values, linear_pred)),
        "quadratic_r2": float(r2_score(y_values, quadratic_pred)),
        "quadratic_r2_gain": float(r2_score(y_values, quadratic_pred) - r2_score(y_values, linear_pred)),
    }


def bootstrap_huber_coefficients(
    x: pd.Series,
    y: pd.Series,
    *,
    n_bootstrap: int = 200,
    random_state: int = 42,
) -> pd.DataFrame:
    valid = x.notna() & y.notna()
    x_values = x.loc[valid].to_numpy().reshape(-1, 1)
    y_values = y.loc[valid].to_numpy()
    rng = np.random.default_rng(random_state)
    rows = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(y_values), len(y_values))
        model = HuberRegressor().fit(x_values[indices], y_values[indices])
        rows.append({"slope": float(model.coef_[0]), "intercept": float(model.intercept_)})
    return pd.DataFrame(rows)


def fold_locus_stability(
    x: pd.Series,
    y: pd.Series,
    *,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    valid = x.notna() & y.notna()
    x_values = x.loc[valid].to_numpy().reshape(-1, 1)
    y_values = y.loc[valid].to_numpy()
    splitter = KFold(n_splits=min(n_splits, len(y_values)), shuffle=True, random_state=random_state)
    rows = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(x_values), start=1):
        model = HuberRegressor().fit(x_values[train_idx], y_values[train_idx])
        pred_train = model.predict(x_values[train_idx])
        pred_test = model.predict(x_values[test_idx])
        rows.append(
            {
                "fold": fold,
                "slope": float(model.coef_[0]),
                "intercept": float(model.intercept_),
                "train_r2": float(r2_score(y_values[train_idx], pred_train)),
                "validation_r2": float(r2_score(y_values[test_idx], pred_test)),
                "train_sigma_mad": robust_sigma_mad(y_values[train_idx] - pred_train),
                "validation_sigma_mad": robust_sigma_mad(y_values[test_idx] - pred_test),
            }
        )
    return pd.DataFrame(rows)
