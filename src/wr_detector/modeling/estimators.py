from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def build_model_pipeline(
    model_config: Mapping[str, object],
    *,
    random_state: int,
    positive_weight: float = 1.0,
    sampler_config: Mapping[str, object] | None = None,
) -> Pipeline:
    estimator_name = str(model_config["estimator"])
    steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if estimator_name == "logistic_regression":
        steps.append(("scaler", StandardScaler()))
    sampler = build_sampler(sampler_config or model_config.get("sampler", "none"), random_state=random_state)
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("estimator", _make_estimator(estimator_name, random_state=random_state, positive_weight=positive_weight)))
    return Pipeline(steps)


def build_sampler(config: Mapping[str, object] | str | None, *, random_state: int):
    if config is None:
        return None
    if isinstance(config, str):
        if config in {"none", "", "null"}:
            return None
        name = config
        k_neighbors = 3
    else:
        name = str(config.get("type", "none"))
        k_neighbors = int(config.get("k_neighbors", 3))
    if name in {"none", "", "null"}:
        return None
    if name == "smote":
        return SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    if name == "smote_enn":
        return SMOTEENN(random_state=random_state, smote=SMOTE(random_state=random_state, k_neighbors=k_neighbors))
    if name == "smote_tomek":
        return SMOTETomek(random_state=random_state, smote=SMOTE(random_state=random_state, k_neighbors=k_neighbors))
    if name == "random_over_sampler":
        return RandomOverSampler(random_state=random_state)
    raise ValueError(f"Unsupported sampler: {name}")


def _make_estimator(name: str, *, random_state: int, positive_weight: float):
    if name == "logistic_regression":
        return LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            random_state=random_state,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=0.1,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            class_weight={0: 1.0, 1: float(positive_weight)},
            random_state=random_state,
        )
    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError("xgboost is optional. Install the modeling extra to enable XGBClassifier.") from exc
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            reg_alpha=0.0,
            scale_pos_weight=float(positive_weight),
            n_jobs=-1,
            random_state=random_state,
        )
    raise ValueError(f"Unsupported estimator: {name}")


def positive_class_weight(y) -> float:
    counts = np.bincount(np.asarray(y, dtype="int64"), minlength=2)
    return float(counts[0] / counts[1]) if counts[1] else 1.0
