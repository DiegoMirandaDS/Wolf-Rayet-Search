from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.base import clone

from wr_detector.config import load_yaml, resolve_path
from wr_detector.modeling.data import build_model_matrix, ensure_parent_dir, load_modeling_dataset
from wr_detector.modeling.estimators import build_model_pipeline, positive_class_weight
from wr_detector.modeling.evaluation import compute_binary_metrics, select_threshold
from wr_detector.modeling.splits import make_cv, make_global_holdout_mask


def run_model_benchmark(config_path: str | Path = "configs/models.yaml", *, verbose: bool = False) -> pd.DataFrame:
    config = load_yaml(config_path)
    random_state = int(config.get("random_state", 42))
    rows: list[dict[str, object]] = []
    total_experiments = len(config["dataset_variants"]) * len(config["feature_sets"]) * len(config["models"])
    experiment_index = 0
    started = perf_counter()
    if verbose:
        _log_progress(f"Starting model benchmark: {total_experiments} configured experiments.")
    for variant in config["dataset_variants"]:
        dataset = load_modeling_dataset(config, variant)
        if verbose:
            _log_progress(
                f"Loaded {variant}: rows={len(dataset)}, WR={int(dataset['target'].sum())}, "
                f"non_WR={int((dataset['target'] == 0).sum())}"
            )
        for feature_set_name, feature_columns in config["feature_sets"].items():
            x_all, y_all = build_model_matrix(dataset, list(feature_columns))
            aligned_source_ids = dataset.loc[x_all.index, "source_id"].reset_index(drop=True)
            aligned_holdout = make_global_holdout_mask(
                aligned_source_ids,
                holdout_fraction=float(config["holdout_fraction"]),
                random_state=random_state,
                target=y_all,
            )
            aligned_holdout.index = x_all.index
            x_train, y_train = x_all.loc[~aligned_holdout], y_all.loc[~aligned_holdout]
            x_holdout, y_holdout = x_all.loc[aligned_holdout], y_all.loc[aligned_holdout]
            if y_train.nunique() < 2 or y_holdout.nunique() < 2:
                experiment_index += len(config["models"])
                if verbose:
                    _log_progress(
                        f"Skipping {variant} / {feature_set_name}: train or holdout has a single class."
                    )
                continue
            for model_name, model_config in config["models"].items():
                experiment_index += 1
                if verbose:
                    elapsed = perf_counter() - started
                    avg = elapsed / max(experiment_index - 1, 1)
                    remaining = avg * (total_experiments - experiment_index + 1)
                    _log_progress(
                        f"[{experiment_index}/{total_experiments}] {variant} / {feature_set_name} / {model_name} "
                        f"(elapsed={_format_seconds(elapsed)}, eta={_format_seconds(remaining)})"
                    )
                try:
                    row = _benchmark_one_model(
                        variant=variant,
                        feature_set_name=feature_set_name,
                        model_name=model_name,
                        model_config=model_config,
                        x_train=x_train,
                        y_train=y_train,
                        x_holdout=x_holdout,
                        y_holdout=y_holdout,
                        config=config,
                        random_state=random_state,
                    )
                except ImportError:
                    if bool(model_config.get("optional", False)):
                        if verbose:
                            _log_progress(f"Skipping optional model {model_name}: dependency unavailable.")
                        continue
                    raise
                rows.append(row)
                if verbose:
                    _log_progress(
                        f"Completed {model_name}: holdout_f2={row['holdout_f2_wr']:.3f}, "
                        f"precision={row['holdout_precision_wr']:.3f}, recall={row['holdout_recall_wr']:.3f}"
                    )
    results = pd.DataFrame(rows).sort_values(
        ["holdout_f2_wr", "holdout_recall_wr", "holdout_precision_wr"],
        ascending=False,
        na_position="last",
    )
    output_path = ensure_parent_dir(config["outputs"]["benchmark_table"])
    results.to_csv(output_path, index=False)
    if verbose:
        _log_progress(f"Benchmark complete: {len(results)} completed experiments -> {output_path}")
    return results


def _benchmark_one_model(
    *,
    variant: str,
    feature_set_name: str,
    model_name: str,
    model_config: dict,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_holdout: pd.DataFrame,
    y_holdout: pd.Series,
    config: dict,
    random_state: int,
) -> dict[str, object]:
    cv_cfg = config["cv"]
    cv = make_cv(
        y_train,
        n_splits=int(cv_cfg["n_splits"]),
        n_repeats=int(cv_cfg["n_repeats"]),
        random_state=random_state,
    )
    positive_weight = positive_class_weight(y_train)
    pipeline = build_model_pipeline(model_config, random_state=random_state, positive_weight=positive_weight)
    oof_score = _repeated_oof_predict_proba(pipeline, x_train, y_train, cv=cv)
    threshold = select_threshold(
        y_train,
        oof_score,
        beta=2.0,
        min_precision=float(config["threshold"]["min_precision"]),
    )
    cv_metrics = compute_binary_metrics(y_train, oof_score, threshold=float(threshold["threshold"]))
    fitted = clone(pipeline).fit(x_train, y_train)
    train_score = fitted.predict_proba(x_train)[:, 1]
    holdout_score = fitted.predict_proba(x_holdout)[:, 1]
    train_metrics = compute_binary_metrics(y_train, train_score, threshold=float(threshold["threshold"]))
    holdout_metrics = compute_binary_metrics(y_holdout, holdout_score, threshold=float(threshold["threshold"]))
    row: dict[str, object] = {
        "dataset_variant": variant,
        "feature_set": feature_set_name,
        "model": model_name,
        "estimator": model_config["estimator"],
        "sampler": model_config.get("sampler", "none"),
        "n_train": int(len(y_train)),
        "n_holdout": int(len(y_holdout)),
        "wr_train": int(y_train.sum()),
        "wr_holdout": int(y_holdout.sum()),
        "selected_threshold": float(threshold["threshold"]),
        "threshold_precision_floor_met": bool(threshold["precision_floor_met"]),
        "cv_train_gap_f2": float(train_metrics["f2_wr"] - cv_metrics["f2_wr"]),
        "holdout_cv_gap_f2": float(holdout_metrics["f2_wr"] - cv_metrics["f2_wr"]),
    }
    row.update({f"cv_{key}": value for key, value in cv_metrics.items()})
    row.update({f"train_{key}": value for key, value in train_metrics.items()})
    row.update({f"holdout_{key}": value for key, value in holdout_metrics.items()})
    return row


def _repeated_oof_predict_proba(pipeline, x: pd.DataFrame, y: pd.Series, *, cv) -> np.ndarray:
    score_sum = np.zeros(len(y), dtype="float64")
    score_count = np.zeros(len(y), dtype="int64")
    for train_idx, validation_idx in cv.split(x, y):
        fitted = clone(pipeline).fit(x.iloc[train_idx], y.iloc[train_idx])
        score_sum[validation_idx] += fitted.predict_proba(x.iloc[validation_idx])[:, 1]
        score_count[validation_idx] += 1
    if np.any(score_count == 0):
        raise ValueError("Every row must receive at least one out-of-fold prediction.")
    return score_sum / score_count


def _format_seconds(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{sec:02d}s"
    if minutes:
        return f"{minutes}m{sec:02d}s"
    return f"{sec}s"


def _log_progress(message: str) -> None:
    print(message, flush=True)


def load_benchmark_results(config_path: str | Path = "configs/models.yaml") -> pd.DataFrame:
    config = load_yaml(config_path)
    path = resolve_path(config["outputs"]["benchmark_table"])
    return pd.read_csv(path)
