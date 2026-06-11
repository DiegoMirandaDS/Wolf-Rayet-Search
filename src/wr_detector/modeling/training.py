from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    fbeta_score,
    precision_score,
)

from wr_detector.config import load_yaml, resolve_path
from wr_detector.modeling.benchmark import _format_seconds, _repeated_oof_predict_proba
from wr_detector.modeling.data import build_model_matrix, ensure_parent_dir, load_modeling_dataset
from wr_detector.modeling.estimators import build_model_pipeline, positive_class_weight
from wr_detector.modeling.evaluation import compute_binary_metrics, compute_ranking_metrics, select_threshold
from wr_detector.modeling.history import cleanup_unreferenced_model_artifacts, make_project_relative, make_training_run_id, sync_training_history_frame
from wr_detector.modeling.negative_reduction import (
    configured_negative_ratios,
    negative_ratio_label,
    reduce_negative_dataset,
    reduced_dataset_path,
)
from wr_detector.modeling.splits import make_cv


def train_models(
    config_path: str | Path = "configs/models.yaml",
    *,
    variants: list[str] | None = None,
    models: list[str] | None = None,
    samplers: list[str] | None = None,
    feature_sets: list[str] | None = None,
    negative_ratios: list[int | str] | None = None,
    n_iter: int | None = None,
    run_id: str | None = None,
    resume_run: bool = False,
    reduce_if_missing: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    config = load_yaml(config_path)
    run_id = run_id or make_training_run_id(config_path)
    selected_variants = variants or list(config["dataset_variants"])
    selected_models = models or list(config["models"])
    selected_samplers = samplers or list(config["samplers"])
    selected_feature_sets = feature_sets or list(config["feature_sets"])
    selected_ratios = negative_ratios or configured_negative_ratios(config)
    rows: list[dict[str, object]] = []
    run_results_path = run_training_results_path(config, run_id)
    completed = load_completed_run_results(run_results_path) if resume_run else pd.DataFrame()
    completed_keys = completed_configuration_keys(completed)
    total = len(selected_variants) * len(selected_ratios) * len(selected_feature_sets) * len(selected_models) * len(selected_samplers)
    current = 0
    started = perf_counter()
    for variant in selected_variants:
        for negative_ratio in selected_ratios:
            dataset = load_reduced_or_source_dataset(config, variant, negative_ratio=negative_ratio, reduce_if_missing=reduce_if_missing)
            train_df = dataset[dataset["modeling_split"] == "train"].copy()
            calibration_df = dataset[dataset["modeling_split"] == "threshold_calibration"].copy()
            holdout_df = dataset[dataset["modeling_split"] == "holdout"].copy()
            for feature_set_name in selected_feature_sets:
                feature_columns = list(config["feature_sets"][feature_set_name])
                x_train, y_train = build_model_matrix(train_df, feature_columns)
                x_calibration, y_calibration = (
                    build_model_matrix(calibration_df, feature_columns) if not calibration_df.empty else (pd.DataFrame(), pd.Series(dtype="int8"))
                )
                x_holdout, y_holdout = build_model_matrix(holdout_df, feature_columns)
                train_identity = make_prediction_identity(train_df, x_train.index)
                holdout_identity = make_prediction_identity(holdout_df, x_holdout.index)
                for model_name in selected_models:
                    model_config = config["models"][model_name]
                    for sampler_name in selected_samplers:
                        current += 1
                        key = configuration_key(
                            variant=variant,
                            negative_ratio=negative_ratio,
                            feature_set_name=feature_set_name,
                            model_name=model_name,
                            sampler_name=sampler_name,
                        )
                        if key in completed_keys:
                            if verbose:
                                print(f"[{current}/{total}] Skipping completed {key}", flush=True)
                            continue
                        if verbose:
                            elapsed = perf_counter() - started
                            avg = elapsed / max(current - 1, 1)
                            eta = avg * (total - current + 1)
                            print(
                                f"[{current}/{total}] {variant} / neg={negative_ratio_label(negative_ratio)} / "
                                f"{feature_set_name} / {model_name} / {sampler_name} "
                                f"(run_id={run_id}, elapsed={_format_seconds(elapsed)}, eta={_format_seconds(eta)})",
                                flush=True,
                            )
                        try:
                            row = train_one_configuration(
                                config=config,
                                run_id=run_id,
                                variant=variant,
                                negative_ratio=negative_ratio,
                                feature_set_name=feature_set_name,
                                model_name=model_name,
                                model_config=model_config,
                                sampler_name=sampler_name,
                                sampler_config=config["samplers"][sampler_name],
                                x_train=x_train,
                                y_train=y_train,
                                x_calibration=x_calibration,
                                y_calibration=y_calibration,
                                x_holdout=x_holdout,
                                y_holdout=y_holdout,
                                train_identity=train_identity,
                                holdout_identity=holdout_identity,
                                n_iter=n_iter,
                            )
                        except ImportError:
                            if bool(model_config.get("optional", False)):
                                if verbose:
                                    print(f"Skipping optional model {model_name}: dependency unavailable.", flush=True)
                                continue
                            raise
                        rows.append(row)
                        append_run_result(run_results_path, row)
                        completed_keys.add(key)
                        if verbose:
                            print(
                                f"Completed: holdout_f2={row['holdout_f2_wr']:.3f}, "
                                f"precision={row['holdout_precision_wr']:.3f}, recall={row['holdout_recall_wr']:.3f}, "
                                f"flag={row['selection_status']}",
                                flush=True,
                            )
    current_results = pd.concat([completed, pd.DataFrame(rows)], ignore_index=True, sort=False) if not completed.empty else pd.DataFrame(rows)
    results = current_results.copy()
    out_path = ensure_parent_dir(config["outputs"]["training_results"])
    if out_path.exists() and not results.empty:
        previous = pd.read_csv(out_path)
        results = merge_training_results(previous, results)
    results = results.sort_values(
        ["selection_status", "holdout_f2_wr", "holdout_recall_wr", "holdout_precision_wr"],
        ascending=[True, False, False, False],
        na_position="last",
    )
    results.to_csv(out_path, index=False)
    maybe_sync_training_history(
        config_path=config_path,
        config=config,
        run_id=run_id,
        current_results=current_results,
        replace_run=resume_run,
        verbose=verbose,
    )
    return results


def maybe_sync_training_history(
    *,
    config_path: str | Path,
    config: dict,
    run_id: str,
    current_results: pd.DataFrame,
    replace_run: bool,
    verbose: bool,
) -> None:
    outputs = config.get("outputs", {})
    if current_results.empty or not bool(outputs.get("auto_sync_training_history", False)):
        return
    run_name = outputs.get("training_run_name") or "train_models"
    synced = sync_training_history_frame(
        config_path,
        current_results,
        run_id=run_id,
        run_name=str(run_name),
        notes=outputs.get("training_run_notes"),
        replace_run=replace_run,
    )
    if verbose:
        print(
            f"Synced training history: run_id={synced['run_id']} rows={synced['rows']} "
            f"prediction_rows={synced['prediction_rows']}",
            flush=True,
        )
    if bool(outputs.get("compact_db_backed_sidecars_after_sync", False)):
        cleanup = cleanup_unreferenced_model_artifacts(
            config_path,
            apply=True,
            remove_db_backed_sidecars=True,
        )
        removed = cleanup[~cleanup["keep"]]
        if verbose:
            print(f"Compacted DB-backed sidecars: removed={len(removed)}", flush=True)


def merge_training_results(previous: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    previous = previous.copy()
    current = current.copy()
    if "negative_ratio_label" in previous.columns or "negative_ratio_label" in current.columns:
        if "negative_ratio_label" not in previous.columns:
            previous["negative_ratio_label"] = "10x"
        if "negative_ratio_label" not in current.columns:
            current["negative_ratio_label"] = "10x"
    if "run_id" in previous.columns or "run_id" in current.columns:
        if "run_id" not in previous.columns:
            previous["run_id"] = "legacy_csv"
        if "run_id" not in current.columns:
            current["run_id"] = "legacy_csv"
    key = ["dataset_variant", "feature_set", "model", "sampler"]
    if "negative_ratio_label" in previous.columns or "negative_ratio_label" in current.columns:
        key.insert(1, "negative_ratio_label")
    if "run_id" in previous.columns or "run_id" in current.columns:
        key.insert(0, "run_id")
    if previous.empty:
        return current
    previous_keyed = previous.set_index(key, drop=False)
    current_keyed = current.set_index(key, drop=False)
    previous_keyed = previous_keyed.drop(index=current_keyed.index, errors="ignore")
    return pd.concat([previous_keyed.reset_index(drop=True), current_keyed.reset_index(drop=True)], ignore_index=True, sort=False)


def train_one_configuration(
    *,
    config: dict,
    run_id: str,
    variant: str,
    negative_ratio: int | str,
    feature_set_name: str,
    model_name: str,
    model_config: dict,
    sampler_name: str,
    sampler_config: dict,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_calibration: pd.DataFrame,
    y_calibration: pd.Series,
    x_holdout: pd.DataFrame,
    y_holdout: pd.Series,
    n_iter: int | None,
    train_identity: pd.DataFrame | None = None,
    holdout_identity: pd.DataFrame | None = None,
) -> dict[str, object]:
    random_state = int(config.get("random_state", 42))
    cv_cfg = config["cv"]
    cv = make_cv(
        y_train,
        n_splits=int(cv_cfg["n_splits"]),
        n_repeats=int(cv_cfg.get("n_repeats", 1)),
        random_state=random_state,
    )
    sampler_type = str(sampler_config.get("type", "none"))
    weight = 1.0 if sampler_type != "none" else positive_class_weight(y_train)
    pipeline = build_model_pipeline(
        model_config,
        random_state=random_state,
        positive_weight=weight,
        sampler_config=sampler_config,
    )
    search_space = build_search_space(model_config.get("search_space", {}))
    effective_n_iter = int(n_iter or model_config.get("bayes_iter", config["bayes_search"]["n_iter"]))
    search = BayesSearchCV(
        estimator=pipeline,
        search_spaces=search_space,
        n_iter=effective_n_iter,
        scoring=make_bayes_scorer(config),
        cv=cv,
        n_jobs=int(config["bayes_search"].get("n_jobs", 1)),
        random_state=random_state,
        refit=True,
        error_score="raise",
    )
    search.fit(x_train, y_train)
    best_pipeline = clone(search.best_estimator_)
    oof_score = _repeated_oof_predict_proba(best_pipeline, x_train, y_train, cv=cv)
    fitted = clone(search.best_estimator_).fit(x_train, y_train)
    threshold_y, threshold_score = make_threshold_selection_scores(
        fitted=fitted,
        y_train=y_train,
        oof_score=oof_score,
        x_calibration=x_calibration,
        y_calibration=y_calibration,
    )
    threshold = select_threshold(
        threshold_y,
        threshold_score,
        beta=float(config["selection"].get("beta", 2.0)),
        min_precision=float(config["selection"]["min_precision"]),
        min_accuracy=0.0,
        min_balanced_accuracy=float(config["selection"].get("min_balanced_accuracy", 0.0)),
    )
    cv_metrics = compute_binary_metrics(y_train, oof_score, threshold=float(threshold["threshold"]))
    train_score = fitted.predict_proba(x_train)[:, 1]
    holdout_score = fitted.predict_proba(x_holdout)[:, 1]
    train_metrics = compute_binary_metrics(y_train, train_score, threshold=float(threshold["threshold"]))
    holdout_metrics = compute_binary_metrics(y_holdout, holdout_score, threshold=float(threshold["threshold"]))
    holdout_ranking = compute_ranking_metrics(
        y_holdout,
        holdout_score,
        top_k=list(config.get("ranking", {}).get("top_k", [50, 100, 500, 1000])),
        fpr_levels=list(config.get("ranking", {}).get("fpr_levels", [0.001, 0.005, 0.01])),
    )
    row = build_training_row(
        config=config,
        run_id=run_id,
        variant=variant,
        negative_ratio=negative_ratio,
        feature_set_name=feature_set_name,
        model_name=model_name,
        sampler_name=sampler_name,
        x_train=x_train,
        y_train=y_train,
        x_holdout=x_holdout,
        y_holdout=y_holdout,
        search=search,
        threshold=threshold,
        cv_metrics=cv_metrics,
        train_metrics=train_metrics,
        holdout_metrics=holdout_metrics,
        holdout_ranking=holdout_ranking,
    )
    model_path = save_trained_model(config, fitted, row)
    row["model_path"] = make_project_relative(model_path)
    artifact_paths = save_validation_artifacts(
        config=config,
        model=fitted,
        row=row,
        x_train=x_train,
        y_train=y_train,
        train_score=train_score,
        x_holdout=x_holdout,
        y_holdout=y_holdout,
        holdout_score=holdout_score,
        oof_score=oof_score,
        train_identity=train_identity,
        holdout_identity=holdout_identity,
        threshold=float(threshold["threshold"]),
    )
    row.update(artifact_paths)
    save_model_metadata(model_path.with_suffix(".json"), row, search.best_params_)
    return row


def build_search_space(config: dict) -> dict[str, object]:
    space: dict[str, object] = {}
    for name, spec in config.items():
        kind = spec["type"]
        if kind == "integer":
            space[name] = Integer(int(spec["low"]), int(spec["high"]))
        elif kind == "real":
            space[name] = Real(float(spec["low"]), float(spec["high"]), prior=spec.get("prior", "uniform"))
        elif kind == "categorical":
            space[name] = Categorical(list(spec["values"]))
        else:
            raise ValueError(f"Unsupported Bayes search dimension type: {kind}")
    return space


def make_threshold_selection_scores(
    *,
    fitted,
    y_train: pd.Series,
    oof_score,
    x_calibration: pd.DataFrame,
    y_calibration: pd.Series,
) -> tuple[pd.Series, np.ndarray]:
    positive_mask = y_train.astype(int).eq(1)
    positive_y = y_train.loc[positive_mask]
    positive_scores = np.asarray(oof_score, dtype="float64")[positive_mask.to_numpy()]
    if x_calibration.empty:
        return y_train, np.asarray(oof_score, dtype="float64")
    calibration_scores = fitted.predict_proba(x_calibration)[:, 1]
    threshold_y = pd.concat([positive_y.reset_index(drop=True), y_calibration.reset_index(drop=True)], ignore_index=True)
    threshold_score = np.concatenate([positive_scores, calibration_scores])
    return threshold_y, threshold_score


def make_constrained_fbeta_scorer(config: dict):
    selection = config["selection"]
    beta = float(selection.get("beta", 2.0))
    min_precision = float(selection.get("min_precision", 0.0))
    min_balanced_accuracy = float(selection.get("min_balanced_accuracy", 0.0))

    def scorer(estimator, x, y_true) -> float:
        y_pred = estimator.predict(x)
        f_beta = float(fbeta_score(y_true, y_pred, beta=beta, zero_division=0))
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        balanced = float(balanced_accuracy_score(y_true, y_pred))
        penalty = 1.0
        if precision < min_precision:
            penalty *= max(precision / min_precision, 0.05) if min_precision else 1.0
        if balanced < min_balanced_accuracy:
            penalty *= max(balanced / min_balanced_accuracy, 0.05) if min_balanced_accuracy else 1.0
        return f_beta * penalty

    return scorer


def make_bayes_scorer(config: dict):
    scoring = str(config.get("bayes_search", {}).get("scoring", "average_precision"))
    if scoring == "average_precision":
        def scorer(estimator, x, y_true) -> float:
            return float(average_precision_score(y_true, estimator.predict_proba(x)[:, 1]))

        return scorer
    if scoring in {"constrained_f2", "constrained_f_beta", "constrained_fbeta"}:
        return make_constrained_fbeta_scorer(config)
    if scoring == "f2":
        def scorer(estimator, x, y_true) -> float:
            return float(fbeta_score(y_true, estimator.predict(x), beta=2.0, zero_division=0))

        return scorer
    raise ValueError(f"Unsupported BayesSearchCV scoring: {scoring}")


def load_reduced_or_source_dataset(
    config: dict,
    variant: str,
    *,
    negative_ratio: int | str,
    reduce_if_missing: bool,
) -> pd.DataFrame:
    path = reduced_dataset_path(config, variant, negative_ratio=negative_ratio)
    if path.exists():
        return pd.read_parquet(path)
    if not reduce_if_missing:
        raise FileNotFoundError(f"Reduced modeling dataset not found: {path}")
    reduced, _, _, _ = reduce_negative_dataset(config, variant, negative_ratio=negative_ratio)
    path.parent.mkdir(parents=True, exist_ok=True)
    reduced.to_parquet(path, index=False)
    return reduced


def build_training_row(
    *,
    config: dict,
    run_id: str,
    variant: str,
    negative_ratio: int | str,
    feature_set_name: str,
    model_name: str,
    sampler_name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_holdout: pd.DataFrame,
    y_holdout: pd.Series,
    search: BayesSearchCV,
    threshold: dict[str, object],
    cv_metrics: dict[str, object],
    train_metrics: dict[str, object],
    holdout_metrics: dict[str, object],
    holdout_ranking: dict[str, object],
) -> dict[str, object]:
    selection = config["selection"]
    stability = compute_model_stability_diagnostics(config, train_metrics, cv_metrics, holdout_metrics, holdout_ranking)
    status = "accepted"
    if stability["overfit_warning_flag"]:
        status = "overfit_warning"
    if holdout_metrics["balanced_accuracy"] < float(selection["min_balanced_accuracy"]):
        status = "low_balanced_accuracy"
    if holdout_metrics["precision_wr"] < float(selection["min_precision"]):
        status = "holdout_precision_floor_not_met"
    if not bool(threshold.get("balanced_accuracy_floor_met", True)):
        status = "cv_balanced_accuracy_floor_not_met"
    if not bool(threshold["precision_floor_met"]):
        status = "cv_precision_floor_not_met"
    row: dict[str, object] = {
        "run_id": run_id,
        "dataset_variant": variant,
        "negative_ratio": negative_ratio,
        "negative_ratio_label": negative_ratio_label(negative_ratio),
        "feature_set": feature_set_name,
        "model": model_name,
        "sampler": sampler_name,
        "n_train": int(len(y_train)),
        "n_holdout": int(len(y_holdout)),
        "wr_train": int(y_train.sum()),
        "wr_holdout": int(y_holdout.sum()),
        "bayes_best_score": float(search.best_score_),
        "bayes_best_params": json.dumps(search.best_params_, sort_keys=True, default=str),
        "selected_threshold": float(threshold["threshold"]),
        "threshold_precision_floor_met": bool(threshold["precision_floor_met"]),
        "threshold_accuracy_floor_met": bool(threshold.get("accuracy_floor_met", True)),
        "threshold_balanced_accuracy_floor_met": bool(threshold.get("balanced_accuracy_floor_met", True)),
        "threshold_cv_accuracy": float(threshold.get("accuracy", np.nan)),
        "threshold_cv_balanced_accuracy": float(threshold.get("balanced_accuracy", np.nan)),
        "selection_status": status,
    }
    row.update(stability)
    row.update({f"cv_{key}": value for key, value in cv_metrics.items()})
    row.update({f"train_{key}": value for key, value in train_metrics.items()})
    row.update({f"holdout_{key}": value for key, value in holdout_metrics.items()})
    row.update({f"holdout_{key}": value for key, value in holdout_ranking.items()})
    return row


def compute_model_stability_diagnostics(
    config: dict,
    train_metrics: dict[str, object],
    cv_metrics: dict[str, object],
    holdout_metrics: dict[str, object],
    holdout_ranking: dict[str, object],
) -> dict[str, object]:
    selection = config.get("selection", {})
    train_cv_gap_f2 = _metric_gap(train_metrics, cv_metrics, "f2_wr")
    holdout_cv_gap_f2 = _metric_gap(holdout_metrics, cv_metrics, "f2_wr")
    train_cv_gap_ap = _metric_gap(train_metrics, cv_metrics, "average_precision")
    cv_holdout_drop_ap = _metric_gap(cv_metrics, holdout_metrics, "average_precision")
    holdout_to_cv_ap_ratio = _safe_ratio(
        _metric_value(holdout_metrics, "average_precision"),
        _metric_value(cv_metrics, "average_precision"),
    )

    max_f2_gap = float(selection.get("max_train_cv_f2_gap", 0.15))
    max_ap_gap = float(selection.get("max_train_cv_average_precision_gap", 0.20))
    min_holdout_cv_ap_ratio = float(selection.get("min_holdout_cv_average_precision_ratio", 0.25))

    f2_gap_flag = bool(np.isfinite(train_cv_gap_f2) and train_cv_gap_f2 > max_f2_gap)
    ap_gap_flag = bool(np.isfinite(train_cv_gap_ap) and train_cv_gap_ap > max_ap_gap)
    holdout_ap_drop_flag = bool(
        np.isfinite(holdout_to_cv_ap_ratio)
        and holdout_to_cv_ap_ratio < min_holdout_cv_ap_ratio
    )

    ranking_instability_flag = _ranking_instability_flag(holdout_ranking)
    overfit_warning_flag = bool(f2_gap_flag or ap_gap_flag or holdout_ap_drop_flag or ranking_instability_flag)
    overfit_warning_count = int(f2_gap_flag) + int(ap_gap_flag) + int(holdout_ap_drop_flag) + int(ranking_instability_flag)
    overfit_risk_score = (
        _positive_ratio(train_cv_gap_f2, max_f2_gap)
        + _positive_ratio(train_cv_gap_ap, max_ap_gap)
        + _positive_ratio(min_holdout_cv_ap_ratio - holdout_to_cv_ap_ratio, min_holdout_cv_ap_ratio)
        + float(ranking_instability_flag)
    )

    return {
        "cv_train_gap_f2": train_cv_gap_f2,
        "holdout_cv_gap_f2": holdout_cv_gap_f2,
        "train_cv_gap_average_precision": train_cv_gap_ap,
        "cv_holdout_drop_average_precision": cv_holdout_drop_ap,
        "holdout_to_cv_average_precision_ratio": holdout_to_cv_ap_ratio,
        "overfit_gap_f2_flag": f2_gap_flag,
        "overfit_gap_average_precision_flag": ap_gap_flag,
        "holdout_average_precision_drop_flag": holdout_ap_drop_flag,
        "ranking_instability_flag": ranking_instability_flag,
        "overfit_warning_flag": overfit_warning_flag,
        "overfit_warning_count": overfit_warning_count,
        "overfit_risk_score": float(overfit_risk_score),
    }


def _ranking_instability_flag(holdout_ranking: dict[str, object]) -> bool:
    recall_100 = _metric_value(holdout_ranking, "recall_at_100")
    recall_low_fpr = _metric_value(holdout_ranking, "recall_at_fpr_0p005")
    if not np.isfinite(recall_100) or not np.isfinite(recall_low_fpr):
        return False
    return bool(recall_100 < 0.10 and recall_low_fpr < 0.10)


def _metric_gap(left: dict[str, object], right: dict[str, object], key: str) -> float:
    return float(_metric_value(left, key) - _metric_value(right, key))


def _metric_value(metrics: dict[str, object], key: str) -> float:
    try:
        return float(metrics.get(key, np.nan))
    except (TypeError, ValueError):
        return float("nan")


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0:
        return float("nan")
    return float(numerator / denominator)


def _positive_ratio(value: float, scale: float) -> float:
    if not np.isfinite(value) or scale <= 0:
        return 0.0
    return float(max(value, 0.0) / scale)


def save_trained_model(config: dict, model, row: dict[str, object]) -> Path:
    out_dir = run_artifact_dir(config, str(row.get("run_id", "")), "models")
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"{row['dataset_variant']}__neg_{row.get('negative_ratio_label', '10x')}__"
        f"{row['feature_set']}__{row['model']}__{row['sampler']}"
        f"__f2_{row['holdout_f2_wr']:.3f}.joblib"
    )
    path = out_dir / filename
    dump(model, path)
    return path


def save_model_metadata(path: Path, row: dict[str, object], best_params: dict[str, object]) -> None:
    metadata = dict(row)
    metadata["best_params"] = best_params
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str), encoding="utf-8")


def save_validation_artifacts(
    *,
    config: dict,
    model,
    row: dict[str, object],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    train_score,
    x_holdout: pd.DataFrame,
    y_holdout: pd.Series,
    holdout_score,
    oof_score,
    train_identity: pd.DataFrame | None,
    holdout_identity: pd.DataFrame | None,
    threshold: float,
) -> dict[str, str]:
    base = _artifact_stem(row)
    figures_dir = run_artifact_dir(config, str(row.get("run_id", "")), "figures")
    reports_dir = run_artifact_dir(config, str(row.get("run_id", "")), "reports")
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    holdout_pred = (holdout_score >= threshold).astype("int8")
    cm_path = figures_dir / f"{base}__holdout_confusion_matrix.png"
    _save_confusion_matrix(cm_path, y_holdout, holdout_pred)

    roc_path = figures_dir / f"{base}__holdout_roc_curve.png"
    _save_roc_curve(roc_path, y_holdout, holdout_score)

    pr_path = figures_dir / f"{base}__holdout_precision_recall_curve.png"
    _save_precision_recall_curve(pr_path, y_holdout, holdout_score)

    predictions_path = reports_dir / f"{base}__predictions.csv"
    make_predictions_frame(
        y_train=y_train,
        oof_score=oof_score,
        y_holdout=y_holdout,
        holdout_score=holdout_score,
        holdout_pred=holdout_pred,
        train_identity=train_identity,
        holdout_identity=holdout_identity,
        threshold=threshold,
    ).to_csv(predictions_path, index=False)

    importance_path, importance_fig_path = _save_feature_importance_artifacts(
        config=config,
        model=model,
        row=row,
        x_holdout=x_holdout,
        y_holdout=y_holdout,
        base=base,
        reports_dir=reports_dir,
        figures_dir=figures_dir,
    )

    return {
        "holdout_confusion_matrix_path": make_project_relative(cm_path),
        "holdout_roc_curve_path": make_project_relative(roc_path),
        "holdout_pr_curve_path": make_project_relative(pr_path),
        "predictions_path": make_project_relative(predictions_path),
        "feature_importance_path": make_project_relative(importance_path),
        "feature_importance_figure_path": make_project_relative(importance_fig_path),
    }


def make_prediction_identity(df: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    identity = pd.DataFrame({"row_id": index.to_numpy()}, index=index)
    if "source_id" in df.columns:
        identity["source_id"] = df.loc[index, "source_id"].to_numpy()
    else:
        identity["source_id"] = pd.NA
    return identity.reset_index(drop=True)


def make_predictions_frame(
    *,
    y_train: pd.Series,
    oof_score,
    y_holdout: pd.Series,
    holdout_score,
    holdout_pred,
    train_identity: pd.DataFrame | None,
    holdout_identity: pd.DataFrame | None,
    threshold: float,
) -> pd.DataFrame:
    train_pred = (np.asarray(oof_score, dtype="float64") >= threshold).astype("int8")
    train = _prediction_split_frame(
        split="train_oof",
        y=y_train,
        score=oof_score,
        predicted=train_pred,
        identity=train_identity,
        threshold=threshold,
    )
    holdout = _prediction_split_frame(
        split="holdout",
        y=y_holdout,
        score=holdout_score,
        predicted=holdout_pred,
        identity=holdout_identity,
        threshold=threshold,
    )
    return pd.concat([train, holdout], ignore_index=True, sort=False)


def _prediction_split_frame(
    *,
    split: str,
    y: pd.Series,
    score,
    predicted,
    identity: pd.DataFrame | None,
    threshold: float,
) -> pd.DataFrame:
    if identity is None:
        identity = pd.DataFrame({"row_id": y.index.to_numpy(), "source_id": pd.NA})
    else:
        identity = identity.reset_index(drop=True).copy()
    out = pd.DataFrame(
        {
            "split": split,
            "row_id": identity["row_id"].to_numpy(),
            "source_id": identity["source_id"].to_numpy(),
            "target": y.astype(int).to_numpy(),
            "score": np.asarray(score, dtype="float64"),
            "predicted": np.asarray(predicted, dtype="int8"),
            "threshold": threshold,
        }
    )
    return out


def _save_confusion_matrix(path: Path, y_true, y_pred) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["non-WR", "WR"])
    ax.set_yticks([0, 1], labels=["non-WR", "WR"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Holdout confusion matrix")
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            ax.text(col_idx, row_idx, str(matrix[row_idx, col_idx]), ha="center", va="center")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_roc_curve(path: Path, y_true, y_score) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_true, y_score, ax=ax)
    ax.set_title("Holdout ROC curve")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_precision_recall_curve(path: Path, y_true, y_score) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    PrecisionRecallDisplay.from_predictions(y_true, y_score, ax=ax)
    ax.set_title("Holdout precision-recall curve")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_feature_importance_artifacts(
    *,
    config: dict,
    model,
    row: dict[str, object],
    x_holdout: pd.DataFrame,
    y_holdout: pd.Series,
    base: str,
    reports_dir: Path,
    figures_dir: Path,
) -> tuple[Path, Path]:
    table = _model_importance_table(model, x_holdout.columns)
    if table.empty and len(y_holdout) >= 2 and y_holdout.nunique() == 2:
        result = permutation_importance(
            model,
            x_holdout,
            y_holdout,
            scoring=make_constrained_fbeta_scorer(config),
            n_repeats=5,
            random_state=int(config.get("random_state", 42)),
            n_jobs=1,
        )
        table = pd.DataFrame(
            {
                "feature": x_holdout.columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
                "importance_type": "permutation_holdout_f2",
            }
        )
    if table.empty:
        table = pd.DataFrame({"feature": x_holdout.columns, "importance_mean": 0.0, "importance_std": 0.0, "importance_type": "unavailable"})
    table = table.sort_values("importance_mean", ascending=False).reset_index(drop=True)
    csv_path = reports_dir / f"{base}__feature_importance.csv"
    fig_path = figures_dir / f"{base}__feature_importance.png"
    table.to_csv(csv_path, index=False)
    _save_feature_importance_plot(fig_path, table)
    return csv_path, fig_path


def _model_importance_table(model, feature_names) -> pd.DataFrame:
    estimator = model.named_steps.get("estimator")
    if estimator is None:
        return pd.DataFrame()
    if hasattr(estimator, "feature_importances_"):
        return pd.DataFrame(
            {
                "feature": feature_names,
                "importance_mean": estimator.feature_importances_,
                "importance_std": 0.0,
                "importance_type": "model_feature_importance",
            }
        )
    if hasattr(estimator, "coef_"):
        coef = estimator.coef_[0]
        return pd.DataFrame(
            {
                "feature": feature_names,
                "importance_mean": abs(coef),
                "importance_std": 0.0,
                "importance_type": "abs_coefficient",
            }
        )
    return pd.DataFrame()


def _save_feature_importance_plot(path: Path, table: pd.DataFrame) -> None:
    plot_df = table.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.3 * len(plot_df))))
    ax.barh(plot_df["feature"], plot_df["importance_mean"])
    ax.set_xlabel("Importance")
    ax.set_title("Feature importance")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _artifact_stem(row: dict[str, object]) -> str:
    parts = [row["dataset_variant"], f"neg_{row.get('negative_ratio_label', '10x')}", row["feature_set"], row["model"], row["sampler"]]
    return "__".join(str(part).replace("/", "_").replace(" ", "_") for part in parts)


def load_training_results(config_path: str | Path = "configs/models.yaml") -> pd.DataFrame:
    config = load_yaml(config_path)
    return pd.read_csv(resolve_path(config["outputs"]["training_results"]))


def run_root_dir(config: dict, run_id: str) -> Path:
    return resolve_path(config["outputs"]["reports_dir"]) / "runs" / run_id


def run_artifact_dir(config: dict, run_id: str, artifact_kind: str) -> Path:
    if not run_id:
        return resolve_path(config["outputs"]["reports_dir"])
    return run_root_dir(config, run_id) / artifact_kind


def run_training_results_path(config: dict, run_id: str) -> Path:
    return run_root_dir(config, run_id) / "model_training_results.csv"


def load_completed_run_results(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def append_run_result(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    incoming = pd.DataFrame([row])
    if path.exists():
        existing = pd.read_csv(path)
        results = merge_training_results(existing, incoming)
    else:
        results = incoming
    results.to_csv(path, index=False)


def completed_configuration_keys(results: pd.DataFrame) -> set[tuple[str, str, str, str, str]]:
    if results.empty:
        return set()
    return {
        configuration_key(
            variant=str(row.dataset_variant),
            negative_ratio=str(getattr(row, "negative_ratio_label", getattr(row, "negative_ratio", "10x"))),
            feature_set_name=str(row.feature_set),
            model_name=str(row.model),
            sampler_name=str(row.sampler),
        )
        for row in results.itertuples(index=False)
    }


def configuration_key(
    *,
    variant: str,
    negative_ratio: int | str,
    feature_set_name: str,
    model_name: str,
    sampler_name: str,
) -> tuple[str, str, str, str, str]:
    return (variant, negative_ratio_label(negative_ratio), feature_set_name, model_name, sampler_name)
