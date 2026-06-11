from __future__ import annotations

import json
import re
from pathlib import Path
from time import perf_counter
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from wr_detector.config import load_yaml, resolve_path
from wr_detector.modeling.history import make_project_relative, make_training_run_id
from wr_detector.modeling.negative_reduction import configured_negative_ratios, negative_ratio_label
from wr_detector.modeling.training import load_reduced_or_source_dataset


SUBTYPE_RE = {
    "WO": re.compile(r"\bWO", re.IGNORECASE),
    "WC": re.compile(r"\bWC", re.IGNORECASE),
    "WN": re.compile(r"\bWN", re.IGNORECASE),
}

DERIVED_FEATURES: dict[str, tuple[str, ...]] = {
    "W2_W3": ("W2", "W3"),
    "W3_W4": ("W3", "W4"),
    "W1_W3": ("W1", "W3"),
    "K_W2": ("Ks", "W2"),
    "J_W2": ("J", "W2"),
    "G_W1": ("G", "W1"),
    "pm_total": ("pmra", "pmdec"),
    "G_flux_snr": ("G_flux", "G_flux_error"),
    "BP_flux_snr": ("BP_flux", "BP_flux_error"),
    "RP_flux_snr": ("RP_flux", "RP_flux_error"),
    "J_snr": ("J_error",),
    "H_snr": ("H_error",),
    "Ks_snr": ("Ks_error",),
    "W1_snr": ("W1_error",),
    "W2_snr": ("W2_error",),
    "W3_snr": ("W3_error",),
    "W4_snr": ("W4_error",),
}


def train_second_layer_validators(
    config_path: str | Path = "configs/second_layer.yaml",
    *,
    variants: list[str] | None = None,
    feature_sets: list[str] | None = None,
    methods: list[str] | None = None,
    subtypes: list[str] | None = None,
    negative_ratios: list[int | str] | None = None,
    run_id: str | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    config = load_second_layer_config(config_path)
    model_config = config["models_config"]
    run_id = run_id or make_training_run_id(config_path)
    selected_variants = variants or list(config.get("dataset_variants") or model_config["dataset_variants"])
    selected_feature_sets = feature_sets or list(config["feature_sets"])
    selected_methods = methods or list(config["methods"])
    selected_subtypes = [value.upper() for value in (subtypes or config.get("subtypes", ["WN", "WC"]))]
    selected_ratios = negative_ratios or list(config.get("negative_ratios") or configured_negative_ratios(model_config))

    reference_subtypes = load_reference_subtypes(config["reference_db"])
    rows: list[dict[str, object]] = []
    total = len(selected_variants) * len(selected_ratios) * len(selected_feature_sets) * len(selected_methods) * len(selected_subtypes)
    current = 0
    started = perf_counter()
    for variant in selected_variants:
        for ratio in selected_ratios:
            dataset = load_reduced_or_source_dataset(
                model_config,
                variant,
                negative_ratio=ratio,
                reduce_if_missing=False,
            )
            dataset = annotate_wr_subtypes(dataset, reference_subtypes)
            for feature_set_name in selected_feature_sets:
                requested_features = list(config["feature_sets"][feature_set_name])
                prepared = prepare_second_layer_features(dataset, requested_features)
                available_features = prepared.attrs["available_features"]
                missing_features = prepared.attrs["missing_features"]
                if len(available_features) < int(config.get("min_features", 4)):
                    rows.append(
                        skipped_row(
                            run_id=run_id,
                            variant=variant,
                            negative_ratio=ratio,
                            feature_set=feature_set_name,
                            method="all",
                            subtype="all",
                            reason="insufficient_available_features",
                            available_features=available_features,
                            missing_features=missing_features,
                        )
                    )
                    continue
                for subtype in selected_subtypes:
                    for method_name in selected_methods:
                        current += 1
                        if verbose:
                            print(
                                f"[{current}/{total}] second-layer {variant} neg={negative_ratio_label(ratio)} "
                                f"{feature_set_name} {subtype} {method_name}",
                                flush=True,
                            )
                        rows.append(
                            fit_second_layer_validator(
                                config=config,
                                run_id=run_id,
                                dataset=prepared,
                                variant=variant,
                                negative_ratio=ratio,
                                feature_set_name=feature_set_name,
                                method_name=method_name,
                                subtype=subtype,
                                available_features=available_features,
                                missing_features=missing_features,
                            )
                        )
    results = pd.DataFrame(rows)
    if not results.empty:
        sort_columns = [
            column
            for column in ["status", "holdout_average_precision", "threshold_calibration_negative_pass_rate"]
            if column in results.columns
        ]
        ascending = [column != "holdout_average_precision" for column in sort_columns]
        results = results.sort_values(sort_columns, ascending=ascending, na_position="last").reset_index(drop=True)
    path = second_layer_results_path(config, run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(path, index=False)
    latest_path = resolve_path(config["outputs"].get("latest_results", "reports/tables/second_layer_validation_results.csv"))
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(latest_path, index=False)
    if verbose:
        print(f"Second-layer run completed in {_format_seconds(perf_counter() - started)}", flush=True)
        print(f"results: {path}", flush=True)
    return results


def load_second_layer_config(config_path: str | Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    model_config_path = config.get("models_config", "configs/models.yaml")
    config["models_config_path"] = model_config_path
    config["models_config"] = load_yaml(model_config_path)
    config["reference_db"] = resolve_path(config.get("reference_db", "data/databases/wr_reference.duckdb"))
    return config


def load_reference_subtypes(reference_db: str | Path) -> pd.DataFrame:
    path = resolve_path(reference_db)
    con = duckdb.connect(str(path), read_only=True)
    try:
        frame = con.execute(
            """
            SELECT source_id, wr_id, "Spectral Type" AS spectral_type
            FROM wr_reference
            WHERE source_id IS NOT NULL
            """
        ).fetchdf()
    finally:
        con.close()
    frame["wr_broad_subtype"] = frame["spectral_type"].map(broad_wr_subtype)
    return frame


def broad_wr_subtype(spectral_type: object) -> str:
    value = "" if pd.isna(spectral_type) else str(spectral_type)
    has_wo = bool(SUBTYPE_RE["WO"].search(value))
    has_wc = bool(SUBTYPE_RE["WC"].search(value))
    has_wn = bool(SUBTYPE_RE["WN"].search(value))
    if has_wo:
        return "WO"
    if has_wc and has_wn:
        return "WN/WC"
    if has_wc:
        return "WC"
    if has_wn:
        return "WN"
    return "other_or_missing"


def annotate_wr_subtypes(dataset: pd.DataFrame, reference_subtypes: pd.DataFrame) -> pd.DataFrame:
    columns = ["source_id", "wr_id", "spectral_type", "wr_broad_subtype"]
    merged = dataset.merge(reference_subtypes[columns], on="source_id", how="left", suffixes=("", "_reference"))
    merged.loc[merged["target"].astype(int).eq(0), ["spectral_type", "wr_broad_subtype"]] = pd.NA
    return merged


def prepare_second_layer_features(dataset: pd.DataFrame, requested_features: list[str]) -> pd.DataFrame:
    out = add_derived_features(dataset)
    available = [feature for feature in requested_features if feature in out.columns and not out[feature].isna().all()]
    missing = [feature for feature in requested_features if feature not in available]
    prepared = out.copy()
    prepared.attrs["available_features"] = available
    prepared.attrs["missing_features"] = missing
    return prepared


def add_derived_features(dataset: pd.DataFrame) -> pd.DataFrame:
    out = dataset.copy()
    for feature, columns in DERIVED_FEATURES.items():
        if feature in out.columns:
            continue
        if not all(column in out.columns for column in columns):
            continue
        if len(columns) == 2 and feature.endswith("_snr"):
            numerator, denominator = columns
            out[feature] = _safe_divide(out[numerator], out[denominator])
        elif len(columns) == 1 and feature.endswith("_snr"):
            error_column = columns[0]
            magnitude_column = feature.replace("_snr", "")
            if magnitude_column in out.columns:
                out[feature] = _safe_divide(1.0857362047581296, out[error_column])
        elif feature == "pm_total":
            out[feature] = np.sqrt(out["pmra"].astype(float) ** 2 + out["pmdec"].astype(float) ** 2)
        else:
            left, right = columns
            out[feature] = out[left].astype(float) - out[right].astype(float)
    return out


def fit_second_layer_validator(
    *,
    config: dict[str, Any],
    run_id: str,
    dataset: pd.DataFrame,
    variant: str,
    negative_ratio: int | str,
    feature_set_name: str,
    method_name: str,
    subtype: str,
    available_features: list[str],
    missing_features: list[str],
) -> dict[str, object]:
    train = dataset[dataset["modeling_split"].eq("train")].copy()
    holdout = dataset[dataset["modeling_split"].eq("holdout")].copy()
    calibration = dataset[dataset["modeling_split"].eq("threshold_calibration")].copy()
    train_positive = train[train["wr_broad_subtype"].eq(subtype)]
    min_train = int(config.get("min_train_positives", 20))
    if len(train_positive) < min_train:
        return skipped_row(
            run_id=run_id,
            variant=variant,
            negative_ratio=negative_ratio,
            feature_set=feature_set_name,
            method=method_name,
            subtype=subtype,
            reason="insufficient_train_positives",
            available_features=available_features,
            missing_features=missing_features,
            train_positive_count=len(train_positive),
        )

    try:
        model = build_one_class_pipeline(method_name, config["methods"][method_name], len(train_positive), len(available_features))
        x_train = train_positive[available_features].astype("float64")
        model.fit(x_train)
        train_score = score_one_class_model(model, x_train)
        target_recall = float(config.get("target_positive_recall", 0.90))
        threshold = float(np.quantile(train_score, max(0.0, min(1.0, 1.0 - target_recall))))
        metrics = evaluate_second_layer_model(
            model=model,
            threshold=threshold,
            holdout=holdout,
            calibration=calibration,
            subtype=subtype,
            features=available_features,
        )
        model_path = save_second_layer_model(
            config=config,
            run_id=run_id,
            variant=variant,
            negative_ratio=negative_ratio,
            feature_set=feature_set_name,
            method=method_name,
            subtype=subtype,
            model=model,
            threshold=threshold,
            features=available_features,
        )
        return {
            "run_id": run_id,
            "dataset_variant": variant,
            "negative_ratio": negative_ratio,
            "negative_ratio_label": negative_ratio_label(negative_ratio),
            "feature_set": feature_set_name,
            "method": method_name,
            "subtype": subtype,
            "status": "accepted",
            "threshold": threshold,
            "target_positive_recall": target_recall,
            "n_features": len(available_features),
            "available_features": ",".join(available_features),
            "missing_features": ",".join(missing_features),
            "train_positive_count": len(train_positive),
            "model_path": make_project_relative(model_path),
            **metrics,
        }
    except Exception as exc:
        return skipped_row(
            run_id=run_id,
            variant=variant,
            negative_ratio=negative_ratio,
            feature_set=feature_set_name,
            method=method_name,
            subtype=subtype,
            reason=f"fit_failed:{type(exc).__name__}:{exc}",
            available_features=available_features,
            missing_features=missing_features,
            train_positive_count=len(train_positive),
        )


def build_one_class_pipeline(method_name: str, method_config: dict[str, Any], n_samples: int, n_features: int) -> Pipeline:
    estimator: object
    random_state = int(method_config.get("random_state", 42))
    if method_name == "gaussian_mixture":
        n_components = min(int(method_config.get("n_components", 2)), max(1, n_samples))
        estimator = GaussianMixture(
            n_components=n_components,
            covariance_type=str(method_config.get("covariance_type", "full")),
            reg_covar=float(method_config.get("reg_covar", 1e-5)),
            random_state=random_state,
        )
    elif method_name == "one_class_svm":
        estimator = OneClassSVM(
            kernel=str(method_config.get("kernel", "rbf")),
            nu=float(method_config.get("nu", 0.10)),
            gamma=method_config.get("gamma", "scale"),
        )
    elif method_name == "isolation_forest":
        estimator = IsolationForest(
            n_estimators=int(method_config.get("n_estimators", 300)),
            contamination=method_config.get("contamination", "auto"),
            random_state=random_state,
        )
    elif method_name == "robust_covariance":
        if n_samples <= n_features:
            raise ValueError(f"robust_covariance needs more samples than features: {n_samples} <= {n_features}")
        estimator = EllipticEnvelope(
            contamination=float(method_config.get("contamination", 0.10)),
            support_fraction=method_config.get("support_fraction"),
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unsupported second-layer method: {method_name}")
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("estimator", estimator),
        ]
    )


def score_one_class_model(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "score_samples"):
        return np.asarray(model.score_samples(x), dtype="float64")
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(x), dtype="float64")
    raise TypeError("One-class model does not expose score_samples or decision_function.")


def evaluate_second_layer_model(
    *,
    model: Pipeline,
    threshold: float,
    holdout: pd.DataFrame,
    calibration: pd.DataFrame,
    subtype: str,
    features: list[str],
) -> dict[str, object]:
    holdout_eval = holdout[holdout["target"].astype(int).eq(0) | holdout["wr_broad_subtype"].eq(subtype)].copy()
    holdout_score = score_one_class_model(model, holdout_eval[features].astype("float64")) if not holdout_eval.empty else np.array([])
    holdout_eval["second_layer_score"] = holdout_score
    holdout_eval["second_layer_pass"] = holdout_eval["second_layer_score"].ge(threshold)
    y_true = holdout_eval["wr_broad_subtype"].eq(subtype).astype(int) if not holdout_eval.empty else pd.Series(dtype="int8")
    holdout_positive = holdout_eval[y_true.eq(1)] if not holdout_eval.empty else pd.DataFrame()
    holdout_negative = holdout_eval[y_true.eq(0)] if not holdout_eval.empty else pd.DataFrame()

    calibration_negative = calibration[calibration["target"].astype(int).eq(0)].copy()
    if not calibration_negative.empty:
        calibration_score = score_one_class_model(model, calibration_negative[features].astype("float64"))
        calibration_negative["second_layer_pass"] = calibration_score >= threshold

    return {
        "holdout_positive_count": int(len(holdout_positive)),
        "holdout_negative_count": int(len(holdout_negative)),
        "holdout_positive_retention": _mean_bool(holdout_positive.get("second_layer_pass")),
        "holdout_negative_pass_rate": _mean_bool(holdout_negative.get("second_layer_pass")),
        "threshold_calibration_negative_count": int(len(calibration_negative)),
        "threshold_calibration_negative_pass_rate": _mean_bool(calibration_negative.get("second_layer_pass")),
        "holdout_average_precision": _safe_average_precision(y_true, holdout_score),
        "holdout_roc_auc": _safe_roc_auc(y_true, holdout_score),
    }


def save_second_layer_model(
    *,
    config: dict[str, Any],
    run_id: str,
    variant: str,
    negative_ratio: int | str,
    feature_set: str,
    method: str,
    subtype: str,
    model: Pipeline,
    threshold: float,
    features: list[str],
) -> Path:
    out_dir = second_layer_run_dir(config, run_id) / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (
        f"{variant}__neg_{negative_ratio_label(negative_ratio)}__{feature_set}__"
        f"{subtype.lower()}__{method}.joblib"
    )
    dump(
        {
            "model": model,
            "threshold": threshold,
            "features": features,
            "subtype": subtype,
            "method": method,
            "dataset_variant": variant,
            "negative_ratio": negative_ratio,
        },
        path,
    )
    metadata = path.with_suffix(".json")
    metadata.write_text(
        json.dumps(
            {
                "threshold": threshold,
                "features": features,
                "subtype": subtype,
                "method": method,
                "dataset_variant": variant,
                "negative_ratio": negative_ratio,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def skipped_row(
    *,
    run_id: str,
    variant: str,
    negative_ratio: int | str,
    feature_set: str,
    method: str,
    subtype: str,
    reason: str,
    available_features: list[str],
    missing_features: list[str],
    train_positive_count: int | None = None,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "dataset_variant": variant,
        "negative_ratio": negative_ratio,
        "negative_ratio_label": negative_ratio_label(negative_ratio),
        "feature_set": feature_set,
        "method": method,
        "subtype": subtype,
        "status": "skipped",
        "skip_reason": reason,
        "n_features": len(available_features),
        "available_features": ",".join(available_features),
        "missing_features": ",".join(missing_features),
        "train_positive_count": train_positive_count,
    }


def second_layer_run_dir(config: dict[str, Any], run_id: str) -> Path:
    template = config["outputs"].get("run_dir_template", "reports/modeling/second_layer/runs/{run_id}")
    return resolve_path(template.format(run_id=run_id))


def second_layer_results_path(config: dict[str, Any], run_id: str) -> Path:
    return second_layer_run_dir(config, run_id) / "second_layer_validation_results.csv"


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    den = denominator.astype("float64").replace(0, np.nan)
    return numerator.astype("float64") / den


def _mean_bool(values: pd.Series | None) -> float:
    if values is None or len(values) == 0:
        return float("nan")
    return float(values.astype(bool).mean())


def _safe_average_precision(y_true: pd.Series, y_score: np.ndarray) -> float:
    if len(y_true) == 0 or y_true.nunique() < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def _safe_roc_auc(y_true: pd.Series, y_score: np.ndarray) -> float:
    if len(y_true) == 0 or y_true.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _format_seconds(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{sec:02d}s"
    if minutes:
        return f"{minutes}m{sec:02d}s"
    return f"{sec}s"
