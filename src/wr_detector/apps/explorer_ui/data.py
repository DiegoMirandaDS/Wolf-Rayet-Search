"""Cached data access for the Model Explorer pages.

All loaders delegate to the tested query layer in
``wr_detector.modeling.explorer`` and ``wr_detector.modeling.cases``;
this module only adds Streamlit caching and session-state plumbing.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from wr_detector.modeling import cases as case_review
from wr_detector.modeling.case_visualization import (
    CaseDataFingerprint,
    CasePlotData,
    CasePlotSettings,
    case_data_fingerprint,
    prepare_case_plot_data,
)
from wr_detector.modeling.candidate_stacks import (
    compatible_layer_results,
    evaluate_candidate_stack,
    list_compatible_stack_runs,
    validator_pairs,
)
from wr_detector.modeling.layers import VALIDATION_LAYERS, list_layer_runs, load_layer_results
from wr_detector.modeling import prediction_pool_review as pool_review
from wr_detector.modeling.explorer import (
    enrich_model_results,
    list_explorer_runs,
    load_feature_importance,
    load_prediction_summary,
    load_run_results,
)


def config_path() -> str:
    return st.session_state.get("explorer_config_path", "configs/models.yaml")


def run_id() -> str:
    return str(st.session_state.get("run_id", "") or "")


def candidate_config_path() -> str:
    return st.session_state.get(
        "candidate_config_path",
        "configs/prediction_pool_candidates.yaml",
    )


def runs() -> pd.DataFrame:
    return _runs(config_path())


def results() -> pd.DataFrame:
    run = run_id()
    if not run:
        return _empty_results()
    return _results(config_path(), run)


def feature_importance(result_id: str) -> pd.DataFrame:
    return _feature_importance(config_path(), run_id(), result_id)


def prediction_summary(result_id: str) -> pd.DataFrame:
    return _prediction_summary(config_path(), run_id(), result_id)


def cases(
    result_id: str,
    split: str,
    fingerprint: CaseDataFingerprint | None = None,
) -> pd.DataFrame:
    fingerprint = fingerprint or case_fingerprint(result_id, split)
    return _cases(config_path(), run_id(), result_id, split, fingerprint.value)


def case_fingerprint(result_id: str, split: str) -> CaseDataFingerprint:
    return case_data_fingerprint(
        config_path(),
        run_id=run_id(),
        result_id=result_id,
        split=split,
    )


def case_plot_data(
    result_id: str,
    split: str,
    settings: CasePlotSettings,
    fingerprint: CaseDataFingerprint | None = None,
) -> CasePlotData:
    fingerprint = fingerprint or case_fingerprint(result_id, split)
    return _case_plot_data(
        config_path(),
        run_id(),
        result_id,
        split,
        fingerprint.value,
        settings,
    )


def case_plot_bases(
    result_id: str,
    split: str,
    settings: CasePlotSettings,
    fingerprint: CaseDataFingerprint | None = None,
) -> dict[str, dict[str, object]]:
    fingerprint = fingerprint or case_fingerprint(result_id, split)
    return _case_plot_bases(
        config_path(),
        run_id(),
        result_id,
        split,
        fingerprint.value,
        settings,
    )


def case_overlap(split: str, kind: str, top_k: int) -> pd.DataFrame:
    return _case_overlap(config_path(), run_id(), split, kind, top_k)


def layer_runs(layer_key: str) -> pd.DataFrame:
    return _layer_runs(layer_key)


def layer_results(layer_key: str, run_id: str) -> pd.DataFrame:
    return _layer_results(layer_key, run_id)


def compatible_stack_runs(layer_key: str, result_id: str) -> pd.DataFrame:
    return _compatible_stack_runs(config_path(), run_id(), layer_key, result_id)


def stack_validator_pairs(layer_key: str, layer_run_id: str, result_id: str) -> pd.DataFrame:
    return _stack_validator_pairs(config_path(), run_id(), layer_key, layer_run_id, result_id)


def candidate_stack(
    layer_key: str,
    layer_run_id: str,
    result_id: str,
    feature_set: str,
    method: str,
) -> dict[str, pd.DataFrame]:
    return _candidate_stack(
        config_path(),
        run_id(),
        layer_key,
        layer_run_id,
        result_id,
        feature_set,
        method,
    )


def candidate_availability() -> dict[str, object]:
    return pool_review.review_availability(candidate_config_path())


def candidate_review_runs() -> pd.DataFrame:
    return _candidate_review_runs(
        candidate_config_path(),
        _candidate_revision(),
    )


def configured_pool_build_id() -> str | None:
    return pool_review.configured_pool_build_id(candidate_config_path())


def configured_scoring_run_id() -> str | None:
    return pool_review.configured_scoring_run_id(candidate_config_path())


def available_pool_build_ids() -> list[str]:
    return _available_pool_build_ids(candidate_config_path(), _candidate_revision())


def available_scoring_run_ids() -> list[str]:
    return _available_scoring_run_ids(candidate_config_path(), _candidate_revision())


def pool_status(pool_build_id: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _pool_status(
        candidate_config_path(),
        _candidate_revision(),
        pool_build_id or "",
    )


def scoring_status(scoring_run_id: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _scoring_status(
        candidate_config_path(),
        _candidate_revision(),
        scoring_run_id or "",
    )


def candidate_models() -> pd.DataFrame:
    return _candidate_models(candidate_config_path(), _candidate_revision())


def candidate_summary() -> dict[str, object]:
    return _candidate_summary(candidate_config_path(), _candidate_revision())


def candidate_dispositions() -> pd.DataFrame:
    return _candidate_dispositions(
        candidate_config_path(),
        _candidate_revision(),
    )


def candidate_rankings(
    *,
    view: str = "consensus",
    result_id: str | None = None,
    limit: int = 500,
    dispositions: tuple[str, ...] = (),
    min_model_support: int = 1,
    followup_only: bool = False,
    require_halpha: bool = False,
) -> pd.DataFrame:
    return _candidate_rankings(
        candidate_config_path(),
        _candidate_revision(),
        view,
        result_id,
        int(limit),
        tuple(dispositions),
        int(min_model_support),
        bool(followup_only),
        bool(require_halpha),
    )


def candidate_detail(source_id: int) -> pd.DataFrame:
    return _candidate_detail(
        candidate_config_path(),
        _candidate_revision(),
        int(source_id),
    )


def candidate_model_evidence(source_id: int) -> pd.DataFrame:
    return _candidate_model_evidence(
        candidate_config_path(),
        _candidate_revision(),
        int(source_id),
    )


def candidate_jaccard(top_k: int) -> pd.DataFrame:
    return _candidate_jaccard(
        candidate_config_path(),
        _candidate_revision(),
        int(top_k),
    )


def candidate_plot_frame(
    *,
    min_parallax_over_error: float = 2.0,
    max_distance_kpc: float = 15.0,
) -> pd.DataFrame:
    return _candidate_plot_frame(
        candidate_config_path(),
        _candidate_revision(),
        float(min_parallax_over_error),
        float(max_distance_kpc),
    )


def _candidate_revision() -> str:
    return pool_review.prediction_pool_review_revision(
        candidate_config_path()
    )


def _layer(layer_key: str):
    return next(layer for layer in VALIDATION_LAYERS if layer.key == layer_key)


@st.cache_data(show_spinner=False, ttl=60)
def _layer_runs(layer_key: str) -> pd.DataFrame:
    return list_layer_runs(_layer(layer_key))


@st.cache_data(show_spinner=False, ttl=60)
def _layer_results(layer_key: str, run_id: str) -> pd.DataFrame:
    return load_layer_results(_layer(layer_key), run_id)


@st.cache_data(show_spinner=False, ttl=60)
def _compatible_stack_runs(
    config: str,
    model_run: str,
    layer_key: str,
    result_id: str,
) -> pd.DataFrame:
    return list_compatible_stack_runs(
        config,
        model_run_id=model_run,
        result_id=result_id,
        layer=_layer(layer_key),
    )


@st.cache_data(show_spinner=False, ttl=60)
def _stack_validator_pairs(
    config: str,
    model_run: str,
    layer_key: str,
    layer_run_id: str,
    result_id: str,
) -> pd.DataFrame:
    results = _layer_results(layer_key, layer_run_id)
    first = _results(config, model_run)
    match = first[first["result_id"].astype(str).eq(str(result_id))]
    if match.empty:
        return pd.DataFrame()
    return validator_pairs(compatible_layer_results(match.iloc[0], results))


@st.cache_data(show_spinner="Evaluating first-stage + second-layer stack...")
def _candidate_stack(
    config: str,
    model_run: str,
    layer_key: str,
    layer_run_id: str,
    result_id: str,
    feature_set: str,
    method: str,
) -> dict[str, pd.DataFrame]:
    return evaluate_candidate_stack(
        config,
        model_run_id=model_run,
        result_id=result_id,
        layer=_layer(layer_key),
        layer_run_id=layer_run_id,
        feature_set=feature_set,
        method=method,
    )


@st.cache_data(show_spinner=False)
def _candidate_review_runs(
    config: str,
    revision: str,
) -> pd.DataFrame:
    return pool_review.load_candidate_review_runs(config)


@st.cache_data(show_spinner=False)
def _available_pool_build_ids(config: str, revision: str) -> list[str]:
    return pool_review.available_pool_build_ids(config)


@st.cache_data(show_spinner=False)
def _available_scoring_run_ids(config: str, revision: str) -> list[str]:
    return pool_review.available_scoring_run_ids(config)


@st.cache_data(show_spinner=False)
def _pool_status(
    config: str,
    revision: str,
    pool_build_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return pool_review.load_pool_status(
        config,
        pool_build_id=pool_build_id or None,
    )


@st.cache_data(show_spinner=False)
def _scoring_status(
    config: str,
    revision: str,
    scoring_run_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return pool_review.load_scoring_status(
        config,
        scoring_run_id=scoring_run_id or None,
    )


@st.cache_data(show_spinner=False)
def _candidate_models(
    config: str,
    revision: str,
) -> pd.DataFrame:
    return pool_review.load_candidate_models(config)


@st.cache_data(show_spinner=False)
def _candidate_summary(
    config: str,
    revision: str,
) -> dict[str, object]:
    return pool_review.load_candidate_summary(config)


@st.cache_data(show_spinner=False)
def _candidate_dispositions(
    config: str,
    revision: str,
) -> pd.DataFrame:
    return pool_review.load_disposition_counts(config)


@st.cache_data(show_spinner="Loading candidate rankings...")
def _candidate_rankings(
    config: str,
    revision: str,
    view: str,
    result_id: str | None,
    limit: int,
    dispositions: tuple[str, ...],
    min_model_support: int,
    followup_only: bool,
    require_halpha: bool,
) -> pd.DataFrame:
    return pool_review.load_candidate_rankings(
        config,
        view=view,
        result_id=result_id,
        limit=limit,
        dispositions=dispositions,
        min_model_support=min_model_support,
        followup_only=followup_only,
        require_halpha=require_halpha,
    )


@st.cache_data(show_spinner=False)
def _candidate_detail(
    config: str,
    revision: str,
    source_id: int,
) -> pd.DataFrame:
    return pool_review.load_candidate_detail(
        config,
        source_id=source_id,
    )


@st.cache_data(show_spinner=False)
def _candidate_model_evidence(
    config: str,
    revision: str,
    source_id: int,
) -> pd.DataFrame:
    return pool_review.load_candidate_model_evidence(
        config,
        source_id=source_id,
    )


@st.cache_data(show_spinner=False)
def _candidate_jaccard(
    config: str,
    revision: str,
    top_k: int,
) -> pd.DataFrame:
    return pool_review.load_candidate_jaccard(config, top_k=top_k)


@st.cache_data(show_spinner="Preparing candidate visualizations...")
def _candidate_plot_frame(
    config: str,
    revision: str,
    min_parallax_over_error: float,
    max_distance_kpc: float,
) -> pd.DataFrame:
    return pool_review.load_candidate_plot_frame(
        config,
        min_parallax_over_error=min_parallax_over_error,
        max_distance_kpc=max_distance_kpc,
    )


@st.cache_data(show_spinner=False)
def _runs(config: str) -> pd.DataFrame:
    return list_explorer_runs(config)


@st.cache_data(show_spinner=False)
def _results(config: str, run: str) -> pd.DataFrame:
    return load_run_results(config, run_id=run)


def _empty_results() -> pd.DataFrame:
    return enrich_model_results(pd.DataFrame())


@st.cache_data(show_spinner=False)
def _feature_importance(config: str, run: str, result_id: str) -> pd.DataFrame:
    return load_feature_importance(config, run_id=run, result_id=result_id)


@st.cache_data(show_spinner=False)
def _prediction_summary(config: str, run: str, result_id: str) -> pd.DataFrame:
    return load_prediction_summary(config, run_id=run, result_id=result_id)


@st.cache_data(show_spinner="Loading case predictions...")
def _cases(
    config: str,
    run: str,
    result_id: str,
    split: str,
    fingerprint: str,
) -> pd.DataFrame:
    return case_review.load_case_predictions(config, run_id=run, result_id=result_id, split=split)


@st.cache_data(show_spinner="Preparing Case Review visualizations...")
def _case_plot_data(
    config: str,
    run: str,
    result_id: str,
    split: str,
    fingerprint: str,
    settings: CasePlotSettings,
) -> CasePlotData:
    cases = _cases(config, run, result_id, split, fingerprint)
    return prepare_case_plot_data(
        cases,
        settings=settings,
        fingerprint=fingerprint,
    )


@st.cache_resource(show_spinner=False)
def _case_plot_bases(
    config: str,
    run: str,
    result_id: str,
    split: str,
    fingerprint: str,
    settings: CasePlotSettings,
) -> dict[str, dict[str, object]]:
    from wr_detector.apps.explorer_ui.case_plots import (
        build_case_plot_base,
        chart_payload_bytes,
    )

    prepared = _case_plot_data(
        config,
        run,
        result_id,
        split,
        fingerprint,
        settings,
    )
    charts: dict[str, dict[str, object]] = {}
    for kind in ["photometric", "mollweide", "galactic_plane"]:
        chart = build_case_plot_base(kind, prepared)
        if chart is not None:
            charts[kind] = {
                "chart": chart,
                "payload_bytes": chart_payload_bytes(chart),
            }
    return charts


@st.cache_data(show_spinner="Aggregating cases across models...")
def _case_overlap(config: str, run: str, split: str, kind: str, top_k: int) -> pd.DataFrame:
    return case_review.load_case_overlap(config, run_id=run, split=split, kind=kind, top_k=top_k)
