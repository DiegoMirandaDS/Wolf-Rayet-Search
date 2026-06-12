"""Cached data access for the Model Explorer pages.

All loaders delegate to the tested query layer in
``wr_detector.modeling.explorer`` and ``wr_detector.modeling.cases``;
this module only adds Streamlit caching and session-state plumbing.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from wr_detector.modeling import cases as case_review
from wr_detector.modeling.layers import VALIDATION_LAYERS, list_layer_runs, load_layer_results
from wr_detector.modeling.explorer import (
    list_explorer_runs,
    load_feature_importance,
    load_prediction_summary,
    load_run_results,
)


def config_path() -> str:
    return st.session_state.get("explorer_config_path", "configs/models.yaml")


def run_id() -> str:
    return st.session_state["run_id"]


def runs() -> pd.DataFrame:
    return _runs(config_path())


def results() -> pd.DataFrame:
    return _results(config_path(), run_id())


def feature_importance(result_id: str) -> pd.DataFrame:
    return _feature_importance(config_path(), run_id(), result_id)


def prediction_summary(result_id: str) -> pd.DataFrame:
    return _prediction_summary(config_path(), run_id(), result_id)


def cases(result_id: str, split: str) -> pd.DataFrame:
    return _cases(config_path(), run_id(), result_id, split)


def case_overlap(split: str, kind: str, top_k: int) -> pd.DataFrame:
    return _case_overlap(config_path(), run_id(), split, kind, top_k)


def layer_runs(layer_key: str) -> pd.DataFrame:
    return _layer_runs(layer_key)


def layer_results(layer_key: str, run_id: str) -> pd.DataFrame:
    return _layer_results(layer_key, run_id)


def _layer(layer_key: str):
    return next(layer for layer in VALIDATION_LAYERS if layer.key == layer_key)


@st.cache_data(show_spinner=False, ttl=60)
def _layer_runs(layer_key: str) -> pd.DataFrame:
    return list_layer_runs(_layer(layer_key))


@st.cache_data(show_spinner=False, ttl=60)
def _layer_results(layer_key: str, run_id: str) -> pd.DataFrame:
    return load_layer_results(_layer(layer_key), run_id)


@st.cache_data(show_spinner=False)
def _runs(config: str) -> pd.DataFrame:
    return list_explorer_runs(config)


@st.cache_data(show_spinner=False)
def _results(config: str, run: str) -> pd.DataFrame:
    return load_run_results(config, run_id=run)


@st.cache_data(show_spinner=False)
def _feature_importance(config: str, run: str, result_id: str) -> pd.DataFrame:
    return load_feature_importance(config, run_id=run, result_id=result_id)


@st.cache_data(show_spinner=False)
def _prediction_summary(config: str, run: str, result_id: str) -> pd.DataFrame:
    return load_prediction_summary(config, run_id=run, result_id=result_id)


@st.cache_data(show_spinner="Loading case predictions...")
def _cases(config: str, run: str, result_id: str, split: str) -> pd.DataFrame:
    return case_review.load_case_predictions(config, run_id=run, result_id=result_id, split=split)


@st.cache_data(show_spinner="Aggregating cases across models...")
def _case_overlap(config: str, run: str, split: str, kind: str, top_k: int) -> pd.DataFrame:
    return case_review.load_case_overlap(config, run_id=run, split=split, kind=kind, top_k=top_k)
