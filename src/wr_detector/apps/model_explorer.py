"""Streamlit entry point for the WR Model Explorer.

Launch through the CLI (``wr-detector explore-models``) or directly:

    streamlit run src/wr_detector/apps/model_explorer.py -- --config configs/models.yaml

Page logic lives in ``wr_detector.apps.explorer_ui``; query logic lives in
``wr_detector.modeling.explorer`` and ``wr_detector.modeling.cases``.
"""

from __future__ import annotations

import argparse

import streamlit as st

from wr_detector.apps.explorer_ui import data
from wr_detector.apps.explorer_ui.pages import (
    candidate_stack_page,
    candidate_rankings_page,
    candidate_review_page,
    cases_page,
    compare,
    layers_page,
    model_detail,
    overview,
    pool_status_page,
    stats_page,
)
from wr_detector.modeling.explorer import explorer_db_path


def main() -> None:
    st.set_page_config(page_title="WR Detector Explorer", page_icon="*", layout="wide")
    args = _parse_args()
    st.session_state["explorer_config_path"] = str(args.config)
    st.session_state["candidate_config_path"] = str(args.candidate_config)

    runs = data.runs()
    model_pages = [
        st.Page(overview.render, title="Overview", icon=":material/dashboard:", url_path="overview", default=True),
        st.Page(compare.render, title="Compare models", icon=":material/leaderboard:", url_path="compare"),
        st.Page(model_detail.render, title="Model detail", icon=":material/query_stats:", url_path="model"),
        st.Page(cases_page.render, title="Case review", icon=":material/travel_explore:", url_path="cases"),
        st.Page(stats_page.render, title="Statistics", icon=":material/insights:", url_path="stats"),
    ]
    validation_pages = [
        st.Page(layers_page.render, title="Validation layers", icon=":material/layers:", url_path="layers"),
        st.Page(
            candidate_stack_page.render,
            title="Candidate stack",
            icon=":material/account_tree:",
            url_path="candidate-stack",
        ),
    ]
    candidate_review_navigation = st.Page(
        candidate_review_page.render,
        title="Candidate review",
        icon=":material/find_in_page:",
        url_path="candidate-review",
    )
    prediction_pool_pages = [
        st.Page(
            pool_status_page.render,
            title="Pool status",
            icon=":material/database:",
            url_path="pool-status",
        ),
        st.Page(
            candidate_rankings_page.render,
            title="Candidate rankings",
            icon=":material/format_list_numbered:",
            url_path="candidate-rankings",
        ),
        candidate_review_navigation,
    ]
    st.session_state["_candidate_review_page"] = candidate_review_navigation
    navigation = st.navigation(
        {
            "Models": model_pages,
            "Validation / second layer": validation_pages,
            "Prediction pool": prediction_pool_pages,
        },
        expanded=True,
    )

    with st.sidebar:
        if runs.empty:
            st.error("No training runs found in the configured DuckDB history.")
            st.session_state["run_id"] = ""
        else:
            labels = {
                row.run_id: f"{row.run_id} | {int(row.row_count)} models"
                for row in runs.itertuples(index=False)
            }
            run_id = st.selectbox(
                "Training run",
                options=list(labels),
                format_func=lambda value: labels.get(value, value),
            )
            st.session_state["run_id"] = run_id
        st.caption(f"History: `{explorer_db_path(args.config)}`")
        st.caption(f"Candidate review: `{args.candidate_config}`")

    navigation.run()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default="configs/models.yaml")
    parser.add_argument(
        "--candidate-config",
        default="configs/prediction_pool_candidates.yaml",
    )
    return parser.parse_known_args()[0]


if __name__ == "__main__":
    main()
