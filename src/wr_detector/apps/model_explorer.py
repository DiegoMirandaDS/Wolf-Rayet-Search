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
from wr_detector.apps.explorer_ui.pages import cases_page, compare, layers_page, model_detail, overview, stats_page
from wr_detector.modeling.explorer import explorer_db_path


def main() -> None:
    st.set_page_config(page_title="WR Model Explorer", page_icon="*", layout="wide")
    args = _parse_args()
    st.session_state["explorer_config_path"] = str(args.config)

    runs = data.runs()
    if runs.empty:
        st.error("No training runs found in the configured DuckDB history.")
        st.caption(f"History database: `{explorer_db_path(args.config)}`")
        return

    pages = [
        st.Page(overview.render, title="Overview", icon=":material/dashboard:", url_path="overview", default=True),
        st.Page(compare.render, title="Compare models", icon=":material/leaderboard:", url_path="compare"),
        st.Page(model_detail.render, title="Model detail", icon=":material/query_stats:", url_path="model"),
        st.Page(cases_page.render, title="Case review", icon=":material/travel_explore:", url_path="cases"),
        st.Page(stats_page.render, title="Statistics", icon=":material/insights:", url_path="stats"),
        st.Page(layers_page.render, title="Validation layers", icon=":material/layers:", url_path="layers"),
    ]
    navigation = st.navigation(pages)

    with st.sidebar:
        labels = {
            row.run_id: f"{row.run_id} | {int(row.row_count)} models"
            for row in runs.itertuples(index=False)
        }
        run_id = st.selectbox("Training run", options=list(labels), format_func=lambda value: labels.get(value, value))
        st.session_state["run_id"] = run_id
        st.caption(f"History: `{explorer_db_path(args.config)}`")

    navigation.run()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default="configs/models.yaml")
    return parser.parse_known_args()[0]


if __name__ == "__main__":
    main()
