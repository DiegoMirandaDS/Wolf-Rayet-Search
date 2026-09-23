"""Consensus and specialist rankings for the prediction pool."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from wr_detector.apps.explorer_ui import candidate_plots, data, ui


def render() -> None:
    st.title("Candidate rankings")
    st.caption(
        "Reciprocal Rank Fusion preserves each model's original rank and "
        "score. Consensus, recommendations and top-5 follow-up order by "
        "consensus_rank (original RRF); stakeholder delivery reorders by "
        "eligibility_rank (rrf_score / eligible_model_count). Scores are "
        "ranking values, not calibrated probabilities."
    )
    try:
        availability = data.candidate_availability()
    except Exception as exc:
        st.info(f"Candidate configuration is unavailable: {exc}")
        return
    if not availability["candidate_db_exists"]:
        st.info(
            "Candidate-review DuckDB is unavailable. Build and audit the "
            "candidate review before opening rankings."
        )
        return

    models = data.candidate_models()
    dispositions = data.candidate_dispositions()
    if models.empty:
        st.warning("The candidate review has no synchronized model metadata.")
        return

    controls = st.columns([1.3, 2.1, 1, 1])
    with controls[0]:
        view = st.selectbox(
            "Ranking view",
            options=["consensus", "specialist"],
            format_func={
                "consensus": "RRF consensus",
                "specialist": "Specialist model",
            }.get,
        )
    role_by_result = models.set_index("result_id")["role"].astype(str).to_dict()
    result_id: str | None = None
    with controls[1]:
        if view == "specialist":
            result_id = st.selectbox(
                "Specialist model",
                options=models["result_id"].astype(str).tolist(),
                format_func=lambda value: role_by_result.get(value, value),
            )
        else:
            st.caption("Equal-weight RRF · k=60 · five frozen models")
    with controls[2]:
        limit = int(
            st.selectbox(
                "Top N",
                options=[20, 100, 500, 1_000, 10_000],
                index=2 if view == "consensus" else 3,
            )
        )
    with controls[3]:
        jaccard_k = int(
            st.selectbox(
                "Overlap K",
                options=[20, 100, 500, 1_000, 10_000],
                index=2,
            )
        )

    disposition_options = (
        dispositions["review_disposition"].astype(str).tolist()
        if not dispositions.empty
        else []
    )
    if view == "consensus":
        with st.expander("Filters", expanded=False):
            filter_columns = st.columns([2, 1, 1, 1])
            selected_dispositions = tuple(
                filter_columns[0].multiselect(
                    "SIMBAD disposition",
                    options=disposition_options,
                    default=[],
                )
            )
            min_support = int(
                filter_columns[1].slider(
                    "Minimum model support",
                    min_value=1,
                    max_value=max(1, len(models)),
                    value=1,
                )
            )
            followup_only = filter_columns[2].toggle(
                "Follow-up eligible",
                value=False,
                help="Exclude known WR and unequivocal catalogued non-WR.",
            )
            require_halpha = filter_columns[3].toggle(
                "Hα available",
                value=False,
            )
    else:
        selected_dispositions = ()
        min_support = 1
        followup_only = False
        require_halpha = False

    rankings = data.candidate_rankings(
        view=view,
        result_id=result_id,
        limit=limit,
        dispositions=selected_dispositions,
        min_model_support=min_support,
        followup_only=followup_only,
        require_halpha=require_halpha,
    )
    if rankings.empty:
        st.warning("No candidates match the selected ranking view and filters.")
        return

    dispositions_series = rankings.get(
        "review_disposition",
        pd.Series(index=rankings.index, dtype="string"),
    ).astype("string")
    support = pd.to_numeric(
        rankings.get("model_support", pd.Series(index=rankings.index)),
        errors="coerce",
    )
    ui.kpi_row(
        [
            ("Visible candidates", f"{len(rankings):,}", None),
            (
                "Supported by all models",
                f"{int(support.eq(len(models)).sum()):,}",
                f"of {len(models)} models",
            ),
            (
                "No SIMBAD match",
                f"{int(dispositions_series.eq('no_exact_match').sum()):,}",
                "Absence does not prove novelty",
            ),
            (
                "Known exclusions",
                f"{int(dispositions_series.isin(['known_wr', 'catalogued_non_wr']).sum()):,}",
                "Retained for audit",
            ),
        ]
    )

    if view == "consensus":
        left, right = st.columns(2, gap="large")
        with left:
            st.subheader("Support across models")
            chart = candidate_plots.support_distribution(rankings)
            if chart is not None:
                st.altair_chart(chart)
        with right:
            st.subheader("Consensus vs best individual rank")
            chart = candidate_plots.rank_agreement(rankings)
            if chart is not None:
                st.altair_chart(chart)

    st.subheader("Cross-model overlap")
    jaccard = data.candidate_jaccard(jaccard_k)
    heatmap = candidate_plots.jaccard_heatmap(jaccard)
    if heatmap is None:
        st.info("Cross-model overlap is unavailable.")
    else:
        st.altair_chart(heatmap)
        st.caption(
            f"Jaccard intersection over union among each model's top "
            f"{jaccard_k:,}; diagonal values are 100%."
        )

    st.subheader("Ranked candidates")
    selected_source = _ranking_table(rankings, view=view)
    if selected_source is not None:
        st.session_state[ui.SELECTED_CANDIDATE_KEY] = selected_source
        detail, action = st.columns([4, 1])
        with detail:
            evidence = data.candidate_model_evidence(selected_source)
            if not evidence.empty:
                st.caption(
                    f"Gaia DR3 {selected_source} · present in "
                    f"{len(evidence)} specialist top-10,000 lists."
                )
                st.dataframe(
                    evidence[
                        [
                            "role",
                            "dataset_variant",
                            "model_rank",
                            "score",
                            "rrf_contribution",
                        ]
                    ],
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "role": st.column_config.TextColumn("Model role"),
                        "dataset_variant": st.column_config.TextColumn("Variant"),
                        "model_rank": st.column_config.NumberColumn(
                            "Original rank",
                            format="%d",
                        ),
                        "score": st.column_config.NumberColumn(
                            "Ranking score",
                            format="%.5f",
                        ),
                        "rrf_contribution": st.column_config.NumberColumn(
                            "RRF contribution",
                            format="%.6f",
                        ),
                    },
                )
        with action:
            page = st.session_state.get("_candidate_review_page")
            if page is not None and st.button(
                "Open candidate",
                icon=":material/open_in_new:",
                type="primary",
                width="stretch",
            ):
                st.switch_page(
                    page,
                    query_params={"source_id": str(selected_source)},
                )


def _ranking_table(rankings: pd.DataFrame, *, view: str) -> int | None:
    if view == "specialist":
        columns = [
            "source_id",
            "rank",
            "score",
            "consensus_rank",
            "model_support",
            "gaia_designation",
            "simbad_main_id",
            "review_disposition",
        ]
    else:
        columns = [
            "source_id",
            "rank",
            "rrf_score",
            "model_support",
            "variant_support",
            "estimator_support",
            "best_model_rank",
            "gaia_designation",
            "simbad_main_id",
            "review_disposition",
            "G",
            "BP_RP",
            "J_K",
            "W1_W2",
            "ruwe",
            "halpha_ew",
            "espels_class",
        ]
    visible = [column for column in columns if column in rankings.columns]
    table = rankings[visible].reset_index(drop=True)
    event = st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        height=460,
        on_select="rerun",
        selection_mode="single-row",
        key=f"candidate_ranking_table_{view}",
        column_config={
            "source_id": None,
            "rank": st.column_config.NumberColumn("#", format="%d", width="small"),
            "score": st.column_config.NumberColumn("Ranking score", format="%.5f"),
            "rrf_score": st.column_config.NumberColumn("RRF", format="%.6f"),
            "model_support": st.column_config.NumberColumn("Models", format="%d"),
            "variant_support": st.column_config.NumberColumn("Variants", format="%d"),
            "estimator_support": st.column_config.NumberColumn("Estimators", format="%d"),
            "best_model_rank": st.column_config.NumberColumn("Best rank", format="%d"),
            "consensus_rank": st.column_config.NumberColumn("RRF rank", format="%d"),
            "gaia_designation": st.column_config.TextColumn("Gaia"),
            "simbad_main_id": st.column_config.TextColumn("SIMBAD"),
            "review_disposition": st.column_config.TextColumn("Disposition"),
            "G": st.column_config.NumberColumn("G", format="%.3f"),
            "BP_RP": st.column_config.NumberColumn("BP-RP", format="%.3f"),
            "J_K": st.column_config.NumberColumn("J-Ks", format="%.3f"),
            "W1_W2": st.column_config.NumberColumn("W1-W2", format="%.3f"),
            "ruwe": st.column_config.NumberColumn("RUWE", format="%.3f"),
            "halpha_ew": st.column_config.NumberColumn("Hα EW", format="%.3f"),
            "espels_class": st.column_config.TextColumn("ESP-ELS"),
        },
    )
    selected = event.selection.rows if event.selection else []
    if selected:
        return int(table.iloc[selected[0]]["source_id"])
    current = st.session_state.get(ui.SELECTED_CANDIDATE_KEY)
    if current is not None and table["source_id"].eq(int(current)).any():
        return int(current)
    return None
