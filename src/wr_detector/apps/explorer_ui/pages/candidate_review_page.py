"""Case-style scientific review of prediction-pool candidates."""

from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from wr_detector.apps.explorer_ui import candidate_plots, data, ui


VIEW_MODES = {
    "Top 20 follow-up": {
        "limit": 20,
        "dispositions": (),
        "followup_only": True,
    },
    "Follow-up shortlist": {
        "limit": 500,
        "dispositions": (),
        "followup_only": True,
    },
    "No SIMBAD match": {
        "limit": 500,
        "dispositions": ("no_exact_match",),
        "followup_only": False,
    },
    "Emission / ambiguous": {
        "limit": 500,
        "dispositions": ("emission_or_ambiguous",),
        "followup_only": False,
    },
    "Known references": {
        "limit": 500,
        "dispositions": ("known_wr", "catalogued_non_wr"),
        "followup_only": False,
    },
    "All top 500": {
        "limit": 500,
        "dispositions": (),
        "followup_only": False,
    },
}


def render() -> None:
    st.title("Candidate review")
    st.caption(
        "Case-level review of the frozen RRF shortlist. SIMBAD, Gaia Hα and "
        "ESP-ELS annotate the ranking; they did not define or re-rank it."
    )
    try:
        availability = data.candidate_availability()
    except Exception as exc:
        st.info(f"Candidate configuration is unavailable: {exc}")
        return
    if not availability["candidate_db_exists"]:
        st.info("Candidate-review DuckDB is unavailable.")
        return

    controls = st.columns([1.6, 1, 1, 1])
    with controls[0]:
        mode = st.selectbox("View", options=list(VIEW_MODES))
    with controls[1]:
        min_support = int(
            st.slider(
                "Minimum model support",
                min_value=1,
                max_value=max(1, len(data.candidate_models())),
                value=1,
            )
        )
    with controls[2]:
        min_parallax_over_error = float(
            st.select_slider(
                "Minimum parallax/error",
                options=[0.0, 1.0, 2.0, 3.0, 5.0],
                value=2.0,
            )
        )
    with controls[3]:
        max_distance_kpc = float(
            st.slider(
                "Maximum distance (kpc)",
                min_value=5,
                max_value=25,
                value=15,
                step=1,
            )
        )

    definition = VIEW_MODES[mode]
    candidates = data.candidate_rankings(
        view="consensus",
        limit=int(definition["limit"]),
        dispositions=tuple(definition["dispositions"]),
        min_model_support=min_support,
        followup_only=bool(definition["followup_only"]),
    )
    if candidates.empty:
        st.warning("No candidates match this review view.")
        return

    requested = st.query_params.get("source_id")
    if requested is not None:
        try:
            requested_id = int(requested)
        except (TypeError, ValueError):
            requested_id = None
        if requested_id is not None and candidates["source_id"].eq(requested_id).any():
            st.session_state[ui.SELECTED_CANDIDATE_KEY] = requested_id

    plot_frame = data.candidate_plot_frame(
        min_parallax_over_error=min_parallax_over_error,
        max_distance_kpc=max_distance_kpc,
    )
    ui.kpi_row(
        [
            ("Candidates in view", f"{len(candidates):,}", None),
            (
                "Five-model support",
                f"{int(candidates['model_support'].eq(5).sum()):,}",
                None,
            ),
            (
                "No SIMBAD match",
                f"{int(candidates['review_disposition'].eq('no_exact_match').sum()):,}",
                "Not proof of novelty",
            ),
            (
                "Hα available",
                f"{int(candidates['halpha_ew'].notna().sum()):,}",
                "Auxiliary evidence",
            ),
        ]
    )
    _candidate_review_fragment(candidates, plot_frame)

    with st.expander("Review lineage", expanded=False):
        runs = data.candidate_review_runs()
        st.dataframe(runs, hide_index=True, width="stretch")
        st.caption(
            "A live SIMBAD refresh is intentionally unavailable in the app. "
            "Use `wr-detector build-prediction-pool-candidates "
            "--config configs/prediction_pool_candidates.yaml "
            "--refresh-simbad` and re-audit the snapshot."
        )


@st.fragment
def _candidate_review_fragment(
    candidates: pd.DataFrame,
    plot_frame: pd.DataFrame,
) -> None:
    selected_id = _selected_candidate_id(candidates)
    table_col, detail_col = st.columns([1.35, 1], gap="large")
    with table_col:
        table_selection = _candidate_table(candidates)
        if table_selection is not None:
            selected_id = table_selection
            st.session_state[ui.SELECTED_CANDIDATE_KEY] = selected_id
    with detail_col:
        selected_id = _candidate_detail(candidates, selected_id)

    st.subheader("Candidate evidence in the five-model stack")
    evidence = data.candidate_model_evidence(selected_id)
    evidence_col, ladder_col = st.columns([1.35, 1], gap="large")
    with evidence_col:
        if evidence.empty:
            st.info("This source is absent from all specialist top-10,000 lists.")
        else:
            st.dataframe(
                evidence[
                    [
                        "role",
                        "dataset_variant",
                        "model",
                        "sampler",
                        "model_rank",
                        "score",
                        "rrf_contribution",
                    ]
                ],
                hide_index=True,
                width="stretch",
                column_config={
                    "role": st.column_config.TextColumn("Role"),
                    "dataset_variant": st.column_config.TextColumn("Variant"),
                    "model": st.column_config.TextColumn("Estimator"),
                    "sampler": st.column_config.TextColumn("Sampler"),
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
            st.caption(
                "Original model scores remain separate and are not averaged."
            )
    with ladder_col:
        chart = candidate_plots.model_rank_ladder(evidence)
        if chart is None:
            st.info("No model-rank ladder for this source.")
        else:
            st.altair_chart(chart)

    st.subheader("Photometric context")
    left, right = st.columns(2, gap="large")
    with left:
        chart = candidate_plots.candidate_photometric(
            plot_frame,
            selected_source_id=selected_id,
            x="BP_RP",
            y="G",
        )
        if chart is None:
            st.info("Gaia BP-RP / G coverage is unavailable.")
        else:
            st.altair_chart(chart)
            st.caption("Gaia intra-mission color-magnitude diagnostic.")
    with right:
        chart = candidate_plots.candidate_photometric(
            plot_frame,
            selected_source_id=selected_id,
            x="J_H",
            y="H_K",
        )
        if chart is None:
            st.info("2MASS color-color coverage is unavailable.")
        else:
            st.altair_chart(chart)
            st.caption("2MASS intra-mission color-color diagnostic.")

    st.subheader("Spatial context")
    sky_col, plane_col = st.columns(2, gap="large")
    with sky_col:
        chart = candidate_plots.candidate_mollweide(
            plot_frame,
            selected_source_id=selected_id,
        )
        if chart is None:
            st.info("Galactic coordinates are unavailable.")
        else:
            st.altair_chart(chart)
        st.caption("Mollweide: l=0 centered and longitude increases left.")
    with plane_col:
        chart = candidate_plots.candidate_galactic_plane(
            plot_frame,
            selected_source_id=selected_id,
        )
        if chart is None:
            st.info("No source meets the selected parallax-quality gates.")
        else:
            st.altair_chart(chart)
        plotted = (
            int(plot_frame["distance_plotted"].fillna(False).sum())
            if "distance_plotted" in plot_frame.columns
            else 0
        )
        st.caption(
            f"{plotted:,} top-500 sources plotted. Distances use "
            "d=1/parallax and are diagnostic only."
        )


def _selected_candidate_id(candidates: pd.DataFrame) -> int:
    current = st.session_state.get(ui.SELECTED_CANDIDATE_KEY)
    available = candidates["source_id"].astype("int64").tolist()
    if current not in available:
        current = int(available[0])
        st.session_state[ui.SELECTED_CANDIDATE_KEY] = current
    return int(current)


def _candidate_table(candidates: pd.DataFrame) -> int | None:
    columns = [
        "source_id",
        "rank",
        "simbad_main_id",
        "review_disposition",
        "model_support",
        "best_model_rank",
        "G",
        "BP_RP",
        "J_K",
        "W1_W2",
        "halpha_ew",
        "espels_class",
        "ruwe",
    ]
    table = candidates[
        [column for column in columns if column in candidates.columns]
    ].reset_index(drop=True)
    event = st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        height=480,
        on_select="rerun",
        selection_mode="single-row",
        key="prediction_pool_candidate_table",
        column_config={
            "source_id": None,
            "rank": st.column_config.NumberColumn("#", format="%d", width="small"),
            "simbad_main_id": st.column_config.TextColumn("Object"),
            "review_disposition": st.column_config.TextColumn("Disposition"),
            "model_support": st.column_config.NumberColumn("Models", format="%d"),
            "best_model_rank": st.column_config.NumberColumn("Best rank", format="%d"),
            "G": st.column_config.NumberColumn("G", format="%.3f"),
            "BP_RP": st.column_config.NumberColumn("BP-RP", format="%.3f"),
            "J_K": st.column_config.NumberColumn("J-Ks", format="%.3f"),
            "W1_W2": st.column_config.NumberColumn("W1-W2", format="%.3f"),
            "halpha_ew": st.column_config.NumberColumn("Hα EW", format="%.3f"),
            "espels_class": st.column_config.TextColumn("ESP-ELS"),
            "ruwe": st.column_config.NumberColumn("RUWE", format="%.3f"),
        },
    )
    selected = event.selection.rows if event.selection else []
    return int(table.iloc[selected[0]]["source_id"]) if selected else None


def _candidate_detail(candidates: pd.DataFrame, source_id: int) -> int:
    candidates = candidates.reset_index(drop=True)
    matches = candidates.index[candidates["source_id"].eq(source_id)].tolist()
    index = matches[0] if matches else 0
    row = candidates.iloc[index]
    source_id = int(row["source_id"])
    st.session_state[ui.SELECTED_CANDIDATE_KEY] = source_id

    previous, following = st.columns(2)
    if previous.button(
        "Previous",
        icon=":material/chevron_left:",
        disabled=index <= 0,
        width="stretch",
    ):
        source_id = int(candidates.iloc[index - 1]["source_id"])
        st.session_state[ui.SELECTED_CANDIDATE_KEY] = source_id
        st.rerun(scope="fragment")
    if following.button(
        "Next",
        icon=":material/chevron_right:",
        disabled=index >= len(candidates) - 1,
        width="stretch",
    ):
        source_id = int(candidates.iloc[index + 1]["source_id"])
        st.session_state[ui.SELECTED_CANDIDATE_KEY] = source_id
        st.rerun(scope="fragment")
    st.caption(f"Candidate {index + 1} of {len(candidates)}")

    name = _text(row.get("simbad_main_id")) or _text(
        row.get("gaia_designation")
    )
    badge = ui.candidate_disposition_badge(row.get("review_disposition"))
    st.markdown(f"### {name} {badge}")
    st.caption(f"Gaia DR3 `{source_id}`")
    ui.kpi_row(
        [
            ("Consensus rank", f"{int(row['consensus_rank']):,}", None),
            ("Model support", f"{int(row['model_support'])} / 5", None),
            ("Best model rank", f"{int(row['best_model_rank']):,}", None),
        ]
    )

    links = [
        f"[SIMBAD]({_text(row.get('simbad_url'))})",
        f"[Gaia Archive]({_text(row.get('gaia_url'))})",
    ]
    if _finite(row.get("ra")) and _finite(row.get("dec")):
        ra = float(row["ra"])
        dec = float(row["dec"])
        links.extend(
            [
                "[Aladin Lite]("
                f"https://aladin.cds.unistra.fr/AladinLite/?target={ra}%20{dec}"
                "&fov=0.05&survey=P%2FDSS2%2Fcolor)",
                f"[ESASky](https://sky.esa.int/esasky/?target={ra}%20{dec}&fov=0.1)",
            ]
        )
    st.markdown(" | ".join(links))

    identity = pd.DataFrame(
        [
            {
                "SIMBAD type": row.get("simbad_main_type"),
                "Other types": row.get("simbad_other_types"),
                "Spectral type": row.get("simbad_sp_type"),
                "Match method": row.get("match_method"),
                "Matches": row.get("match_count"),
                "Separation [arcsec]": row.get("separation_arcsec"),
            }
        ]
    )
    st.dataframe(identity, hide_index=True, width="stretch")
    if _text(row.get("review_comment")):
        st.info(str(row["review_comment"]))

    _source_measurements(row)
    with st.expander("Gaia Archive query", expanded=False):
        st.code(_text(row.get("gaia_adql")), language="sql")
    return source_id


def _source_measurements(row: pd.Series) -> None:
    tabs = st.tabs(
        [
            "Photometry",
            "Counterparts & quality",
            "Auxiliary spectral evidence",
        ]
    )
    with tabs[0]:
        _one_row_table(
            row,
            [
                "G",
                "BP",
                "RP",
                "J",
                "H",
                "Ks",
                "W1",
                "W2",
                "BP_RP",
                "J_H",
                "J_K",
                "H_K",
                "W1_W2",
                "ag_gspphot",
            ],
        )
    with tabs[1]:
        _one_row_table(
            row,
            [
                "parallax",
                "parallax_error",
                "parallax_over_error",
                "ruwe",
                "astrometric_excess_noise",
                "phot_bp_n_blended_transits",
                "phot_rp_n_blended_transits",
                "tmass_id",
                "tmass_quality",
                "tmass_angular_distance",
                "wise_id",
                "wise_quality",
                "wise_angular_distance",
                "wise_cc_flags",
                "wise_ext_flag",
                "phot_variable_flag",
            ],
        )
    with tabs[2]:
        _one_row_table(
            row,
            [
                "halpha_ew",
                "halpha_ew_error",
                "halpha_ew_flag",
                "espels_class",
                "espels_class_flag",
                "espels_wn_probability",
                "espels_wc_probability",
                "espels_be_probability",
                "espels_pne_probability",
            ],
        )
        st.caption(
            "Hα and ESP-ELS are auxiliary evidence. Missing values do not "
            "remove a candidate and these fields are not hidden RRF inputs."
        )


def _one_row_table(row: pd.Series, fields: list[str]) -> None:
    values = {
        field: row.get(field)
        for field in fields
        if field in row.index and not _missing(row.get(field))
    }
    if not values:
        st.info("No values available.")
        return
    st.dataframe(pd.DataFrame([values]), hide_index=True, width="stretch")


def _text(value: object) -> str:
    return "" if _missing(value) else str(value).strip()


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False
