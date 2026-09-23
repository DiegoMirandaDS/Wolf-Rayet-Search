"""Reproducible candidate shortlists from audited prediction-pool scores."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
from typing import Any, Callable, Iterable, Mapping

import duckdb
import numpy as np
import pandas as pd

from wr_detector.config import load_yaml, resolve_path
from wr_detector.pipelines.prediction_pool_scoring import (
    file_sha256,
    load_prediction_scoring_config,
)


REQUIRED_REVIEW_TABLES = {
    "candidate_review_runs",
    "candidate_models",
    "candidate_model_ranks",
    "candidate_consensus",
    "candidate_model_eligibility",
    "candidate_source_features",
    "candidate_simbad_matches",
    "candidate_review",
    "candidate_recommendations",
}

POOL_FEATURE_COLUMNS = (
    "source_id",
    "gaia_designation",
    "ra",
    "dec",
    "galactic_l",
    "galactic_b",
    "G",
    "BP",
    "RP",
    "J",
    "H",
    "Ks",
    "W1",
    "W2",
    "G_BP",
    "G_RP",
    "BP_RP",
    "J_H",
    "J_K",
    "H_K",
    "W1_W2",
    "parallax",
    "parallax_error",
    "parallax_over_error",
    "pmra",
    "pmdec",
    "ruwe",
    "astrometric_excess_noise",
    "visibility_periods_used",
    "phot_bp_rp_excess_factor",
    "phot_bp_n_contaminated_transits",
    "phot_bp_n_blended_transits",
    "phot_rp_n_contaminated_transits",
    "phot_rp_n_blended_transits",
    "duplicated_source",
    "non_single_star",
    "phot_variable_flag",
    "tmass_id",
    "tmass_quality",
    "tmass_angular_distance",
    "tmass_candidate_count",
    "tmass_number_of_neighbours",
    "tmass_number_of_mates",
    "wise_id",
    "wise_quality",
    "wise_angular_distance",
    "wise_number_of_neighbours",
    "wise_number_of_mates",
    "wise_cc_flags",
    "wise_ext_flag",
    "wise_var_flag",
    "ag_gspphot",
    "ebpminrp_gspphot",
    "teff_gspphot",
    "logg_gspphot",
    "mh_gspphot",
    "halpha_ew",
    "halpha_ew_error",
    "halpha_ew_flag",
    "espels_class",
    "espels_class_flag",
    "espels_wc_probability",
    "espels_wn_probability",
    "espels_be_probability",
    "espels_pne_probability",
    "compatible_variant_mask",
    "compatible_variant_count",
    "source_hash_v1",
)

GENERIC_SIMBAD_TYPES = {
    "",
    "*",
    "star",
    "unknown",
    "pm*",
    "v*",
    "ir",
    "x",
}
EMISSION_AMBIGUOUS_TOKENS = (
    "em*",
    "emission",
    "be*",
    "bestar",
    "yso",
    "tt*",
    "ttauri",
    "herbig",
    "pne",
    "planetarynebula",
    "blue",
    "hmx",
)
KNOWN_NON_WR_TOKENS = (
    "galaxy",
    "qso",
    "agn",
    "white dwarf",
    "wd*",
    "brown dwarf",
    "rgb*",
    "agb*",
    "carbon",
    "c*",
    "supernova",
    "snr",
    "hii",
)


def load_candidate_review_config(
    config_path: str | Path,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    required = {"scoring_config", "review", "ranking", "simbad", "outputs"}
    missing = sorted(required - set(config))
    if missing:
        raise KeyError(
            f"Prediction candidate config is missing keys: {missing}"
        )
    models = config["review"].get("models", [])
    if not models:
        raise ValueError("Candidate review requires explicit model result_ids.")
    result_ids = [str(item["result_id"]) for item in models]
    if len(result_ids) != len(set(result_ids)):
        raise ValueError("Candidate model result_ids must be unique.")
    return config


def reciprocal_rank_consensus(
    ranks: pd.DataFrame,
    *,
    rrf_k: float = 60.0,
) -> pd.DataFrame:
    """Fuse deterministic per-model ranks without treating scores as probabilities."""
    required = {
        "source_id",
        "result_id",
        "dataset_variant",
        "model",
        "model_rank",
    }
    missing = sorted(required - set(ranks.columns))
    if missing:
        raise KeyError(f"Candidate ranks are missing columns: {missing}")
    if ranks.duplicated(["result_id", "source_id"]).any():
        raise ValueError("Per-model candidate ranks contain duplicates.")
    if float(rrf_k) <= 0:
        raise ValueError("rrf_k must be positive.")

    frame = ranks.copy()
    frame["rrf_contribution"] = 1.0 / (
        float(rrf_k) + pd.to_numeric(frame["model_rank"])
    )
    consensus = (
        frame.groupby("source_id", as_index=False)
        .agg(
            rrf_score=("rrf_contribution", "sum"),
            model_support=("result_id", "nunique"),
            variant_support=("dataset_variant", "nunique"),
            estimator_support=("model", "nunique"),
            best_model_rank=("model_rank", "min"),
            mean_model_rank=("model_rank", "mean"),
        )
        .sort_values(
            [
                "rrf_score",
                "model_support",
                "variant_support",
                "best_model_rank",
                "source_id",
            ],
            ascending=[False, False, False, True, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    consensus.insert(0, "consensus_rank", np.arange(1, len(consensus) + 1))
    return consensus


def eligibility_aware_consensus(
    consensus: pd.DataFrame,
    model_ranks: pd.DataFrame,
    model_metadata: pd.DataFrame,
    source_eligibility: pd.DataFrame,
    *,
    rrf_k: float = 60.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add model-opportunity-aware ranks without replacing the original RRF.

    A missing per-model rank has two scientifically different meanings.  The
    source can fail that model's exact locus, photometric-quality or
    astrometric rule, or it can pass all of them but rank below the retained
    per-model budget. The returned long table preserves those distinctions and
    the consensus divides accumulated RRF evidence only by the number of models
    that could evaluate the source.
    """
    required_consensus = {"source_id", "rrf_score", "model_support"}
    required_ranks = {"source_id", "result_id", "model_rank", "score"}
    required_models = {"result_id", "variant_bit"}
    required_sources = {
        "source_id",
        "exact_locus_variant_mask",
        "photometry_variant_mask",
        "astrometry_variant_mask",
        "compatible_variant_mask",
    }
    checks = [
        ("consensus", required_consensus, consensus),
        ("model ranks", required_ranks, model_ranks),
        ("model metadata", required_models, model_metadata),
        ("source eligibility", required_sources, source_eligibility),
    ]
    for label, required, frame in checks:
        missing = sorted(required - set(frame.columns))
        if missing:
            raise KeyError(f"{label} is missing columns: {missing}")
    if source_eligibility["source_id"].duplicated().any():
        raise ValueError("Source eligibility contains duplicate source_ids.")
    if model_metadata["result_id"].duplicated().any():
        raise ValueError("Model metadata contains duplicate result_ids.")

    mask_columns = [
        "exact_locus_variant_mask",
        "photometry_variant_mask",
        "astrometry_variant_mask",
        "compatible_variant_mask",
    ]
    sources = source_eligibility[["source_id", *mask_columns]].copy()
    models = model_metadata.copy()
    sources["_join"] = 1
    models["_join"] = 1
    opportunities = sources.merge(models, on="_join", validate="many_to_many")
    opportunities = opportunities.drop(columns="_join")
    bits = np.left_shift(
        np.uint64(1),
        opportunities["variant_bit"].astype("uint64").to_numpy(),
    )
    for mask_column, keep_column in [
        ("exact_locus_variant_mask", "locus_keep"),
        ("photometry_variant_mask", "photometry_keep"),
        ("astrometry_variant_mask", "astrometry_keep"),
        ("compatible_variant_mask", "eligible"),
    ]:
        masks = opportunities[mask_column].astype("uint64").to_numpy()
        opportunities[keep_column] = (masks & bits) != 0
    expected_eligible = (
        opportunities["locus_keep"]
        & opportunities["photometry_keep"]
        & opportunities["astrometry_keep"]
    )
    if not opportunities["eligible"].eq(expected_eligible).all():
        raise ValueError("Compatible-variant bit disagrees with component masks.")

    rank_columns = [
        column
        for column in [
            "source_id",
            "result_id",
            "model_rank",
            "score",
            "predicted",
            "threshold",
        ]
        if column in model_ranks.columns
    ]
    opportunities = opportunities.merge(
        model_ranks[rank_columns],
        on=["source_id", "result_id"],
        how="left",
        validate="one_to_one",
    )
    ranked = opportunities["model_rank"].notna()
    if (ranked & ~opportunities["eligible"]).any():
        raise ValueError("A ranked candidate is ineligible for its model variant.")
    opportunities["model_status"] = np.select(
        [ranked, opportunities["eligible"]],
        ["ranked_top_10000", "eligible_below_top_10000"],
        default="ineligible_variant",
    )
    opportunities["eligibility_reason"] = np.where(
        opportunities["eligible"],
        "passed_all_variant_filters",
        opportunities.apply(_eligibility_failure_reason, axis=1),
    )
    opportunities["rrf_contribution"] = np.where(
        ranked,
        1.0 / (float(rrf_k) + pd.to_numeric(opportunities["model_rank"])),
        0.0,
    )

    support = (
        opportunities.groupby("source_id", as_index=False)
        .agg(
            eligible_model_count=("eligible", "sum"),
            ranked_model_count=("model_rank", "count"),
        )
    )
    if (support["eligible_model_count"] <= 0).any():
        raise ValueError("Every consensus source must be eligible for a model.")
    enriched = consensus.merge(
        support,
        on="source_id",
        how="left",
        validate="one_to_one",
    )
    if not enriched["ranked_model_count"].eq(enriched["model_support"]).all():
        raise ValueError("Ranked-model support disagrees with original RRF support.")
    enriched["eligible_support_fraction"] = (
        enriched["ranked_model_count"] / enriched["eligible_model_count"]
    )
    enriched["eligibility_normalized_rrf"] = (
        enriched["rrf_score"] / enriched["eligible_model_count"]
    )
    eligibility_order = enriched.sort_values(
        [
            "eligibility_normalized_rrf",
            "eligible_support_fraction",
            "model_support",
            "best_model_rank",
            "source_id",
        ],
        ascending=[False, False, False, True, True],
        kind="mergesort",
    ).index
    ranks = pd.Series(
        np.arange(1, len(enriched) + 1),
        index=eligibility_order,
        dtype="int64",
    )
    enriched["eligibility_rank"] = ranks.sort_index().to_numpy()
    return enriched, opportunities


def classify_simbad_match(row: Mapping[str, Any]) -> str:
    """Classify catalog context conservatively without hiding raw SIMBAD types."""
    method = str(row.get("match_method") or "")
    if method == "position_ambiguous":
        return "ambiguous_positional_match"
    if not bool(row.get("match_found", False)):
        return "no_exact_match"
    text = " ".join(
        str(row.get(key) or "")
        for key in ["simbad_main_type", "simbad_other_types", "simbad_sp_type"]
    ).lower()
    if re.search(r"wolf\s*-?\s*rayet|\bwr\*|\bwn\b|\bwc\b|\bwo\b", text):
        return "known_wr"
    if any(token in text for token in EMISSION_AMBIGUOUS_TOKENS):
        return "emission_or_ambiguous"
    if any(token in text for token in KNOWN_NON_WR_TOKENS):
        return "catalogued_non_wr"
    main_type = str(row.get("simbad_main_type") or "").strip().lower()
    if main_type in GENERIC_SIMBAD_TYPES:
        return "generic_or_uninformative"
    return "emission_or_ambiguous"


def candidate_review_comment(row: Mapping[str, Any]) -> str:
    """Build a compact, transparent scientific-review note."""
    disposition = str(row.get("review_disposition") or "no_exact_match")
    support = _optional_int(row.get("model_support"))
    best_rank = _optional_int(row.get("best_model_rank"))
    pieces = [
        (
            f"Apoyo de {support} modelos; mejor rango individual "
            f"{best_rank}."
        )
    ]
    context = {
        "known_wr": "SIMBAD lo clasifica como WR conocido.",
        "catalogued_non_wr": (
            "SIMBAD contiene una clasificación no-WR inequívoca."
        ),
        "emission_or_ambiguous": (
            "SIMBAD aporta un tipo emisivo o ambiguo; no basta para "
            "confirmar ni descartar WR."
        ),
        "generic_or_uninformative": (
            "SIMBAD aporta solo una clasificación genérica."
        ),
        "no_exact_match": (
            "Sin coincidencia SIMBAD exacta ni posicional única a 1″; "
            "esto no demuestra novedad."
        ),
        "ambiguous_positional_match": (
            "El fallback posicional a 1″ es ambiguo."
        ),
    }
    pieces.append(context.get(disposition, "Contexto SIMBAD no concluyente."))
    espels_class = _clean_text(row.get("espels_class"))
    halpha = _optional_float(row.get("halpha_ew"))
    if espels_class:
        pieces.append(f"ESP-ELS: {espels_class}.")
    if halpha is not None:
        pieces.append(f"Gaia Hα EW={halpha:.3f}.")
    ruwe = _optional_float(row.get("ruwe"))
    if ruwe is not None and ruwe > 1.4:
        pieces.append(f"RUWE elevado ({ruwe:.2f}); revisar astrometría.")
    wise_flags = _clean_text(row.get("wise_cc_flags"))
    if wise_flags and any(character != "0" for character in wise_flags):
        pieces.append(
            f"WISE cc_flags={wise_flags}; revisar contaminación visual."
        )
    pieces.append("Requiere inspección de SED/imágenes y espectroscopía.")
    return " ".join(pieces)


def build_prediction_pool_candidates(
    config_path: str | Path,
    *,
    refresh_simbad: bool = False,
    skip_simbad: bool = False,
    simbad_query: Callable[[pd.DataFrame, Mapping[str, Any]], pd.DataFrame]
    | None = None,
) -> dict[str, Any]:
    """Build an immutable, source-backed candidate-review database and exports."""
    config_path = resolve_path(config_path)
    config = load_candidate_review_config(config_path)
    scoring_config_path = resolve_path(config["scoring_config"])
    scoring_config = load_prediction_scoring_config(scoring_config_path)
    pool_config_path = resolve_path(scoring_config["pool_config"])
    pool_config = load_yaml(pool_config_path)
    scoring_db = resolve_path(scoring_config["output_db"])
    pool_db = resolve_path(pool_config["output_db"])
    if not scoring_db.exists():
        raise FileNotFoundError(scoring_db)
    if not pool_db.exists():
        raise FileNotFoundError(pool_db)

    review = config["review"]
    ranking = config["ranking"]
    simbad_config = config["simbad"]
    outputs = config["outputs"]
    review_run_id = str(review["review_run_id"])
    scoring_run_id = str(review["scoring_run_id"])
    model_run_id = str(review["model_run_id"])
    selected_models = [dict(item) for item in review["models"]]
    result_ids = [str(item["result_id"]) for item in selected_models]
    per_model_limit = int(ranking.get("per_model_limit", 10_000))
    review_limit = int(ranking.get("review_limit", 500))
    deep_review_limit = int(ranking.get("deep_review_limit", 20))
    headline_limit = int(ranking.get("headline_limit", 5))
    rrf_k = float(ranking.get("rrf_k", 60.0))
    if min(
        per_model_limit,
        review_limit,
        deep_review_limit,
        headline_limit,
    ) <= 0:
        raise ValueError("Candidate ranking limits must be positive.")
    if not headline_limit <= deep_review_limit <= review_limit:
        raise ValueError(
            "Expected headline_limit <= deep_review_limit <= review_limit."
        )

    scoring_contract = _validate_scoring_contract(
        scoring_db,
        scoring_run_id=scoring_run_id,
        model_run_id=model_run_id,
        result_ids=result_ids,
    )
    model_metadata = _model_metadata(
        scoring_db,
        scoring_run_id=scoring_run_id,
        selected_models=selected_models,
    )
    candidate_models = _load_candidate_model_metrics(
        model_metadata,
        training_history_db=(
            resolve_path(config["training_history_db"])
            if config.get("training_history_db")
            else None
        ),
        model_run_id=model_run_id,
    )
    model_ranks = _load_model_ranks(
        scoring_db,
        scoring_run_id=scoring_run_id,
        model_metadata=model_metadata,
        per_model_limit=per_model_limit,
    )
    consensus = reciprocal_rank_consensus(model_ranks, rrf_k=rrf_k)
    source_eligibility = _load_pool_eligibility(
        pool_db,
        consensus["source_id"].astype("int64").tolist(),
    )
    consensus, model_eligibility = eligibility_aware_consensus(
        consensus,
        model_ranks,
        model_metadata,
        source_eligibility,
        rrf_k=rrf_k,
    )
    review_consensus = consensus.head(review_limit).copy()
    features = _load_pool_features(
        pool_db,
        review_consensus["source_id"].astype("int64").tolist(),
    )
    review_frame = review_consensus.merge(
        features, on="source_id", how="left", validate="one_to_one"
    )
    if review_frame["source_hash_v1"].isna().any():
        raise ValueError("Candidate feature join lost prediction-pool sources.")

    snapshot_path = resolve_path(
        str(simbad_config["snapshot_path"]).format(
            review_run_id=review_run_id
        )
    )
    if skip_simbad:
        simbad_matches = _empty_simbad_matches(review_frame)
        simbad_snapshot_sha256 = None
    else:
        if refresh_simbad or not snapshot_path.exists():
            query = simbad_query or query_simbad_candidates
            simbad_matches = query(review_frame, simbad_config)
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_snapshot = snapshot_path.with_suffix(
                snapshot_path.suffix + ".tmp"
            )
            simbad_matches.to_parquet(temporary_snapshot, index=False)
            temporary_snapshot.replace(snapshot_path)
        else:
            simbad_matches = pd.read_parquet(snapshot_path)
        simbad_matches = _complete_simbad_rows(
            review_frame[["source_id", "ra", "dec"]],
            simbad_matches,
        )
        simbad_snapshot_sha256 = file_sha256(snapshot_path)

    simbad_matches["review_disposition"] = simbad_matches.apply(
        lambda row: classify_simbad_match(row.to_dict()),
        axis=1,
    )
    review_frame = review_frame.merge(
        simbad_matches,
        on="source_id",
        how="left",
        validate="one_to_one",
    )
    review_frame["simbad_url"] = review_frame["source_id"].map(
        lambda value: (
            "https://simbad.cds.unistra.fr/simbad/sim-id?"
            f"Ident=Gaia+DR3+{int(value)}"
        )
    )
    review_frame["gaia_url"] = "https://gea.esac.esa.int/archive/"
    review_frame["gaia_adql"] = review_frame["source_id"].map(
        lambda value: (
            "SELECT * FROM gaiadr3.gaia_source "
            f"WHERE source_id = {int(value)}"
        )
    )
    review_frame["review_comment"] = review_frame.apply(
        lambda row: candidate_review_comment(row.to_dict()),
        axis=1,
    )
    followup = review_frame[
        ~review_frame["review_disposition"].isin(
            ["known_wr", "catalogued_non_wr"]
        )
    ].copy()
    followup = followup.sort_values(
        ["consensus_rank", "source_id"], kind="mergesort"
    ).reset_index(drop=True)
    followup.insert(0, "followup_rank", np.arange(1, len(followup) + 1))
    recommendations = followup.head(deep_review_limit).copy()

    output_db = resolve_path(outputs["database"])
    output_root = resolve_path(outputs["directory"])
    output_dir = output_root / review_run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_db.parent.mkdir(parents=True, exist_ok=True)
    run_row = pd.DataFrame(
        [
            {
                "review_run_id": review_run_id,
                "scoring_run_id": scoring_run_id,
                "model_run_id": model_run_id,
                "pool_build_id": scoring_contract["pool_build_id"],
                "config_path": str(config_path),
                "config_sha256": file_sha256(config_path),
                "selected_result_ids_json": json.dumps(result_ids),
                "per_model_limit": per_model_limit,
                "review_limit": review_limit,
                "deep_review_limit": deep_review_limit,
                "headline_limit": headline_limit,
                "rrf_k": rrf_k,
                "simbad_snapshot_path": (
                    str(snapshot_path) if not skip_simbad else None
                ),
                "simbad_snapshot_sha256": simbad_snapshot_sha256,
                "created_at": datetime.now(UTC),
            }
        ]
    )
    database_snapshot_path = _write_review_database(
        output_db,
        {
            "candidate_review_runs": run_row,
            "candidate_models": candidate_models,
            "candidate_model_ranks": model_ranks,
            "candidate_consensus": consensus,
            "candidate_model_eligibility": model_eligibility,
            "candidate_source_features": features,
            "candidate_simbad_matches": simbad_matches,
            "candidate_review": review_frame,
            "candidate_recommendations": recommendations,
        },
        history_dir=output_root / "_database_history",
    )

    export_paths = _write_exports(
        output_dir,
        model_ranks=model_ranks,
        review_frame=review_frame,
        recommendations=recommendations,
        simbad_matches=simbad_matches,
        headline_limit=headline_limit,
        deep_review_limit=deep_review_limit,
    )
    manifest = {
        "review_run_id": review_run_id,
        "scoring_run_id": scoring_run_id,
        "model_run_id": model_run_id,
        "pool_build_id": scoring_contract["pool_build_id"],
        "selected_models": selected_models,
        "ranking": {
            "method": "reciprocal_rank_fusion",
            "rrf_k": rrf_k,
            "per_model_limit": per_model_limit,
            "review_limit": review_limit,
            "deep_review_limit": deep_review_limit,
            "headline_limit": headline_limit,
        },
        "counts": {
            "models": int(len(candidate_models)),
            "model_rank_rows": int(len(model_ranks)),
            "consensus_rows": int(len(consensus)),
            "fully_supported_rows": int(
                consensus["eligible_support_fraction"].eq(1.0).sum()
            ),
            "review_rows": int(len(review_frame)),
            "simbad_matches": int(simbad_matches["match_found"].sum()),
            "recommendation_rows": int(len(recommendations)),
        },
        "simbad_snapshot_path": (
            str(snapshot_path) if not skip_simbad else None
        ),
        "simbad_snapshot_sha256": simbad_snapshot_sha256,
        "database_path": str(output_db),
        "database_snapshot_path": str(database_snapshot_path),
        "database_sha256": file_sha256(database_snapshot_path),
        "exports": export_paths,
        "written_at": datetime.now(UTC).isoformat(),
    }
    manifest_path = output_dir / "manifest.json"
    _atomic_write_json(manifest_path, manifest)
    audit = audit_prediction_pool_candidates(
        config_path,
        review_run_id=review_run_id,
    )
    if not audit["ok"]:
        raise RuntimeError(
            "Candidate review failed its audit: "
            + "; ".join(audit["errors"])
        )
    return {
        **manifest,
        "manifest_path": str(manifest_path),
        "audit": audit,
    }


def audit_prediction_pool_candidates(
    config_path: str | Path,
    *,
    review_run_id: str | None = None,
) -> dict[str, Any]:
    config = load_candidate_review_config(config_path)
    expected_run_id = review_run_id or str(
        config["review"]["review_run_id"]
    )
    db_path = resolve_path(config["outputs"]["database"])
    errors: list[str] = []
    if not db_path.exists():
        raise FileNotFoundError(db_path)
    with duckdb.connect(str(db_path), read_only=True) as con:
        tables = {
            str(row[0])
            for row in con.execute("SHOW TABLES").fetchall()
        }
        missing_tables = sorted(REQUIRED_REVIEW_TABLES - tables)
        if missing_tables:
            errors.append(f"missing_tables={missing_tables}")
            return {
                "review_run_id": expected_run_id,
                "database_path": str(db_path),
                "ok": False,
                "errors": errors,
            }
        run = con.execute(
            """
            SELECT selected_result_ids_json, per_model_limit, review_limit,
                   deep_review_limit, headline_limit
            FROM candidate_review_runs
            WHERE review_run_id=?
            """,
            [expected_run_id],
        ).fetchone()
        if run is None:
            errors.append("review_run_missing")
        rank_counts = con.execute(
            """
            SELECT result_id, COUNT(*) AS rows,
                   COUNT(*) - COUNT(DISTINCT source_id) AS duplicate_rows,
                   MIN(model_rank) AS min_rank, MAX(model_rank) AS max_rank
            FROM candidate_model_ranks
            GROUP BY result_id
            ORDER BY result_id
            """
        ).fetchdf()
        model_count = con.execute(
            """
            SELECT COUNT(*) AS rows,
                   COUNT(DISTINCT result_id) AS distinct_result_ids
            FROM candidate_models
            """
        ).fetchone()
        consensus = con.execute(
            """
            SELECT COUNT(*) AS rows,
                   COUNT(*) - COUNT(DISTINCT source_id) AS duplicates,
                   MIN(consensus_rank) AS min_rank,
                   MAX(consensus_rank) AS max_rank,
                   MIN(eligibility_rank) AS min_eligibility_rank,
                   MAX(eligibility_rank) AS max_eligibility_rank,
                   COUNT(*) FILTER (
                       WHERE eligible_model_count < ranked_model_count
                          OR eligible_model_count <= 0
                          OR eligible_support_fraction < 0
                          OR eligible_support_fraction > 1
                   ) AS invalid_eligibility_rows
            FROM candidate_consensus
            """
        ).fetchone()
        eligibility = con.execute(
            """
            SELECT COUNT(*) AS rows,
                   COUNT(*) - COUNT(DISTINCT (source_id, result_id)) AS duplicates,
                   COUNT(*) FILTER (
                       WHERE model_status = 'ranked_top_10000'
                         AND NOT eligible
                   ) AS ranked_ineligible,
                   COUNT(*) FILTER (
                       WHERE model_status NOT IN (
                           'ranked_top_10000',
                           'eligible_below_top_10000',
                           'ineligible_variant'
                       )
                   ) AS invalid_status,
                   COUNT(*) FILTER (
                       WHERE eligible != (
                           locus_keep AND photometry_keep AND astrometry_keep
                       )
                   ) AS inconsistent_component_masks,
                   COUNT(*) FILTER (
                       WHERE eligible
                         AND eligibility_reason != 'passed_all_variant_filters'
                   ) AS invalid_pass_reason,
                   COUNT(*) FILTER (
                       WHERE NOT eligible
                         AND eligibility_reason NOT LIKE 'failed_%'
                   ) AS invalid_failure_reason
            FROM candidate_model_eligibility
            """
        ).fetchone()
        review_counts = con.execute(
            """
            SELECT COUNT(*) AS rows,
                   COUNT(*) - COUNT(DISTINCT source_id) AS duplicates,
                   COUNT(*) FILTER (WHERE source_hash_v1 IS NULL) AS lost_sources
            FROM candidate_review
            """
        ).fetchone()
        rec_count = con.execute(
            "SELECT COUNT(*) FROM candidate_recommendations"
        ).fetchone()[0]
    if run is not None:
        expected_ids = [
            str(item["result_id"]) for item in config["review"]["models"]
        ]
        if json.loads(run[0]) != expected_ids:
            errors.append("selected_result_ids_mismatch")
        if int(model_count[0]) != len(expected_ids):
            errors.append("candidate_model_row_count_mismatch")
        if int(model_count[1]) != len(expected_ids):
            errors.append("candidate_model_result_ids_not_unique")
        per_model_limit = int(run[1])
        if set(rank_counts["result_id"].astype(str)) != set(expected_ids):
            errors.append("ranked_model_set_mismatch")
        for row in rank_counts.to_dict("records"):
            if int(row["duplicate_rows"]):
                errors.append(f"{row['result_id']}: duplicate_source_ids")
            if int(row["min_rank"]) != 1:
                errors.append(f"{row['result_id']}: model_rank_does_not_start_at_1")
            if int(row["max_rank"]) != int(row["rows"]):
                errors.append(f"{row['result_id']}: non_contiguous_model_rank")
            if int(row["rows"]) > per_model_limit:
                errors.append(f"{row['result_id']}: rank_limit_exceeded")
        if int(review_counts[0]) != min(int(run[2]), int(consensus[0])):
            errors.append("review_row_count_mismatch")
        if int(rec_count) > int(run[3]):
            errors.append("deep_review_limit_exceeded")
    if int(consensus[1]):
        errors.append("duplicate_consensus_sources")
    if int(consensus[0]) and (
        int(consensus[2]) != 1 or int(consensus[3]) != int(consensus[0])
    ):
        errors.append("non_contiguous_consensus_rank")
    if int(consensus[0]) and (
        int(consensus[4]) != 1 or int(consensus[5]) != int(consensus[0])
    ):
        errors.append("non_contiguous_eligibility_rank")
    if int(consensus[6]):
        errors.append("invalid_consensus_eligibility")
    if int(eligibility[0]) != int(consensus[0]) * int(model_count[0]):
        errors.append("model_eligibility_row_count_mismatch")
    if any(int(value) for value in eligibility[1:]):
        errors.append("invalid_model_eligibility")
    if int(review_counts[1]):
        errors.append("duplicate_review_sources")
    if int(review_counts[2]):
        errors.append("review_feature_join_loss")
    return {
        "review_run_id": expected_run_id,
        "database_path": str(db_path),
        "model_rows": int(model_count[0]),
        "model_rank_rows": int(rank_counts["rows"].sum()),
        "consensus_rows": int(consensus[0]),
        "model_eligibility_rows": int(eligibility[0]),
        "review_rows": int(review_counts[0]),
        "recommendation_rows": int(rec_count),
        "ok": not errors,
        "errors": errors,
    }


def query_simbad_candidates(
    candidates: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Query exact Gaia identifiers, then bounded positional fallbacks."""
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astroquery.simbad import Simbad

    timeout = int(config.get("timeout_seconds", 180))
    batch_size = int(config.get("batch_size", 100))
    fallback_limit = int(config.get("positional_fallback_limit", 20))
    radius_arcsec = float(config.get("positional_radius_arcsec", 1.0))
    client = Simbad()
    client.TIMEOUT = timeout
    for fields in [("otype", "alltypes", "sp_type"), ("otype", "sp_type")]:
        try:
            client.add_votable_fields(*fields)
            break
        except Exception:
            client.reset_votable_fields()

    exact_frames: list[pd.DataFrame] = []
    names = [
        f"Gaia DR3 {int(value)}"
        for value in candidates["source_id"].astype("int64")
    ]
    for start in range(0, len(names), max(1, batch_size)):
        batch = names[start : start + batch_size]
        table = client.query_objects(
            batch,
            cache=False,
            async_job=True,
        )
        if table is None:
            continue
        exact_frames.append(_normalize_exact_simbad(table.to_pandas()))
    exact = (
        pd.concat(exact_frames, ignore_index=True)
        if exact_frames
        else pd.DataFrame()
    )
    matches = _complete_simbad_rows(
        candidates[["source_id", "ra", "dec"]],
        exact,
    )
    unmatched = matches.loc[~matches["match_found"], "source_id"].head(
        fallback_limit
    )
    fallback_rows: list[dict[str, Any]] = []
    indexed = candidates.set_index("source_id")
    for source_id in unmatched.astype("int64"):
        source = indexed.loc[int(source_id)]
        coord = SkyCoord(
            ra=float(source["ra"]) * u.deg,
            dec=float(source["dec"]) * u.deg,
            frame="icrs",
        )
        table = client.query_region(
            coord,
            radius=radius_arcsec * u.arcsec,
            cache=False,
        )
        if table is None or len(table) == 0:
            continue
        fallback_rows.append(
            _normalize_positional_simbad(
                table.to_pandas(),
                source_id=int(source_id),
                source_ra=float(source["ra"]),
                source_dec=float(source["dec"]),
            )
        )
    if fallback_rows:
        fallback = pd.DataFrame(fallback_rows)
        matches = matches.set_index("source_id")
        for row in fallback.to_dict("records"):
            matches.loc[int(row["source_id"]), list(row.keys())[1:]] = list(
                row.values()
            )[1:]
        matches = matches.reset_index()
    matches["queried_at"] = datetime.now(UTC)
    return _stringify_object_columns(matches)


def _validate_scoring_contract(
    db_path: Path,
    *,
    scoring_run_id: str,
    model_run_id: str,
    result_ids: list[str],
) -> dict[str, Any]:
    with duckdb.connect(str(db_path), read_only=True) as con:
        run = con.execute(
            """
            SELECT model_run_id, pool_build_id, status
            FROM prediction_scoring_runs
            WHERE scoring_run_id=?
            """,
            [scoring_run_id],
        ).fetchone()
        if run is None:
            raise ValueError(f"Unknown scoring_run_id {scoring_run_id!r}.")
        models = con.execute(
            """
            SELECT result_id
            FROM prediction_scoring_models
            WHERE scoring_run_id=?
            ORDER BY result_id
            """,
            [scoring_run_id],
        ).fetchdf()
        tile_status = con.execute(
            """
            SELECT status, COUNT(*) AS rows
            FROM prediction_scoring_tiles
            WHERE scoring_run_id=?
            GROUP BY status
            """,
            [scoring_run_id],
        ).fetchdf()
    if str(run[0]) != model_run_id:
        raise ValueError("Candidate model_run_id does not match scoring.")
    if str(run[2]) != "completed":
        raise ValueError(f"Scoring run is not completed: {run[2]}.")
    found = set(models["result_id"].astype(str))
    if found != set(result_ids):
        raise ValueError(
            "Candidate result_ids do not exactly match the scoring run."
        )
    noncompleted = tile_status.loc[
        ~tile_status["status"].eq("completed"), "rows"
    ].sum()
    if int(noncompleted):
        raise ValueError("Scoring run contains non-completed work units.")
    return {
        "model_run_id": str(run[0]),
        "pool_build_id": str(run[1]),
        "status": str(run[2]),
        "work_units": int(tile_status["rows"].sum()),
    }


def _model_metadata(
    db_path: Path,
    *,
    scoring_run_id: str,
    selected_models: list[dict[str, Any]],
) -> pd.DataFrame:
    with duckdb.connect(str(db_path), read_only=True) as con:
        metadata = con.execute(
            """
            SELECT result_id, dataset_variant, variant_bit
            FROM prediction_scoring_models
            WHERE scoring_run_id=?
            """,
            [scoring_run_id],
        ).fetchdf()
    annotations = pd.DataFrame(selected_models)
    metadata = metadata.merge(
        annotations, on="result_id", how="left", validate="one_to_one"
    )
    if metadata[["model", "role"]].isna().any().any():
        raise ValueError("Every selected model needs model and role annotations.")
    order = {str(item["result_id"]): i for i, item in enumerate(selected_models)}
    metadata["_order"] = metadata["result_id"].map(order)
    return metadata.sort_values("_order").drop(columns="_order")


def _load_candidate_model_metrics(
    model_metadata: pd.DataFrame,
    *,
    training_history_db: Path | None,
    model_run_id: str,
) -> pd.DataFrame:
    """Attach compact training metrics without duplicating experiment history."""
    if training_history_db is None:
        return model_metadata.copy()
    if not training_history_db.exists():
        raise FileNotFoundError(training_history_db)
    result_ids = model_metadata["result_id"].astype(str).tolist()
    placeholders = ", ".join("?" for _ in result_ids)
    with duckdb.connect(str(training_history_db), read_only=True) as con:
        metrics = con.execute(
            f"""
            SELECT result_id, run_id, feature_set,
                   model AS training_model, sampler, n_holdout, wr_holdout,
                   holdout_average_precision,
                   holdout_precision_at_50, holdout_recall_at_50,
                   holdout_precision_at_100, holdout_recall_at_100,
                   holdout_recall_at_fpr_0p001,
                   threshold_calibration_negative_pass_rate,
                   overfit_risk_score, overfit_warning_flag, selection_status
            FROM model_results
            WHERE result_id IN ({placeholders})
            """,
            result_ids,
        ).fetchdf()
    if len(metrics) != len(result_ids):
        found = set(metrics["result_id"].astype(str))
        raise ValueError(
            "Training history is missing candidate models: "
            f"{sorted(set(result_ids) - found)}"
        )
    if set(metrics["run_id"].astype(str)) != {model_run_id}:
        raise ValueError("Candidate model metrics have the wrong training run.")
    out = model_metadata.merge(
        metrics,
        on="result_id",
        how="left",
        validate="one_to_one",
    )
    if not out["model"].eq(out["training_model"]).all():
        raise ValueError("Configured estimator labels disagree with training history.")
    return out.drop(columns="training_model")


def _load_model_ranks(
    db_path: Path,
    *,
    scoring_run_id: str,
    model_metadata: pd.DataFrame,
    per_model_limit: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    with duckdb.connect(str(db_path), read_only=True) as con:
        for row in model_metadata.to_dict("records"):
            frame = con.execute(
                """
                SELECT source_id, score, predicted, threshold, result_id,
                       dataset_variant
                FROM prediction_pool_scores
                WHERE scoring_run_id=? AND result_id=?
                ORDER BY score DESC, source_id
                LIMIT ?
                """,
                [scoring_run_id, row["result_id"], per_model_limit],
            ).fetchdf()
            frame.insert(0, "model_rank", np.arange(1, len(frame) + 1))
            frame["model"] = row["model"]
            frame["role"] = row["role"]
            frames.append(frame)
    if not frames:
        raise ValueError("No scored candidates were found.")
    return pd.concat(frames, ignore_index=True)


def _load_pool_features(
    pool_db: Path,
    source_ids: Iterable[int],
) -> pd.DataFrame:
    ids = pd.DataFrame(
        {"source_id": pd.Series(list(source_ids), dtype="int64")}
    )
    with duckdb.connect() as con:
        con.execute(
            f"ATTACH '{pool_db.as_posix()}' AS pool (READ_ONLY)"
        )
        con.register("candidate_ids", ids)
        selected = ", ".join(f"source.{column}" for column in POOL_FEATURE_COLUMNS)
        frame = con.execute(
            f"""
            SELECT {selected}
            FROM pool.prediction_pool_sources source
            INNER JOIN candidate_ids ids USING (source_id)
            ORDER BY source_id
            """
        ).fetchdf()
    if len(frame) != len(ids) or frame["source_id"].nunique() != len(ids):
        raise ValueError("Prediction-pool feature join did not preserve grain.")
    return frame


def _load_pool_eligibility(
    pool_db: Path,
    source_ids: Iterable[int],
) -> pd.DataFrame:
    ids = pd.DataFrame(
        {"source_id": pd.Series(list(source_ids), dtype="int64")}
    )
    with duckdb.connect() as con:
        con.execute(f"ATTACH '{pool_db.as_posix()}' AS pool (READ_ONLY)")
        con.register("candidate_ids", ids)
        frame = con.execute(
            """
            SELECT source.source_id,
                   source.exact_locus_variant_mask,
                   source.photometry_variant_mask,
                   source.astrometry_variant_mask,
                   source.compatible_variant_mask
            FROM pool.prediction_pool_sources source
            INNER JOIN candidate_ids ids USING (source_id)
            ORDER BY source.source_id
            """
        ).fetchdf()
    if len(frame) != len(ids) or frame["source_id"].nunique() != len(ids):
        raise ValueError("Prediction-pool eligibility join did not preserve grain.")
    return frame


def _eligibility_failure_reason(row: Mapping[str, Any]) -> str:
    failed = [
        label
        for column, label in [
            ("locus_keep", "locus"),
            ("photometry_keep", "photometry_quality"),
            ("astrometry_keep", "astrometry"),
        ]
        if not bool(row[column])
    ]
    return "failed_" + "+".join(failed)


def _normalize_exact_simbad(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    user = _first_column(frame, ["user_specified_id"])
    source_id = user.astype("string").str.extract(r"Gaia\s+DR3\s+(\d+)")[0]
    main_id = _first_column(frame, ["main_id"])
    main_id_text = main_id.astype("string").str.strip()
    match_found = main_id_text.notna() & main_id_text.ne("")
    out = pd.DataFrame(
        {
            "source_id": pd.to_numeric(source_id, errors="coerce"),
            "match_found": match_found,
            "match_method": np.where(
                match_found, "exact_gaia_dr3_id", "none"
            ),
            "match_count": match_found.astype("int64"),
            "separation_arcsec": np.where(match_found, 0.0, np.nan),
            "simbad_main_id": main_id.where(match_found),
            "simbad_main_type": _first_column(frame, ["otype"]),
            "simbad_other_types": _first_column(
                frame, ["alltypes", "alltypes.otypes", "otypes"]
            ),
            "simbad_sp_type": _first_column(frame, ["sp_type"]),
        }
    )
    out = out[out["source_id"].notna()].copy()
    out["source_id"] = out["source_id"].astype("int64")
    return _stringify_object_columns(out)


def _normalize_positional_simbad(
    frame: pd.DataFrame,
    *,
    source_id: int,
    source_ra: float,
    source_dec: float,
) -> dict[str, Any]:
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    count = len(frame)
    ra = pd.to_numeric(_first_column(frame, ["ra"]), errors="coerce")
    dec = pd.to_numeric(_first_column(frame, ["dec"]), errors="coerce")
    source_coord = SkyCoord(source_ra * u.deg, source_dec * u.deg)
    valid = ra.notna() & dec.notna()
    if valid.any():
        matches = SkyCoord(ra[valid].to_numpy() * u.deg, dec[valid].to_numpy() * u.deg)
        separations = source_coord.separation(matches).arcsec
        chosen_index = ra[valid].index[int(np.argmin(separations))]
        separation = float(np.min(separations))
    else:
        chosen_index = frame.index[0]
        separation = float("nan")
    chosen = frame.loc[chosen_index]
    method = "position_unique" if count == 1 else "position_ambiguous"
    return {
        "source_id": int(source_id),
        "match_found": True,
        "match_method": method,
        "match_count": int(count),
        "separation_arcsec": separation,
        "simbad_main_id": _value_from_row(chosen, ["main_id"]),
        "simbad_main_type": _value_from_row(chosen, ["otype"]),
        "simbad_other_types": _value_from_row(
            chosen, ["alltypes", "alltypes.otypes", "otypes"]
        ),
        "simbad_sp_type": _value_from_row(chosen, ["sp_type"]),
        "queried_at": datetime.now(UTC),
    }


def _complete_simbad_rows(
    candidates: pd.DataFrame,
    matches: pd.DataFrame,
) -> pd.DataFrame:
    base = candidates[["source_id"]].drop_duplicates().copy()
    if matches.empty:
        return _empty_simbad_matches(candidates)
    matches = matches.sort_values(
        ["source_id", "match_found"], ascending=[True, False]
    ).drop_duplicates("source_id", keep="first")
    out = base.merge(matches, on="source_id", how="left")
    out["match_found"] = out["match_found"].fillna(False).astype(bool)
    out["match_method"] = out["match_method"].fillna("none")
    out["match_count"] = (
        pd.to_numeric(out["match_count"], errors="coerce")
        .fillna(0)
        .astype("int64")
    )
    for column in [
        "simbad_main_id",
        "simbad_main_type",
        "simbad_other_types",
        "simbad_sp_type",
    ]:
        if column not in out:
            out[column] = pd.NA
    if "separation_arcsec" not in out:
        out["separation_arcsec"] = np.nan
    if "queried_at" not in out:
        out["queried_at"] = datetime.now(UTC)
    return _stringify_object_columns(out)


def _empty_simbad_matches(candidates: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_id": candidates["source_id"].astype("int64"),
            "match_found": False,
            "match_method": "none",
            "match_count": 0,
            "separation_arcsec": np.nan,
            "simbad_main_id": pd.Series(pd.NA, index=candidates.index, dtype="string"),
            "simbad_main_type": pd.Series(pd.NA, index=candidates.index, dtype="string"),
            "simbad_other_types": pd.Series(pd.NA, index=candidates.index, dtype="string"),
            "simbad_sp_type": pd.Series(pd.NA, index=candidates.index, dtype="string"),
            "queried_at": datetime.now(UTC),
        }
    ).reset_index(drop=True)


def _write_review_database(
    db_path: Path,
    tables: Mapping[str, pd.DataFrame],
    *,
    history_dir: Path,
) -> Path:
    """Publish the current review while retaining every database revision by hash."""
    temporary = db_path.with_suffix(db_path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    with duckdb.connect(str(temporary)) as con:
        for name, frame in tables.items():
            con.register("incoming", frame)
            con.execute(f"CREATE TABLE {name} AS SELECT * FROM incoming")
            con.unregister("incoming")
    history_dir.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        _archive_review_database(db_path, history_dir)
    snapshot = _archive_review_database(temporary, history_dir)
    temporary.replace(db_path)
    return snapshot


def _archive_review_database(source: Path, history_dir: Path) -> Path:
    digest = file_sha256(source)
    snapshot = history_dir / f"{digest}.duckdb"
    if snapshot.exists():
        if file_sha256(snapshot) != digest:
            raise ValueError(f"Candidate-review archive is corrupt: {snapshot}")
        return snapshot
    temporary = history_dir / f".{digest}.tmp.duckdb"
    shutil.copy2(source, temporary)
    if file_sha256(temporary) != digest:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"Candidate-review archive copy failed: {source}")
    temporary.replace(snapshot)
    return snapshot


def _write_exports(
    output_dir: Path,
    *,
    model_ranks: pd.DataFrame,
    review_frame: pd.DataFrame,
    recommendations: pd.DataFrame,
    simbad_matches: pd.DataFrame,
    headline_limit: int,
    deep_review_limit: int,
) -> dict[str, str]:
    exports = {
        "top_candidates_by_model": output_dir / "top_candidates_by_model.csv",
        "top_500_candidates": output_dir / "top_500_candidates.csv",
        "simbad_crossmatch": output_dir / "simbad_crossmatch.csv",
        "top_20_review": output_dir / "top_20_review.csv",
        "top_5_candidates": output_dir / "top_5_candidates.csv",
    }
    frames = {
        "top_candidates_by_model": model_ranks,
        "top_500_candidates": review_frame,
        "simbad_crossmatch": simbad_matches,
        "top_20_review": recommendations.head(deep_review_limit),
        "top_5_candidates": recommendations.head(headline_limit),
    }
    for key, path in exports.items():
        temporary = path.with_suffix(path.suffix + ".tmp")
        frames[key].to_csv(temporary, index=False)
        temporary.replace(path)
    return {key: str(path) for key, path in exports.items()}


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def _first_column(
    frame: pd.DataFrame,
    candidates: Iterable[str],
) -> pd.Series:
    lowered = {str(column).lower(): column for column in frame.columns}
    for candidate in candidates:
        column = lowered.get(candidate.lower())
        if column is not None:
            return frame[column]
    return pd.Series(pd.NA, index=frame.index)


def _value_from_row(row: pd.Series, candidates: Iterable[str]) -> Any:
    lowered = {str(column).lower(): column for column in row.index}
    for candidate in candidates:
        column = lowered.get(candidate.lower())
        if column is not None:
            return row[column]
    return pd.NA


def _clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "<na>"} else text


def _optional_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def _optional_int(value: Any) -> int | None:
    parsed = _optional_float(value)
    return int(parsed) if parsed is not None else None


def _stringify_object_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in out.columns:
        if out[column].dtype == object:
            out[column] = out[column].map(_stringify_cell).astype("string")
    return out


def _stringify_cell(value: Any) -> Any:
    if value is None or value is pd.NA:
        return pd.NA
    try:
        if pd.isna(value):
            return pd.NA
    except (TypeError, ValueError):
        pass
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return "|".join(str(item) for item in value)
    return str(value)


def canonical_sha256(value: Any) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
