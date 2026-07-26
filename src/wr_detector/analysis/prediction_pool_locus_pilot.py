"""Reproducible pilot comparing aggregate and exact prediction-pool loci.

The pilot deliberately writes to dedicated audit directories. It never mutates
the canonical prediction-pool DuckDB or its completed parquet tiles.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import qmc
from astropy import units as u
from astropy.coordinates import SkyCoord

from wr_detector.config import load_yaml, resolve_path
from wr_detector.pipelines.prediction_pool import (
    BASE_COLUMNS,
    LOCAL_COLOR_EXPRESSIONS,
    SkyTile,
    build_prediction_pool_adql,
    derive_color_envelope,
    download_prediction_tile,
    load_prediction_pool_config,
)
from wr_detector.pipelines.exact_variant_union import (
    ExactLocus,
    LocusPlane,
    VariantMaskSchema,
    add_local_colors,
    add_compatible_variant_mask,
    build_variant_mask_schema,
    evaluate_locus,
    load_exact_loci_from_exports,
    variant_astrometry_mask,
    variant_photometry_mask,
)


def run_prediction_pool_locus_pilot(
    config_path: str | Path = "configs/prediction_pool_locus_pilot.yaml",
    *,
    download: bool = True,
    verbose: bool = False,
) -> dict[str, object]:
    """Download bounded acquisitions, compare policies, and save audit outputs."""
    pilot = load_yaml(config_path)
    run_id = str(pilot["run_id"])
    pool_config_path = pilot["prediction_pool_config"]
    models_config_path = pilot["models_config"]
    pool_config = load_prediction_pool_config(pool_config_path)
    models_config = load_yaml(models_config_path)
    envelope = derive_color_envelope(pool_config)
    exact_loci = load_exact_loci(
        pool_config,
        variants=list(pilot["exact_locus_variants"]),
    )
    mask_schema = build_variant_mask_schema(list(exact_loci))
    extra_gaia_columns = {
        str(alias): str(column)
        for alias, column in pilot.get("extra_gaia_columns", {}).items()
    }

    raw_dir = _output_dir(pilot, "raw_dir_template", run_id)
    data_dir = _output_dir(pilot, "data_dir_template", run_id)
    report_dir = _output_dir(pilot, "report_dir_template", run_id)
    for path in [raw_dir, data_dir, report_dir]:
        path.mkdir(parents=True, exist_ok=True)

    manifest_path = report_dir / "manifest.json"
    previous_manifest: dict[str, Any] = {}
    if manifest_path.exists():
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    previous_regions = {
        str(region["name"]): region
        for region in previous_manifest.get("regions", [])
        if isinstance(region, dict) and region.get("name")
    }
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "created_at": previous_manifest.get("created_at", datetime.now(UTC).isoformat()),
        "analysis_updated_at": datetime.now(UTC).isoformat(),
        "status": "running",
        "pilot_config_path": str(Path(config_path)),
        "pilot_config_sha256": _file_sha256(resolve_path(config_path)),
        "prediction_pool_config_path": str(Path(pool_config_path)),
        "prediction_pool_config_sha256": _file_sha256(resolve_path(pool_config_path)),
        "models_config_path": str(Path(models_config_path)),
        "models_config_sha256": _file_sha256(resolve_path(models_config_path)),
        "acquisition_envelope": envelope,
        "acquisition_envelope_sha256": _json_sha256(envelope),
        "bitmask_schema": mask_schema.as_manifest(),
        "exact_loci": [
            {
                "variant": locus.variant,
                "bit": mask_schema.bit_for(locus.variant),
                "locus_run_id": locus.locus_run_id,
                "source_path": locus.source_path,
                "source_sha256": locus.source_sha256,
                "required_colors": list(locus.required_colors),
                "photometric_condition": _variant_photometry_description(locus.variant),
                "astrometric_condition": _variant_astrometry_description(locus.variant),
                "aggregate_min_outlier_planes": locus.aggregate_min_outlier_planes,
                "planes": [asdict(plane) for plane in locus.planes],
            }
            for locus in exact_loci.values()
        ],
        "regions": [],
    }
    _write_json(manifest_path, manifest)

    frames: list[pd.DataFrame] = []
    try:
        for region in pilot["regions"]:
            tile = SkyTile(
                tile_id=str(region["name"]),
                ra_min=float(region["ra_min"]),
                ra_max=float(region["ra_max"]),
                dec_min=float(region["dec_min"]),
                dec_max=float(region["dec_max"]),
            )
            query = build_prediction_pool_adql(
                tile,
                envelope,
                pool_config,
                extra_gaia_columns=extra_gaia_columns,
            )
            query_path = raw_dir / f"{tile.tile_id}.adql"
            query_path.write_text(query.strip() + "\n", encoding="utf-8")
            csv_path = raw_dir / f"{tile.tile_id}.csv"
            job_id: str | None = None
            reuse_allowed = bool(pilot.get("reuse_existing_acquisitions", True))
            reused_acquisition = bool(csv_path.exists() and (not download or reuse_allowed))
            if download and not reused_acquisition:
                downloaded_path, job_id = download_prediction_tile(
                    tile,
                    envelope,
                    pool_config,
                    raw_dir,
                    None,
                    query_override=query,
                )
                csv_path = downloaded_path
            if not csv_path.exists():
                raise FileNotFoundError(f"Pilot acquisition is missing: {csv_path}")
            acquired = pd.read_csv(csv_path)
            acquired = _add_region_annotations(acquired, tile=tile, region=region)
            compared = compare_locus_policies(
                acquired,
                envelope=envelope,
                exact_loci=exact_loci,
                feature_sets=models_config["feature_sets"],
                mask_schema=mask_schema,
            )
            frames.append(compared)
            region_record = {
                **region,
                "query_path": str(query_path),
                "query_sha256": _file_sha256(query_path),
                "csv_path": str(csv_path),
                "csv_sha256": _file_sha256(csv_path),
                "csv_last_modified_utc": datetime.fromtimestamp(csv_path.stat().st_mtime, UTC).isoformat(),
                "gaia_job_id": job_id or previous_regions.get(tile.tile_id, {}).get("gaia_job_id"),
                "reused_existing_acquisition": reused_acquisition,
                "acquired_rows": int(len(acquired)),
                "ag_gspphot_available_rows": int(acquired["ag_gspphot"].notna().sum()),
                "ag_gspphot_median": _finite_or_none(acquired["ag_gspphot"].median()),
            }
            if region.get("legacy_csv"):
                region_record["legacy_comparison"] = compare_legacy_acquisition(
                    region,
                    acquired,
                )
            manifest["regions"].append(region_record)
            _write_json(manifest_path, manifest)
            if verbose:
                missed = int((compared["passes_any_exact_locus"] & ~compared["aggregated_locus_keep"]).sum())
                print(f"{tile.tile_id}: acquired={len(compared):,}, exact_not_aggregate={missed:,}", flush=True)

        sources = pd.concat(frames, ignore_index=True, sort=False)
        if sources.duplicated(["region", "source_id"]).any():
            raise ValueError("The pilot contains duplicate source_id rows within a region.")
        sources.to_parquet(data_dir / "source_comparison.parquet", index=False)
        controls = build_known_wr_controls(
            pool_config,
            envelope=envelope,
            exact_loci=exact_loci,
            feature_sets=models_config["feature_sets"],
            mask_schema=mask_schema,
        )
        controls.to_parquet(data_dir / "known_wr_controls.parquet", index=False)
        tables = build_summary_tables(
            sources,
            controls,
            pilot,
            exact_loci=exact_loci,
            mask_schema=mask_schema,
        )
        containment, synthetic_counterexamples = analyze_geometric_containment(
            envelope=envelope,
            exact_loci=exact_loci,
        )
        tables["geometric_containment"] = containment
        tables["synthetic_counterexamples"] = synthetic_counterexamples
        tables["current_pool_reconciliation"] = reconcile_current_pool(
            sources,
            regions=list(pilot["regions"]),
            current_pool_db=resolve_path(pool_config["output_db"]),
        )
        tables["legacy_lineage_summary"] = audit_legacy_csv_lineage(pilot["legacy_lineage"])
        storage_outputs, storage_table = build_storage_sample_and_estimate(
            sources,
            data_dir=data_dir,
            pool_config=pool_config,
        )
        tables["storage_estimate"] = storage_table
        for name, table in tables.items():
            table.to_csv(report_dir / f"{name}.csv", index=False)
        figure_paths = build_pilot_figures(
            sources,
            exact_loci=exact_loci,
            envelope=envelope,
            report_dir=report_dir,
            magnitude_labels=list(pilot["magnitude_labels"]),
        )
        artifact_path = build_report_artifact(tables, report_dir=report_dir, run_id=run_id)
        manifest["status"] = "completed"
        manifest["completed_at"] = datetime.now(UTC).isoformat()
        manifest["outputs"] = {
            "source_comparison": str(data_dir / "source_comparison.parquet"),
            "known_wr_controls": str(data_dir / "known_wr_controls.parquet"),
            "tables": {name: str(report_dir / f"{name}.csv") for name in tables},
            "figures": [str(path) for path in figure_paths],
            "report_artifact": str(artifact_path),
            "storage_samples": storage_outputs,
        }
        output_artifact_paths = [
            data_dir / "source_comparison.parquet",
            data_dir / "known_wr_controls.parquet",
            *[report_dir / f"{name}.csv" for name in tables],
            *figure_paths,
            artifact_path,
        ]
        for directory_key in ["acquisition_dir", "eligible_dir", "acquisition_with_mask_dir"]:
            output_artifact_paths.extend(Path(storage_outputs[directory_key]).glob("*.parquet"))
        manifest["output_artifacts"] = [
            _artifact_record(path)
            for path in sorted(set(output_artifact_paths), key=lambda item: str(item))
        ]
        _write_json(manifest_path, manifest)
        return {
            "run_id": run_id,
            "manifest_path": str(manifest_path),
            "source_rows": int(len(sources)),
            "known_wr_controls": int(len(controls)),
            "report_dir": str(report_dir),
            "data_dir": str(data_dir),
        }
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failed_at"] = datetime.now(UTC).isoformat()
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        _write_json(manifest_path, manifest)
        raise


def load_exact_loci(pool_config: dict[str, Any], *, variants: list[str]) -> dict[str, ExactLocus]:
    filters = pool_config["filters"]
    reference_dir = resolve_path(pool_config["paths"]["processed_reference_dir"])
    minimum_outliers = int(filters["color_locus"].get("aggregate_min_outlier_planes", 1))
    return load_exact_loci_from_exports(
        reference_dir=reference_dir,
        reference_output_template=filters["color_locus"][
            "reference_output_template"
        ],
        variants=variants,
        aggregate_min_outlier_planes=minimum_outliers,
    )


def _add_region_annotations(
    frame: pd.DataFrame,
    *,
    tile: SkyTile,
    region: dict[str, Any],
) -> pd.DataFrame:
    out = frame.copy()
    if "galactic_l" not in out or "galactic_b" not in out:
        coordinates = SkyCoord(
            ra=pd.to_numeric(out["ra"], errors="coerce").to_numpy() * u.deg,
            dec=pd.to_numeric(out["dec"], errors="coerce").to_numpy() * u.deg,
            frame="icrs",
        ).galactic
        out["galactic_l"] = coordinates.l.deg
        out["galactic_b"] = coordinates.b.deg
    if "ag_gspphot" not in out:
        out["ag_gspphot"] = np.nan
    if "ebpminrp_gspphot" not in out:
        out["ebpminrp_gspphot"] = np.nan
    out["region"] = tile.tile_id
    out["region_description"] = str(region.get("description", ""))
    out["region_area_deg2"] = _sky_box_area_deg2(tile)
    out["region_galactic_l_center"] = float(region.get("galactic_l_center", np.nan))
    out["region_galactic_b_center"] = float(region.get("galactic_b_center", np.nan))
    out["disk_zone"] = str(region.get("disk_zone", "unclassified"))
    out["latitude_stratum"] = str(region.get("latitude_stratum", "unclassified"))
    out["extinction_expectation"] = str(region.get("extinction_expectation", "unclassified"))
    out["control_type"] = "gaia_acquisition"
    return out


def _variant_photometry_description(variant: str) -> str:
    return "2MASS J/H/Ks and WISE W1/W2 quality A" if variant.startswith("strict_") else (
        "2MASS J/H/Ks and WISE W1/W2 quality A or B"
    )


def _variant_astrometry_description(variant: str) -> str:
    if variant.endswith("_photometry"):
        return "no parallax condition"
    if variant.endswith("_parallax_soft"):
        return "parallax > 0"
    threshold = variant.rsplit("_", 1)[-1]
    return f"parallax > 0 and parallax_over_error >= {threshold}"


def _finite_or_none(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def compare_locus_policies(
    frame: pd.DataFrame,
    *,
    envelope: dict[str, Any],
    exact_loci: dict[str, ExactLocus],
    feature_sets: dict[str, list[str]],
    mask_schema: VariantMaskSchema | None = None,
) -> pd.DataFrame:
    out = _add_local_colors(frame)
    aggregate_planes = tuple(
        LocusPlane(
            x=str(fit["x"]),
            y=str(fit["y"]),
            slope=(float(fit["slope_min"]) + float(fit["slope_max"])) / 2.0,
            intercept=(float(fit["intercept_min"]) + float(fit["intercept_max"])) / 2.0,
            threshold=float(fit["threshold_max"]),
            transform=str(fit["transform"]),
        )
        for fit in envelope["fits"]
    )
    min_outlier_planes = next(iter(exact_loci.values())).aggregate_min_outlier_planes
    aggregate = _evaluate_locus(out, aggregate_planes, min_outlier_planes=min_outlier_planes)
    out["aggregated_locus_valid"] = aggregate["valid"]
    out["aggregated_outlier_plane_count"] = aggregate["outlier_count"]
    out["aggregated_locus_keep"] = aggregate["keep"]
    for name, values in aggregate["margins"].items():
        out[f"aggregated_margin__{name}"] = values

    exact_keep_columns = []
    for variant, locus in exact_loci.items():
        evaluated = _evaluate_locus(
            out,
            locus.planes,
            min_outlier_planes=locus.aggregate_min_outlier_planes,
        )
        keep_column = f"exact_locus_keep__{variant}"
        astrometry_column = f"astrometry_keep__{variant}"
        quality_column = f"quality_keep__{variant}"
        eligible_column = f"exact_variant_keep__{variant}"
        out[keep_column] = evaluated["keep"]
        out[f"exact_outlier_plane_count__{variant}"] = evaluated["outlier_count"]
        out[quality_column] = variant_photometry_mask(out, variant)
        out[astrometry_column] = variant_astrometry_mask(out, variant)
        out[eligible_column] = out[keep_column] & out[quality_column] & out[astrometry_column]
        exact_keep_columns.append(keep_column)
        for name, values in evaluated["margins"].items():
            margin_column = f"best_exact_margin__{name}"
            if margin_column not in out:
                out[margin_column] = values
            else:
                out[margin_column] = np.maximum(out[margin_column], values)

    out["exact_loci_accept_count"] = out[exact_keep_columns].sum(axis=1).astype("int16")
    out["passes_any_exact_locus"] = out["exact_loci_accept_count"].gt(0)
    schema = mask_schema or build_variant_mask_schema(list(exact_loci))
    out = add_compatible_variant_mask(out, schema)
    out["exact_variants_accept_count"] = out["compatible_variant_count"]
    out["exact_not_aggregate"] = out["passes_any_exact_locus"] & ~out["aggregated_locus_keep"]
    out["exact_variant_not_aggregate"] = out["passes_any_exact_variant"] & ~out["aggregated_locus_keep"]
    out["aggregate_not_exact"] = out["aggregated_locus_keep"] & ~out["passes_any_exact_locus"]
    out["aggregate_not_exact_variant"] = out["aggregated_locus_keep"] & ~out["passes_any_exact_variant"]
    aggregate_margin_columns = [column for column in out if column.startswith("aggregated_margin__")]
    out["aggregated_boundary_distance"] = out[aggregate_margin_columns].abs().min(axis=1)
    for feature_set, columns in feature_sets.items():
        available = pd.Series(True, index=out.index)
        for column in columns:
            available &= out[column].notna() if column in out else False
        out[f"features_available__{feature_set}"] = available
    return out


def _evaluate_locus(
    frame: pd.DataFrame,
    planes: tuple[LocusPlane, ...],
    *,
    min_outlier_planes: int,
) -> dict[str, Any]:
    return evaluate_locus(
        frame,
        planes,
        min_outlier_planes=min_outlier_planes,
    )


def _add_local_colors(frame: pd.DataFrame) -> pd.DataFrame:
    return add_local_colors(frame)


def analyze_geometric_containment(
    *,
    envelope: dict[str, Any],
    exact_loci: dict[str, ExactLocus],
    sobol_power: int = 18,
    seed: int = 20260722,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Check analytic plane containment and find deterministic color counterexamples.

    The synthetic points obey the Gaia and 2MASS color identities and every
    acquisition-envelope color bound. They are geometric witnesses, not a
    population estimate and not labelled astrophysical sources.
    """
    aggregate_planes = tuple(
        LocusPlane(
            x=str(fit["x"]),
            y=str(fit["y"]),
            slope=(float(fit["slope_min"]) + float(fit["slope_max"])) / 2.0,
            intercept=(float(fit["intercept_min"]) + float(fit["intercept_max"])) / 2.0,
            threshold=float(fit["threshold_max"]),
            transform=str(fit["transform"]),
        )
        for fit in envelope["fits"]
    )
    aggregate_by_name = {plane.name: plane for plane in aggregate_planes}
    independent_colors = ["G_BP", "G_RP", "J_H", "H_K"]
    samples = qmc.Sobol(
        d=len(independent_colors),
        scramble=True,
        seed=seed,
    ).random_base2(int(sobol_power))
    for index, color in enumerate(independent_colors):
        bounds = envelope["color_bounds"][color]
        samples[:, index] = float(bounds["min"]) + samples[:, index] * (
            float(bounds["max"]) - float(bounds["min"])
        )
    colors = pd.DataFrame(samples, columns=independent_colors)
    colors["BP_RP"] = colors["G_RP"] - colors["G_BP"]
    colors["J_K"] = colors["J_H"] + colors["H_K"]
    within_derived_bounds = pd.Series(True, index=colors.index)
    for color in ["BP_RP", "J_K"]:
        bounds = envelope["color_bounds"][color]
        within_derived_bounds &= colors[color].between(
            float(bounds["min"]),
            float(bounds["max"]),
            inclusive="both",
        )
    colors = colors[within_derived_bounds].reset_index(drop=True)
    minimum_outliers = next(iter(exact_loci.values())).aggregate_min_outlier_planes
    aggregate = _evaluate_locus(
        colors,
        aggregate_planes,
        min_outlier_planes=minimum_outliers,
    )
    aggregate_margin = pd.concat(aggregate["margins"], axis=1).abs().min(axis=1)

    summary_rows = []
    counterexample_frames = []
    for variant, locus in exact_loci.items():
        plane_slacks = []
        for exact_plane in locus.planes:
            aggregate_plane = aggregate_by_name[exact_plane.name]
            bounds = envelope["color_bounds"][exact_plane.x]
            transformed_bounds = [
                float(_signed_log1p(pd.Series([bounds[side]])).iloc[0])
                for side in ["min", "max"]
            ]
            required_half_width = max(
                abs(
                    (exact_plane.slope - aggregate_plane.slope) * x
                    + exact_plane.intercept
                    - aggregate_plane.intercept
                )
                + exact_plane.threshold
                for x in transformed_bounds
            )
            plane_slacks.append(aggregate_plane.threshold - required_half_width)

        exact = _evaluate_locus(
            colors,
            locus.planes,
            min_outlier_planes=locus.aggregate_min_outlier_planes,
        )
        discrepancy = exact["keep"] & ~aggregate["keep"]
        summary_rows.append(
            {
                "variant": variant,
                "analytic_contained_planes": int(sum(slack >= -1e-12 for slack in plane_slacks)),
                "analytic_total_planes": len(plane_slacks),
                "analytic_all_planes_contained": bool(all(slack >= -1e-12 for slack in plane_slacks)),
                "minimum_plane_containment_slack": float(min(plane_slacks)),
                "sobol_seed": seed,
                "sobol_power": sobol_power,
                "synthetic_points_inside_color_envelope": len(colors),
                "synthetic_exact_keep": int(exact["keep"].sum()),
                "synthetic_exact_not_aggregate": int(discrepancy.sum()),
                "constructive_counterexample_found": bool(discrepancy.any()),
                "interpretation": (
                    "geometric superset disproved by an envelope-valid color witness"
                    if discrepancy.any()
                    else "no witness found; this does not prove containment"
                ),
            }
        )
        if discrepancy.any():
            examples = colors.loc[discrepancy].head(20).copy()
            examples.insert(0, "variant", variant)
            examples["exact_outlier_plane_count"] = exact["outlier_count"].loc[examples.index].to_numpy()
            examples["aggregated_outlier_plane_count"] = aggregate["outlier_count"].loc[examples.index].to_numpy()
            examples["aggregated_boundary_distance"] = aggregate_margin.loc[examples.index].to_numpy()
            examples["sample_role"] = "synthetic_geometric_witness_not_gaia_source"
            counterexample_frames.append(examples.reset_index(drop=True))

    counterexample_columns = [
        "variant",
        *independent_colors,
        "BP_RP",
        "J_K",
        "exact_outlier_plane_count",
        "aggregated_outlier_plane_count",
        "aggregated_boundary_distance",
        "sample_role",
    ]
    counterexamples = (
        pd.concat(counterexample_frames, ignore_index=True)[counterexample_columns]
        if counterexample_frames
        else pd.DataFrame(columns=counterexample_columns)
    )
    return pd.DataFrame(summary_rows), counterexamples


def build_known_wr_controls(
    pool_config: dict[str, Any],
    *,
    envelope: dict[str, Any],
    exact_loci: dict[str, ExactLocus],
    feature_sets: dict[str, list[str]],
    mask_schema: VariantMaskSchema | None = None,
) -> pd.DataFrame:
    filters = pool_config["filters"]
    reference_dir = resolve_path(pool_config["paths"]["processed_reference_dir"])
    path = reference_dir / filters["color_locus"]["reference_output_template"].format(variant="relaxed_photometry")
    controls = pd.read_parquet(path)
    controls["region"] = "known_wr_controls"
    controls["region_description"] = "Known relaxed-photometry WR controls"
    controls["region_area_deg2"] = np.nan
    controls["control_type"] = "known_wr_reference"
    reference_db = resolve_path(pool_config["paths"]["wr_reference_db"])
    if reference_db.exists():
        with duckdb.connect(str(reference_db), read_only=True) as con:
            identities = con.execute(
                'SELECT source_id, wr_id, "Spectral Type" FROM wr_reference WHERE source_id IS NOT NULL'
            ).fetchdf()
        identities = identities.drop_duplicates(subset=["source_id", "wr_id"])
        controls = controls.merge(
            identities,
            on=["source_id", "wr_id"],
            how="left",
            validate="many_to_one",
        )
    within_envelope = pd.Series(True, index=controls.index)
    for color, bounds in envelope["color_bounds"].items():
        within_envelope &= controls[color].between(float(bounds["min"]), float(bounds["max"]), inclusive="both")
    controls["acquisition_envelope_keep"] = within_envelope
    compared = compare_locus_policies(
        controls,
        envelope=envelope,
        exact_loci=exact_loci,
        feature_sets=feature_sets,
        mask_schema=mask_schema,
    )
    compared["current_policy_keep"] = compared["acquisition_envelope_keep"] & compared["aggregated_locus_keep"]
    compared["exact_union_policy_keep"] = compared["acquisition_envelope_keep"] & compared["passes_any_exact_variant"]
    return compared


def build_summary_tables(
    sources: pd.DataFrame,
    controls: pd.DataFrame,
    pilot: dict[str, Any],
    *,
    exact_loci: dict[str, ExactLocus],
    mask_schema: VariantMaskSchema,
) -> dict[str, pd.DataFrame]:
    region_rows = []
    for region, group in sources.groupby("region", sort=False):
        acquired = len(group)
        aggregate = int(group["aggregated_locus_keep"].sum())
        exact_locus = int(group["passes_any_exact_locus"].sum())
        exact_variant = int(group["passes_any_exact_variant"].sum())
        geometric_additions = int(group["exact_not_aggregate"].sum())
        compatible_additions = int(group["exact_variant_not_aggregate"].sum())
        region_rows.append(
            {
                "region": region,
                "galactic_l_center": _finite_or_none(group["region_galactic_l_center"].iloc[0]),
                "galactic_b_center": _finite_or_none(group["region_galactic_b_center"].iloc[0]),
                "disk_zone": str(group["disk_zone"].iloc[0]),
                "latitude_stratum": str(group["latitude_stratum"].iloc[0]),
                "extinction_expectation": str(group["extinction_expectation"].iloc[0]),
                "area_deg2": float(group["region_area_deg2"].iloc[0]),
                "acquisition_rows": acquired,
                "acquisition_density_per_deg2": acquired / float(group["region_area_deg2"].iloc[0]),
                "aggregated_keep": aggregate,
                "any_exact_locus_keep": exact_locus,
                "any_exact_variant_keep": exact_variant,
                "exact_locus_not_aggregate": geometric_additions,
                "exact_locus_not_aggregate_pct_acquisition": geometric_additions / acquired if acquired else np.nan,
                "exact_locus_not_aggregate_pct_exact_union": geometric_additions / exact_locus if exact_locus else np.nan,
                "exact_variant_not_aggregate": compatible_additions,
                "exact_variant_not_aggregate_pct_acquisition": compatible_additions / acquired if acquired else np.nan,
                "exact_variant_not_aggregate_pct_exact_union": compatible_additions / exact_variant if exact_variant else np.nan,
                "aggregate_not_exact": int(group["aggregate_not_exact"].sum()),
                "aggregate_not_exact_variant": int(group["aggregate_not_exact_variant"].sum()),
                "exact_variant_union_growth_over_aggregate": (exact_variant - aggregate) / aggregate if aggregate else np.nan,
                "acquisition_growth_over_aggregate": (acquired - aggregate) / aggregate if aggregate else np.nan,
                "ag_gspphot_available_rows": int(group["ag_gspphot"].notna().sum()),
                "ag_gspphot_available_pct": float(group["ag_gspphot"].notna().mean()),
                "ag_gspphot_median": _finite_or_none(group["ag_gspphot"].median()),
            }
        )
    region_summary = pd.DataFrame(region_rows)
    if len(region_summary) >= 3:
        region_summary["density_stratum"] = pd.qcut(
            region_summary["acquisition_density_per_deg2"].rank(method="first"),
            q=3,
            labels=["low", "medium", "high"],
        ).astype("string")
    else:
        region_summary["density_stratum"] = "unclassified"
    region_summary["ag_gspphot_stratum"] = pd.cut(
        pd.to_numeric(region_summary["ag_gspphot_median"], errors="coerce"),
        bins=[-np.inf, 0.5, 1.5, np.inf],
        labels=["low", "medium", "high"],
    ).astype("string")
    region_summary.loc[
        region_summary["ag_gspphot_available_pct"].lt(0.1),
        "ag_gspphot_stratum",
    ] = "insufficient_coverage"

    variant_rows = []
    variants = [entry.variant for entry in mask_schema.entries]
    for variant in variants:
        exact_locus = sources[f"exact_locus_keep__{variant}"].fillna(False).astype(bool)
        compatible = sources[f"exact_variant_keep__{variant}"].fillna(False).astype(bool)
        locus_missed = exact_locus & ~sources["aggregated_locus_keep"]
        compatible_missed = compatible & ~sources["aggregated_locus_keep"]
        aggregate_not_locus = sources["aggregated_locus_keep"] & ~exact_locus
        aggregate_not_compatible = sources["aggregated_locus_keep"] & ~compatible
        variant_rows.append(
            {
                "variant": variant,
                "bit": mask_schema.bit_for(variant),
                "locus_run_id": exact_loci[variant].locus_run_id,
                "locus_sha256": exact_loci[variant].source_sha256,
                "exact_locus_keep": int(exact_locus.sum()),
                "exact_locus_not_aggregate": int(locus_missed.sum()),
                "exact_locus_not_aggregate_pct": (
                    float(locus_missed.sum() / exact_locus.sum()) if exact_locus.sum() else np.nan
                ),
                "exact_variant_keep": int(compatible.sum()),
                "exact_variant_not_aggregate": int(compatible_missed.sum()),
                "exact_variant_not_aggregate_pct": (
                    float(compatible_missed.sum() / compatible.sum()) if compatible.sum() else np.nan
                ),
                "aggregate_not_exact_locus": int(aggregate_not_locus.sum()),
                "aggregate_not_exact_variant": int(aggregate_not_compatible.sum()),
                "aggregate_not_exact_variant_pct": float(aggregate_not_compatible.mean()),
                "aggregate_not_exact_variant_pct_aggregate": (
                    float(aggregate_not_compatible.sum() / sources["aggregated_locus_keep"].sum())
                    if sources["aggregated_locus_keep"].sum()
                    else np.nan
                ),
                "current_pool_sample_status": "affected" if compatible_missed.any() else "no_loss_observed",
            }
        )
    variant_summary = pd.DataFrame(variant_rows)

    variant_region_rows = []
    for region, group in sources.groupby("region", sort=False):
        for variant in variants:
            exact_locus = group[f"exact_locus_keep__{variant}"].fillna(False).astype(bool)
            compatible = group[f"exact_variant_keep__{variant}"].fillna(False).astype(bool)
            aggregate = group["aggregated_locus_keep"].fillna(False).astype(bool)
            variant_region_rows.append(
                {
                    "region": region,
                    "variant": variant,
                    "acquisition_rows": len(group),
                    "exact_locus_keep": int(exact_locus.sum()),
                    "exact_locus_not_aggregate": int((exact_locus & ~aggregate).sum()),
                    "exact_variant_keep": int(compatible.sum()),
                    "exact_variant_not_aggregate": int((compatible & ~aggregate).sum()),
                    "exact_variant_not_aggregate_pct": (
                        float((compatible & ~aggregate).sum() / compatible.sum())
                        if compatible.sum()
                        else np.nan
                    ),
                    "aggregate_not_exact_variant": int((aggregate & ~compatible).sum()),
                }
            )
    variant_by_region = pd.DataFrame(variant_region_rows)

    magnitude = sources.copy()
    magnitude["magnitude_bin"] = pd.cut(
        pd.to_numeric(magnitude["G"], errors="coerce"),
        bins=list(pilot["magnitude_bins"]),
        labels=list(pilot["magnitude_labels"]),
        right=False,
        include_lowest=True,
    )
    variant_magnitude_rows = []
    for (region, magnitude_bin), group in magnitude.groupby(
        ["region", "magnitude_bin"], observed=True, sort=False
    ):
        for variant in variants:
            exact_locus = group[f"exact_locus_keep__{variant}"].fillna(False).astype(bool)
            compatible = group[f"exact_variant_keep__{variant}"].fillna(False).astype(bool)
            aggregate = group["aggregated_locus_keep"].fillna(False).astype(bool)
            variant_magnitude_rows.append(
                {
                    "region": region,
                    "magnitude_bin": str(magnitude_bin),
                    "variant": variant,
                    "acquisition_rows": len(group),
                    "exact_locus_keep": int(exact_locus.sum()),
                    "exact_locus_not_aggregate": int((exact_locus & ~aggregate).sum()),
                    "exact_variant_keep": int(compatible.sum()),
                    "exact_variant_not_aggregate": int((compatible & ~aggregate).sum()),
                    "exact_variant_not_aggregate_pct": (
                        float((compatible & ~aggregate).sum() / compatible.sum())
                        if compatible.sum()
                        else np.nan
                    ),
                    "aggregate_not_exact_variant": int((aggregate & ~compatible).sum()),
                }
            )
    variant_by_magnitude = pd.DataFrame(variant_magnitude_rows)
    magnitude_summary = (
        magnitude.groupby(["region", "magnitude_bin"], observed=True)
        .agg(
            acquisition_rows=("source_id", "size"),
            aggregated_keep=("aggregated_locus_keep", "sum"),
            any_exact_locus_keep=("passes_any_exact_locus", "sum"),
            any_exact_variant_keep=("passes_any_exact_variant", "sum"),
            exact_locus_not_aggregate=("exact_not_aggregate", "sum"),
            exact_variant_not_aggregate=("exact_variant_not_aggregate", "sum"),
        )
        .reset_index()
    )
    magnitude_summary["exact_locus_not_aggregate_pct"] = (
        magnitude_summary["exact_locus_not_aggregate"] / magnitude_summary["acquisition_rows"]
    )
    magnitude_summary["exact_variant_not_aggregate_pct"] = (
        magnitude_summary["exact_variant_not_aggregate"] / magnitude_summary["acquisition_rows"]
    )
    magnitude_summary["exact_locus_not_aggregate_pct_exact_union"] = (
        magnitude_summary["exact_locus_not_aggregate"] / magnitude_summary["any_exact_locus_keep"]
    )
    magnitude_summary["exact_variant_not_aggregate_pct_exact_union"] = (
        magnitude_summary["exact_variant_not_aggregate"] / magnitude_summary["any_exact_variant_keep"]
    )
    magnitude_overall = (
        magnitude_summary.groupby("magnitude_bin", observed=True, sort=False)
        .agg(
            acquisition_rows=("acquisition_rows", "sum"),
            aggregated_keep=("aggregated_keep", "sum"),
            any_exact_variant_keep=("any_exact_variant_keep", "sum"),
            exact_variant_not_aggregate=("exact_variant_not_aggregate", "sum"),
        )
        .reset_index()
    )
    magnitude_overall["exact_variant_not_aggregate_pct"] = (
        magnitude_overall["exact_variant_not_aggregate"] / magnitude_overall["acquisition_rows"]
    )
    magnitude_overall["exact_variant_not_aggregate_pct_exact_union"] = (
        magnitude_overall["exact_variant_not_aggregate"] / magnitude_overall["any_exact_variant_keep"]
    )

    extinction = sources.copy()
    extinction["ag_gspphot_bin"] = pd.cut(
        pd.to_numeric(extinction["ag_gspphot"], errors="coerce"),
        bins=[-np.inf, 0.5, 1.5, 3.0, np.inf],
        labels=["ag<=0.5", "0.5<ag<=1.5", "1.5<ag<=3", "ag>3"],
    ).astype("string")
    extinction["ag_gspphot_bin"] = extinction["ag_gspphot_bin"].fillna("missing")
    extinction_summary = (
        extinction.groupby("ag_gspphot_bin", sort=False, dropna=False)
        .agg(
            acquisition_rows=("source_id", "size"),
            aggregated_keep=("aggregated_locus_keep", "sum"),
            any_exact_variant_keep=("passes_any_exact_variant", "sum"),
            exact_variant_not_aggregate=("exact_variant_not_aggregate", "sum"),
            aggregate_not_exact_variant=("aggregate_not_exact_variant", "sum"),
        )
        .reset_index()
    )
    extinction_summary["exact_variant_not_aggregate_pct_exact_union"] = (
        extinction_summary["exact_variant_not_aggregate"]
        / extinction_summary["any_exact_variant_keep"]
    )
    extinction_summary["coverage_note"] = np.where(
        extinction_summary["ag_gspphot_bin"].eq("missing"),
        "Gaia GSP-Phot proxy unavailable",
        "descriptive G-band extinction proxy; not a label",
    )

    region_strata = region_summary.set_index("region")[["density_stratum"]]
    stratified_sources = sources.join(region_strata, on="region", validate="many_to_one")
    strata_rows = []
    for dimension, column in [
        ("disk_zone", "disk_zone"),
        ("latitude", "latitude_stratum"),
        ("density", "density_stratum"),
        ("expected_extinction", "extinction_expectation"),
    ]:
        for stratum, group in stratified_sources.groupby(column, sort=False, dropna=False):
            exact_union = int(group["passes_any_exact_variant"].sum())
            missed = int(group["exact_variant_not_aggregate"].sum())
            strata_rows.append(
                {
                    "dimension": dimension,
                    "stratum": str(stratum),
                    "region_count": int(group["region"].nunique()),
                    "acquisition_rows": int(len(group)),
                    "aggregated_keep": int(group["aggregated_locus_keep"].sum()),
                    "any_exact_variant_keep": exact_union,
                    "exact_variant_not_aggregate": missed,
                    "exact_variant_not_aggregate_pct_exact_union": (
                        missed / exact_union if exact_union else np.nan
                    ),
                    "aggregate_not_exact_variant": int(group["aggregate_not_exact_variant"].sum()),
                }
            )
    stratified_summary = pd.DataFrame(strata_rows)

    feature_rows = []
    for column in [name for name in sources if name.startswith("features_available__")]:
        feature_rows.append(
            {
                "feature_set": column.split("__", 1)[1],
                "available_rows": int(sources[column].sum()),
                "available_pct": float(sources[column].mean()),
            }
        )

    control_summary = pd.DataFrame(
        [
            {
                "known_wr_controls": len(controls),
                "within_acquisition_envelope": int(controls["acquisition_envelope_keep"].sum()),
                "outside_acquisition_envelope": int((~controls["acquisition_envelope_keep"]).sum()),
                "current_policy_keep": int(controls["current_policy_keep"].sum()),
                "exact_union_policy_keep": int(controls["exact_union_policy_keep"].sum()),
                "exact_union_not_current": int(
                    (controls["exact_union_policy_keep"] & ~controls["current_policy_keep"]).sum()
                ),
            }
        ]
    )

    affected_control_columns = [
        column
        for column in [
            "source_id",
            "wr_id",
            "Spectral Type",
            "G",
            "BP_RP",
            "J_K",
            "W1_W2",
            "aggregated_outlier_plane_count",
            "exact_loci_accept_count",
            "exact_variants_accept_count",
        ]
        if column in controls
    ]
    known_wr_affected = controls[
        controls["exact_union_policy_keep"] & ~controls["current_policy_keep"]
    ][affected_control_columns].copy()

    variant_wr_rows = []
    for variant in variants:
        affected = (
            controls["acquisition_envelope_keep"]
            & controls[f"exact_variant_keep__{variant}"]
            & ~controls["aggregated_locus_keep"]
        )
        for _, row in controls.loc[affected].iterrows():
            variant_wr_rows.append(
                {
                    "variant": variant,
                    "bit": mask_schema.bit_for(variant),
                    "source_id": int(row["source_id"]),
                    "wr_id": row.get("wr_id"),
                    "Spectral Type": row.get("Spectral Type"),
                    "G": row.get("G"),
                    "aggregated_outlier_plane_count": row.get("aggregated_outlier_plane_count"),
                }
            )
    variant_wr_affected = pd.DataFrame(
        variant_wr_rows,
        columns=[
            "variant",
            "bit",
            "source_id",
            "wr_id",
            "Spectral Type",
            "G",
            "aggregated_outlier_plane_count",
        ],
    )

    overlap_rows = []
    for left in variants:
        left_mask = sources[f"exact_variant_keep__{left}"].fillna(False).astype(bool)
        for right in variants:
            right_mask = sources[f"exact_variant_keep__{right}"].fillna(False).astype(bool)
            intersection = int((left_mask & right_mask).sum())
            union = int((left_mask | right_mask).sum())
            overlap_rows.append(
                {
                    "left_variant": left,
                    "right_variant": right,
                    "intersection": intersection,
                    "union": union,
                    "jaccard": intersection / union if union else np.nan,
                    "left_retained_by_right": intersection / int(left_mask.sum()) if left_mask.sum() else np.nan,
                }
            )
    variant_overlap = pd.DataFrame(overlap_rows)

    variant_contract = pd.DataFrame(
        [
            {
                "variant": entry.variant,
                "bit": entry.bit,
                "bit_value": entry.value,
                "bitmask_schema_version": mask_schema.version,
                "bitmask_schema_sha256": mask_schema.sha256,
                "locus_run_id": exact_loci[entry.variant].locus_run_id,
                "locus_sha256": exact_loci[entry.variant].source_sha256,
                "photometric_condition": _variant_photometry_description(entry.variant),
                "astrometric_condition": _variant_astrometry_description(entry.variant),
                "required_colors": ",".join(exact_loci[entry.variant].required_colors),
            }
            for entry in mask_schema.entries
        ]
    )

    plane_rows = []
    discrepancy = sources[sources["exact_not_aggregate"]]
    for aggregate_margin in [column for column in sources if column.startswith("aggregated_margin__")]:
        plane = aggregate_margin.split("__", 1)[1]
        exact_margin = f"best_exact_margin__{plane}"
        aggregate_outlier = discrepancy[aggregate_margin].lt(0)
        exact_plane_rescue = aggregate_outlier & discrepancy[exact_margin].ge(0)
        plane_rows.append(
            {
                "plane": plane,
                "discrepancy_rows": len(discrepancy),
                "aggregate_outlier_rows": int(aggregate_outlier.sum()),
                "aggregate_outlier_pct": float(aggregate_outlier.mean()) if len(discrepancy) else np.nan,
                "inside_at_least_one_exact_band": int(exact_plane_rescue.sum()),
            }
        )

    boundary_columns = [
        column
        for column in [
            "region",
            "source_id",
            "gaia_designation",
            "G",
            "BP_RP",
            "J_K",
            "W1_W2",
            "aggregated_outlier_plane_count",
            "exact_loci_accept_count",
            "exact_variants_accept_count",
            "aggregated_boundary_distance",
        ]
        if column in sources
    ]
    boundary_samples = (
        sources[sources["exact_not_aggregate"]]
        .sort_values("aggregated_boundary_distance")
        .head(100)[boundary_columns]
        .copy()
    )
    policy_by_region = region_summary[
        ["region", "acquisition_rows", "aggregated_keep", "any_exact_variant_keep"]
    ].melt(
        id_vars="region",
        var_name="policy",
        value_name="sources",
    )
    policy_by_region["policy"] = policy_by_region["policy"].map(
        {
            "acquisition_rows": "Acquisition envelope",
            "aggregated_keep": "Current aggregate",
            "any_exact_variant_keep": "Compatible exact union",
        }
    )
    aggregate_total = int(region_summary["aggregated_keep"].sum())
    exact_total = int(region_summary["any_exact_variant_keep"].sum())
    acquired_total = int(region_summary["acquisition_rows"].sum())
    headline_metrics = pd.DataFrame(
        [
            {
                "acquisition_sources": acquired_total,
                "current_aggregate_keep": aggregate_total,
                "compatible_exact_union_keep": exact_total,
                "geometric_exact_not_aggregate": int(region_summary["exact_locus_not_aggregate"].sum()),
                "compatible_exact_not_aggregate": int(region_summary["exact_variant_not_aggregate"].sum()),
                "aggregate_not_compatible_exact": int(region_summary["aggregate_not_exact_variant"].sum()),
                "net_exact_union_growth": (exact_total - aggregate_total) / aggregate_total,
                "envelope_growth": (acquired_total - aggregate_total) / aggregate_total,
            }
        ]
    )
    return {
        "headline_metrics": headline_metrics,
        "policy_by_region": policy_by_region,
        "region_summary": region_summary,
        "variant_summary": variant_summary,
        "variant_by_region": variant_by_region,
        "variant_by_magnitude": variant_by_magnitude,
        "variant_wr_affected": variant_wr_affected,
        "variant_overlap": variant_overlap,
        "variant_contract": variant_contract,
        "magnitude_summary": magnitude_summary,
        "magnitude_overall": magnitude_overall,
        "extinction_summary": extinction_summary,
        "stratified_summary": stratified_summary,
        "feature_availability": pd.DataFrame(feature_rows),
        "known_wr_summary": control_summary,
        "known_wr_affected": known_wr_affected,
        "plane_discrepancy_summary": pd.DataFrame(plane_rows).sort_values(
            ["aggregate_outlier_rows", "inside_at_least_one_exact_band"], ascending=False
        ),
        "boundary_samples": boundary_samples,
    }


def reconcile_current_pool(
    sources: pd.DataFrame,
    *,
    regions: list[dict[str, Any]],
    current_pool_db: Path,
) -> pd.DataFrame:
    """Confirm the pilot reproduces the current aggregate before known-source exclusion."""
    rows = []
    with duckdb.connect(str(current_pool_db), read_only=True) as con:
        for region in regions:
            current = con.execute(
                """
                SELECT source_id
                FROM prediction_pool_sources
                WHERE ra >= ? AND ra < ? AND dec >= ? AND dec < ?
                """,
                [region["ra_min"], region["ra_max"], region["dec_min"], region["dec_max"]],
            ).fetchdf()
            current_ids = set(pd.to_numeric(current["source_id"], errors="coerce").dropna().astype("int64"))
            pilot_region = sources[sources["region"].eq(region["name"])]
            aggregate_ids = set(
                pd.to_numeric(
                    pilot_region.loc[pilot_region["aggregated_locus_keep"], "source_id"],
                    errors="coerce",
                )
                .dropna()
                .astype("int64")
            )
            rows.append(
                {
                    "region": region["name"],
                    "pilot_aggregate_keep": len(aggregate_ids),
                    "current_pool_rows": len(current_ids),
                    "current_pool_not_pilot_aggregate": len(current_ids - aggregate_ids),
                    "pilot_aggregate_not_current_pool": len(aggregate_ids - current_ids),
                }
            )
    return pd.DataFrame(rows)


def audit_legacy_csv_lineage(config: dict[str, Any]) -> pd.DataFrame:
    """Summarize whether old raw acquisitions can support the primary pilot."""
    raw_dir = resolve_path(config["raw_dir"])
    csv_paths = sorted(raw_dir.glob("*.csv"))
    stems = {path.stem for path in csv_paths}

    def registry_matches(path: Path) -> tuple[int, int, set[str]]:
        with duckdb.connect(str(path), read_only=True) as con:
            registry = con.execute("SELECT tile_id, status FROM prediction_pool_tiles").fetchdf()
        registered = set(registry["tile_id"].astype(str))
        matches = stems & registered
        completed = set(registry.loc[registry["status"].eq("completed"), "tile_id"].astype(str))
        return len(matches), len(matches & completed), stems - registered

    partial_matches, partial_completed, partial_orphans = registry_matches(resolve_path(config["partial_db"]))
    main_matches, _, _ = registry_matches(resolve_path(config["main_db"]))
    return pd.DataFrame(
        [
            {
                "csv_count": len(csv_paths),
                "csv_bytes": sum(path.stat().st_size for path in csv_paths),
                "partial_registry_matches": partial_matches,
                "partial_completed_matches": partial_completed,
                "partial_registry_orphan_count": len(partial_orphans),
                "partial_registry_orphans": ",".join(sorted(partial_orphans)),
                "main_registry_matches": main_matches,
                "config_hash_available": False,
                "acquisition_envelope_hash_available": False,
                "exact_locus_hash_available": False,
                "query_text_or_hash_available": False,
                "primary_analysis_eligible": False,
                "reason": (
                    "Tile/job/timestamp lineage exists for 385 CSVs in the partial registry, "
                    "but the acquisition query and config/envelope/locus hashes were not persisted."
                ),
            }
        ]
    )


def build_storage_sample_and_estimate(
    sources: pd.DataFrame,
    *,
    data_dir: Path,
    pool_config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Measure pilot parquet sizes and extrapolate planning-scale storage scenarios."""
    acquisition_columns = [
        column
        for column in [
            *BASE_COLUMNS,
            "galactic_l",
            "galactic_b",
            "ag_gspphot",
            "ebpminrp_gspphot",
            "region",
        ]
        if column in sources
    ]
    mask_columns = [
        "compatible_variant_mask",
        "compatible_variant_count",
        "passes_any_exact_variant",
    ]
    acquisition_dir = data_dir / "storage_sample" / "acquisition"
    eligible_dir = data_dir / "storage_sample" / "eligible"
    acquisition_mask_dir = data_dir / "storage_sample" / "acquisition_with_mask"
    for directory in [acquisition_dir, eligible_dir, acquisition_mask_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    for region, group in sources.groupby("region", sort=False):
        acquisition = group[acquisition_columns].copy()
        acquisition.to_parquet(
            acquisition_dir / f"{region}.parquet",
            index=False,
            compression="zstd",
        )
        with_mask = group[[*acquisition_columns, *mask_columns]].copy()
        with_mask.to_parquet(
            acquisition_mask_dir / f"{region}.parquet",
            index=False,
            compression="zstd",
        )
        eligible = with_mask[with_mask["passes_any_exact_variant"]].copy()
        eligible.to_parquet(
            eligible_dir / f"{region}.parquet",
            index=False,
            compression="zstd",
        )

    def directory_bytes(path: Path) -> int:
        return sum(file.stat().st_size for file in path.glob("*.parquet"))

    acquisition_bytes = directory_bytes(acquisition_dir)
    acquisition_mask_bytes = directory_bytes(acquisition_mask_dir)
    eligible_bytes = directory_bytes(eligible_dir)
    acquired_rows = len(sources)
    aggregate_rows = int(sources["aggregated_locus_keep"].sum())
    exact_rows = int(sources["passes_any_exact_variant"].sum())
    if not acquired_rows or not aggregate_rows or not exact_rows:
        raise ValueError("Storage estimation requires non-empty acquisition, aggregate and exact-union samples.")

    current_db = resolve_path(pool_config["output_db"])
    with duckdb.connect(str(current_db), read_only=True) as con:
        current_rows = int(con.execute("SELECT COUNT(*) FROM prediction_pool_sources").fetchone()[0])
    current_tile_dir = resolve_path(pool_config["output_dir"])
    current_bytes = sum(file.stat().st_size for file in current_tile_dir.rglob("*.parquet"))
    exact_ratio = exact_rows / aggregate_rows
    acquisition_ratio = acquired_rows / aggregate_rows
    estimated_exact_rows = int(round(current_rows * exact_ratio))
    estimated_acquisition_rows = int(round(current_rows * acquisition_ratio))
    acquisition_bpr = acquisition_bytes / acquired_rows
    acquisition_mask_bpr = acquisition_mask_bytes / acquired_rows
    eligible_bpr = eligible_bytes / exact_rows

    estimates = [
        {
            "scenario": "legacy_aggregate_actual",
            "primary_rows": current_rows,
            "secondary_rows": 0,
            "estimated_bytes": current_bytes,
            "basis": "observed current pool parquet inventory",
        },
        {
            "scenario": "eligible_only",
            "primary_rows": estimated_exact_rows,
            "secondary_rows": 0,
            "estimated_bytes": int(round(estimated_exact_rows * eligible_bpr)),
            "basis": "expanded-audit exact/aggregate ratio and eligible sample bytes/row",
        },
        {
            "scenario": "acquisition_plus_eligible",
            "primary_rows": estimated_acquisition_rows,
            "secondary_rows": estimated_exact_rows,
            "estimated_bytes": int(
                round(
                    estimated_acquisition_rows * acquisition_bpr
                    + estimated_exact_rows * eligible_bpr
                )
            ),
            "basis": "two physical parquet layers measured on expanded audit",
        },
        {
            "scenario": "acquisition_with_mask_logical_views",
            "primary_rows": estimated_acquisition_rows,
            "secondary_rows": 0,
            "estimated_bytes": int(round(estimated_acquisition_rows * acquisition_mask_bpr)),
            "basis": "one acquisition parquet layer plus uint64 mask/count/union columns",
        },
    ]
    table = pd.DataFrame(estimates)
    table["estimated_gb"] = table["estimated_bytes"] / 1_000_000_000
    table["estimated_gib"] = table["estimated_bytes"] / (1024**3)
    table["sample_acquisition_rows"] = acquired_rows
    table["sample_aggregate_rows"] = aggregate_rows
    table["sample_exact_union_rows"] = exact_rows
    table["sample_acquisition_bytes_per_row"] = acquisition_bpr
    table["sample_acquisition_mask_bytes_per_row"] = acquisition_mask_bpr
    table["sample_eligible_bytes_per_row"] = eligible_bpr
    return (
        {
            "acquisition_dir": str(acquisition_dir),
            "eligible_dir": str(eligible_dir),
            "acquisition_with_mask_dir": str(acquisition_mask_dir),
            "acquisition_sample_bytes": acquisition_bytes,
            "eligible_sample_bytes": eligible_bytes,
            "acquisition_with_mask_sample_bytes": acquisition_mask_bytes,
        },
        table,
    )


def _build_legacy_pilot_report_artifact(
    tables: dict[str, pd.DataFrame],
    *,
    report_dir: Path,
    run_id: str,
) -> Path:
    """Create the canonical bounded artifact consumed by the portable report builder."""
    generated_at = datetime.now(UTC).isoformat()
    report_rel = report_dir.relative_to(resolve_path(".")).as_posix()
    dataset_names = [
        "headline_metrics",
        "policy_by_region",
        "region_summary",
        "variant_summary",
        "magnitude_overall",
        "known_wr_summary",
        "known_wr_affected",
        "plane_discrepancy_summary",
        "legacy_lineage_summary",
    ]

    def source(source_id: str, table_name: str, label: str) -> dict[str, Any]:
        relative_path = f"{report_rel}/{table_name}.csv"
        return {
            "id": source_id,
            "label": label,
            "path": relative_path,
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "sql": f"SELECT * FROM read_csv_auto('{relative_path}', header = true)",
                "description": f"Reads the reviewed {table_name} output from pilot {run_id}.",
                "executed_at": generated_at,
                "tables_used": [relative_path],
                "filters": ["Five predeclared pilot sky boxes inside the acquisition envelope."],
            },
        }

    sources = [
        source("headline", "headline_metrics", "Pilot headline metrics"),
        source("policy_region", "policy_by_region", "Policy counts by sky region"),
        source("region", "region_summary", "Region-level comparison"),
        source("variant", "variant_summary", "Variant-level exact-locus comparison"),
        source("magnitude", "magnitude_overall", "Magnitude-level comparison"),
        source("known_wr", "known_wr_summary", "Known WR control summary"),
        source("known_wr_affected", "known_wr_affected", "Known WR controls affected"),
        source("plane", "plane_discrepancy_summary", "Plane discrepancy attribution"),
        source("legacy", "legacy_lineage_summary", "Legacy CSV lineage audit"),
    ]
    title = "Piloto de cobertura del locus en el prediction pool"
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": "Diagnóstico reproducible del filtro agregado frente a la unión exacta compatible.",
            "generatedAt": generated_at,
            "cards": [
                {
                    "id": "geometric_misses",
                    "dataset": "headline_metrics",
                    "sourceId": "headline",
                    "description": "Fuentes adquiridas que pasan al menos un locus exacto pero no el agregado.",
                    "metrics": [
                        {
                            "label": "Pasan locus exacto / falla agregado",
                            "field": "geometric_exact_not_aggregate",
                            "format": "number",
                        }
                    ],
                },
                {
                    "id": "compatible_additions",
                    "dataset": "headline_metrics",
                    "sourceId": "headline",
                    "description": "Fuentes compatibles con al menos una variante exacta que el agregado perdería.",
                    "metrics": [
                        {
                            "label": "Adiciones exactas compatibles",
                            "field": "compatible_exact_not_aggregate",
                            "format": "number",
                        }
                    ],
                },
                {
                    "id": "net_growth",
                    "dataset": "headline_metrics",
                    "sourceId": "headline",
                    "description": "Cambio neto del tamaño frente al filtro agregado dentro del piloto.",
                    "metrics": [
                        {
                            "label": "Crecimiento neto de la unión exacta",
                            "field": "net_exact_union_growth",
                            "format": "percent",
                            "signed": True,
                        }
                    ],
                },
                {
                    "id": "known_wr_rescued",
                    "dataset": "known_wr_summary",
                    "sourceId": "known_wr",
                    "description": "Controles WR que conserva la unión exacta y pierde el agregado.",
                    "metrics": [
                        {
                            "label": "WR conocidas recuperadas",
                            "field": "exact_union_not_current",
                            "format": "number",
                        }
                    ],
                },
            ],
            "charts": [
                {
                    "id": "policy_counts",
                    "title": "Fuentes conservadas por política y región",
                    "subtitle": "La unión exacta compatible supera al agregado en las cinco cajas.",
                    "type": "bar",
                    "dataset": "policy_by_region",
                    "sourceId": "policy_region",
                    "encodings": {
                        "x": {"field": "region", "type": "nominal", "label": "Región"},
                        "y": {"field": "sources", "type": "quantitative", "label": "Fuentes"},
                        "color": {"field": "policy", "type": "nominal", "label": "Política"},
                        "tooltip": [
                            {"field": "region", "type": "nominal"},
                            {"field": "policy", "type": "nominal"},
                            {"field": "sources", "type": "quantitative"},
                        ],
                    },
                },
                {
                    "id": "density_risk",
                    "title": "Pérdida exacta compatible frente a densidad adquirida",
                    "subtitle": "La tasa no es constante; el campo interior l≈30° es el más afectado.",
                    "type": "scatter",
                    "dataset": "region_summary",
                    "sourceId": "region",
                    "encodings": {
                        "x": {
                            "field": "acquisition_density_per_deg2",
                            "type": "quantitative",
                            "label": "Fuentes adquiridas / deg²",
                        },
                        "y": {
                            "field": "exact_variant_not_aggregate_pct_exact_union",
                            "type": "quantitative",
                            "format": "percent",
                            "label": "Pérdida sobre unión exacta",
                        },
                        "size": {"field": "acquisition_rows", "type": "quantitative"},
                        "label": {"field": "region", "type": "nominal"},
                    },
                },
                {
                    "id": "variant_risk",
                    "title": "Fuentes exactas rechazadas por el agregado, por variante",
                    "subtitle": "El impacto es mayor en variantes fotométricas y parallax_soft.",
                    "type": "horizontalBar",
                    "dataset": "variant_summary",
                    "sourceId": "variant",
                    "encodings": {
                        "x": {"field": "variant", "type": "nominal", "label": "Variante"},
                        "y": {
                            "field": "exact_variant_not_aggregate_pct",
                            "type": "quantitative",
                            "format": "percent",
                            "label": "Fracción rechazada",
                        },
                    },
                },
                {
                    "id": "magnitude_risk",
                    "title": "Pérdida exacta compatible por magnitud G",
                    "subtitle": "La discrepancia aumenta hacia fuentes débiles en los campos densos.",
                    "type": "line",
                    "dataset": "magnitude_overall",
                    "sourceId": "magnitude",
                    "encodings": {
                        "x": {"field": "magnitude_bin", "type": "ordinal", "label": "Magnitud G"},
                        "y": {
                            "field": "exact_variant_not_aggregate_pct_exact_union",
                            "type": "quantitative",
                            "format": "percent",
                            "label": "Fracción de la unión exacta",
                        },
                    },
                },
            ],
            "tables": [
                {
                    "id": "region_detail",
                    "title": "Resultados por región",
                    "dataset": "region_summary",
                    "sourceId": "region",
                    "defaultSort": {"field": "acquisition_rows", "direction": "desc"},
                    "columns": [
                        {"field": "region", "label": "Región"},
                        {"field": "acquisition_rows", "label": "Adquiridas", "format": "number"},
                        {"field": "aggregated_keep", "label": "Agregado", "format": "number"},
                        {"field": "any_exact_variant_keep", "label": "Unión exacta", "format": "number"},
                        {
                            "field": "exact_variant_not_aggregate",
                            "label": "Exactas perdidas",
                            "format": "number",
                        },
                        {
                            "field": "exact_variant_not_aggregate_pct_exact_union",
                            "label": "% unión perdida",
                            "format": "percent",
                        },
                    ],
                },
                {
                    "id": "wr_detail",
                    "title": "Controles WR afectados por el agregado",
                    "dataset": "known_wr_affected",
                    "sourceId": "known_wr_affected",
                    "defaultSort": {"field": "wr_id", "direction": "asc"},
                    "columns": [
                        {"field": "wr_id", "label": "WR"},
                        {"field": "Spectral Type", "label": "Tipo espectral"},
                        {"field": "G", "label": "G", "format": "number"},
                        {"field": "BP_RP", "label": "BP−RP", "format": "number"},
                        {"field": "J_K", "label": "J−Ks", "format": "number"},
                        {
                            "field": "aggregated_outlier_plane_count",
                            "label": "Planos fuera (agregado)",
                            "format": "number",
                        },
                    ],
                },
            ],
            "sources": [{"id": item["id"], "label": item["label"], "path": item["path"]} for item in sources],
            "blocks": [
                {"id": "title", "type": "markdown", "body": f"# {title}"},
                {
                    "id": "technical_summary",
                    "type": "markdown",
                    "sourceId": "headline",
                    "body": (
                        "## Resumen técnico\n\n"
                        "El filtro agregado actual **no conserva la unión de los locus exactos**. "
                        "En 9.102 fuentes adquiridas, 333 pasan algún locus exacto pero fallan el agregado; "
                        "327 además son compatibles con al menos una variante. La unión exacta compatible "
                        "crece de 5.858 a 6.151 fuentes: **+293 netas (+5,0%)**."
                    ),
                },
                {
                    "id": "headline_strip",
                    "type": "metric-strip",
                    "cardIds": ["geometric_misses", "compatible_additions", "net_growth", "known_wr_rescued"],
                },
                {
                    "id": "recommendation",
                    "type": "markdown",
                    "body": (
                        "## Recomendación\n\n"
                        "El pool actual **no es suficiente** para los modelos previstos porque perdió fuentes "
                        "que cumplen su locus operacional, incluidas dos WR de control. No iniciar scoring masivo. "
                        "Mantener el pipeline condicionado al locus y reemplazar el promedio por la **unión lógica "
                        "de variantes exactas compatibles**, versionada y persistida por fuente. La opción de sólo "
                        "acquisition envelope es el respaldo de máxima sensibilidad, pero en el piloto aumenta "
                        "el volumen un 55,4% frente al agregado."
                    ),
                },
                {"id": "policy_chart_block", "type": "chart", "chartId": "policy_counts"},
                {
                    "id": "known_controls",
                    "type": "markdown",
                    "sourceId": "known_wr",
                    "body": (
                        "## Controles WR confirman el riesgo científico\n\n"
                        "De 347 controles relaxed_photometry, 343 están dentro de la acquisition envelope. "
                        "La política actual conserva 332 y la unión exacta 334. Las dos recuperadas son "
                        "WR 122-15 (WN6) y WR 157 (WN5o(+B1II))."
                    ),
                },
                {"id": "wr_table_block", "type": "table", "tableId": "wr_detail"},
                {
                    "id": "geometry",
                    "type": "markdown",
                    "sourceId": "plane",
                    "body": (
                        "## La diferencia se concentra en los planos 2MASS con H−Ks\n\n"
                        "Entre las 333 discrepancias geométricas, 328 son outliers del agregado en J_H–H_K "
                        "y 326 en J_K–H_K. Esto demuestra que promediar pendiente e intercepto y tomar el "
                        "umbral máximo no equivale a envolver la unión de bandas exactas. Los PNG del piloto "
                        "muestran las fuentes discrepantes junto a esas fronteras."
                    ),
                },
                {"id": "variant_chart_block", "type": "chart", "chartId": "variant_risk"},
                {"id": "density_chart_block", "type": "chart", "chartId": "density_risk"},
                {"id": "magnitude_chart_block", "type": "chart", "chartId": "magnitude_risk"},
                {"id": "region_table_block", "type": "table", "tableId": "region_detail"},
                {
                    "id": "scope_data",
                    "type": "markdown",
                    "body": (
                        "## Alcance, datos y definiciones\n\n"
                        "El piloto usa cinco cajas persistentes: centro galáctico/extinción alta, plano interior "
                        "l≈30°, Cygnus l≈80°, anticentro l≈180° y una región de alta latitud/baja densidad. "
                        "`passes_any_exact_locus` mide sólo geometría. `passes_any_exact_variant` añade calidad "
                        "fotométrica y condición astrométrica de la variante. La alternativa 2 usa esta última."
                    ),
                },
                {
                    "id": "methodology",
                    "type": "markdown",
                    "body": (
                        "## Metodología reproducible\n\n"
                        "Cada consulta aplica únicamente la acquisition envelope ADQL vigente y conserva su texto, "
                        "CSV, job de Gaia y SHA-256. Localmente se reconstruye literalmente el agregado actual "
                        "y se evalúan ocho loci exactos. El comparador fue reconciliado contra el pool operativo: "
                        "no apareció ninguna fuente del pool que el piloto clasificara fuera del agregado; las seis "
                        "filas adicionales del piloto corresponden al paso posterior de exclusión de fuentes conocidas."
                    ),
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": (
                        "## Limitaciones y robustez\n\n"
                        "Las cajas son deliberadamente representativas, no una muestra aleatoria del cielo; por ello "
                        "no se extrapola el +5,0% directamente a 58 millones. La extinción se seleccionó por línea "
                        "de visión y no se incorporó aún un mapa cuantitativo. Los controles WR no son una validación "
                        "independiente porque contribuyen a definir los locus, pero sí son una prueba de regresión "
                        "válida de la política de adquisición."
                    ),
                },
                {
                    "id": "legacy_lineage",
                    "type": "markdown",
                    "sourceId": "legacy",
                    "body": (
                        "## Los 386 CSV parciales no tienen linaje suficiente\n\n"
                        "Hay 386 CSV (2,50 GB): 385 enlazan con teselas `completed` del DuckDB parcial y uno es "
                        "`test_simple`. Ninguno enlaza con el registro del pool principal ni conserva el texto/hash "
                        "de consulta, configuración, envelope o locus. Se usan sólo como control auxiliar; las cinco "
                        "consultas nuevas son la evidencia primaria. En la caja común, el CSV histórico y la nueva "
                        "consulta coinciden en los 314 source_id (Jaccard 1,0)."
                    ),
                },
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "body": (
                        "## Próximos pasos\n\n"
                        "1. Versionar la definición operacional y persistir por fuente el bitmask de variantes exactas, "
                        "`passes_any_exact_variant`, `source_hash_v1` y hashes de configuración/locus.\n"
                        "2. Ejecutar una auditoría independiente más amplia antes de reconstruir.\n"
                        "3. Si se confirma, reconstruir con unión exacta y documentar métricas within-locus.\n"
                        "4. Después corregir thresholding OOF/calibración, añadir sampler `none`, top-K y candidatos/WR, "
                        "y finalmente habilitar scoring e integración Streamlit."
                    ),
                },
                {
                    "id": "questions",
                    "type": "markdown",
                    "body": (
                        "## Preguntas para la auditoría ampliada\n\n"
                        "¿Se mantiene el patrón en más campos del bulbo y del plano? ¿Qué parte del efecto se asocia "
                        "a crowding/extinción frente a errores 2MASS? ¿Conviene conservar también el bit de fuentes "
                        "aceptadas por el agregado pero no por ninguna variante exacta para análisis de transición?"
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {name: _frame_records(tables[name]) for name in dataset_names},
        },
        "sources": sources,
    }
    path = report_dir / "artifact.json"
    _write_json(path, artifact)
    return path


def build_report_artifact(
    tables: dict[str, pd.DataFrame],
    *,
    report_dir: Path,
    run_id: str,
) -> Path:
    """Build the dynamic report for the expanded, spatially stratified audit."""
    generated_at = datetime.now(UTC).isoformat()
    report_rel = report_dir.relative_to(resolve_path(".")).as_posix()
    dataset_names = [
        "headline_metrics",
        "policy_by_region",
        "region_summary",
        "variant_summary",
        "variant_by_region",
        "variant_by_magnitude",
        "variant_wr_affected",
        "variant_overlap",
        "variant_contract",
        "geometric_containment",
        "synthetic_counterexamples",
        "magnitude_overall",
        "extinction_summary",
        "stratified_summary",
        "known_wr_summary",
        "known_wr_affected",
        "plane_discrepancy_summary",
        "legacy_lineage_summary",
        "storage_estimate",
    ]

    def source(source_id: str, table_name: str, label: str) -> dict[str, Any]:
        relative_path = f"{report_rel}/{table_name}.csv"
        return {
            "id": source_id,
            "label": label,
            "path": relative_path,
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "sql": f"SELECT * FROM read_csv_auto('{relative_path}', header = true)",
                "description": f"Reads reviewed output {table_name} from audit {run_id}.",
                "executed_at": generated_at,
                "tables_used": [relative_path],
                "filters": ["Predeclared stratified sky boxes inside the acquisition envelope."],
            },
        }

    source_specs = {
        "headline": ("headline_metrics", "Headline metrics"),
        "policy_region": ("policy_by_region", "Policy counts by region"),
        "region": ("region_summary", "Region comparison"),
        "variant": ("variant_summary", "Variant comparison"),
        "variant_region": ("variant_by_region", "Variant by region matrix"),
        "variant_magnitude": ("variant_by_magnitude", "Variant by magnitude matrix"),
        "variant_wr": ("variant_wr_affected", "Known WR affected by variant"),
        "variant_overlap": ("variant_overlap", "Compatible-variant overlap matrix"),
        "variant_contract": ("variant_contract", "Versioned variant-bit contract"),
        "containment": ("geometric_containment", "Analytic and constructive containment audit"),
        "counterexamples": ("synthetic_counterexamples", "Envelope-valid geometric witnesses"),
        "magnitude": ("magnitude_overall", "Magnitude comparison"),
        "extinction": ("extinction_summary", "Extinction-proxy comparison"),
        "strata": ("stratified_summary", "Spatial-stratum comparison"),
        "known_wr": ("known_wr_summary", "Known WR summary"),
        "known_wr_affected": ("known_wr_affected", "Known WR affected"),
        "plane": ("plane_discrepancy_summary", "Plane discrepancy attribution"),
        "legacy": ("legacy_lineage_summary", "Legacy CSV lineage"),
        "storage": ("storage_estimate", "Measured storage scenarios"),
    }
    sources = [source(source_id, table_name, label) for source_id, (table_name, label) in source_specs.items()]

    headline = tables["headline_metrics"].iloc[0]
    regions = tables["region_summary"]
    variants = tables["variant_summary"]
    controls = tables["known_wr_summary"].iloc[0]
    containment = tables["geometric_containment"]
    lineage = tables["legacy_lineage_summary"].iloc[0]
    storage = tables["storage_estimate"].set_index("scenario")
    acquired = int(headline["acquisition_sources"])
    aggregate_keep = int(headline["current_aggregate_keep"])
    exact_keep = int(headline["compatible_exact_union_keep"])
    compatible_missed = int(headline["compatible_exact_not_aggregate"])
    exact_loss_rate = compatible_missed / exact_keep if exact_keep else math.nan
    affected_variants = int(variants["exact_variant_not_aggregate"].gt(0).sum())
    geometric_counterexample_variants = int(containment["constructive_counterexample_found"].sum())
    max_region = regions.loc[regions["exact_variant_not_aggregate_pct_exact_union"].idxmax()]
    max_variant = variants.loc[variants["exact_variant_not_aggregate_pct"].idxmax()]
    mask_gib = float(storage.loc["acquisition_with_mask_logical_views", "estimated_gib"])
    two_layer_gib = float(storage.loc["acquisition_plus_eligible", "estimated_gib"])
    variant_decision = (
        f"{affected_variants} de {len(variants)} variantes pierden fuentes Gaia observadas. "
        f"Además, {geometric_counterexample_variants} de {len(variants)} tienen contraejemplos de color "
        "dentro de la envelope; por tanto, ninguna variante tiene garantía de cobertura completa en el legado."
    )
    title = "Auditoría espacial de la unión exacta del prediction pool"
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": "Auditoría reproducible del agregado frente a la unión exacta compatible.",
            "generatedAt": generated_at,
            "cards": [
                {
                    "id": "acquisition",
                    "dataset": "headline_metrics",
                    "sourceId": "headline",
                    "description": "Fuentes Gaia adquiridas dentro de la envelope.",
                    "metrics": [{"label": "Adquiridas", "field": "acquisition_sources", "format": "number"}],
                },
                {
                    "id": "compatible_misses",
                    "dataset": "headline_metrics",
                    "sourceId": "headline",
                    "description": "Fuentes de la unión exacta compatible rechazadas por el agregado.",
                    "metrics": [
                        {"label": "Exactas perdidas", "field": "compatible_exact_not_aggregate", "format": "number"}
                    ],
                },
                {
                    "id": "net_growth",
                    "dataset": "headline_metrics",
                    "sourceId": "headline",
                    "description": "Cambio neto de la unión frente al agregado.",
                    "metrics": [
                        {
                            "label": "Crecimiento neto",
                            "field": "net_exact_union_growth",
                            "format": "percent",
                            "signed": True,
                        }
                    ],
                },
                {
                    "id": "known_wr_rescued",
                    "dataset": "known_wr_summary",
                    "sourceId": "known_wr",
                    "description": "Controles WR de la unión exacta rechazados por el agregado.",
                    "metrics": [
                        {"label": "WR recuperadas", "field": "exact_union_not_current", "format": "number"}
                    ],
                },
            ],
            "charts": [
                {
                    "id": "policy_counts",
                    "title": "Fuentes conservadas por política y región",
                    "subtitle": "Acquisition envelope, agregado actual y unión exacta compatible.",
                    "type": "bar",
                    "dataset": "policy_by_region",
                    "sourceId": "policy_region",
                    "encodings": {
                        "x": {"field": "region", "type": "nominal", "label": "Región"},
                        "y": {"field": "sources", "type": "quantitative", "label": "Fuentes"},
                        "color": {"field": "policy", "type": "nominal", "label": "Política"},
                    },
                },
                {
                    "id": "density_risk",
                    "title": "Discrepancia compatible frente a densidad adquirida",
                    "subtitle": "Tasa descriptiva por tesela; no es una FPR.",
                    "type": "scatter",
                    "dataset": "region_summary",
                    "sourceId": "region",
                    "encodings": {
                        "x": {
                            "field": "acquisition_density_per_deg2",
                            "type": "quantitative",
                            "label": "Fuentes adquiridas / deg²",
                        },
                        "y": {
                            "field": "exact_variant_not_aggregate_pct_exact_union",
                            "type": "quantitative",
                            "format": "percent",
                            "label": "Fracción discrepante",
                        },
                        "size": {"field": "acquisition_rows", "type": "quantitative"},
                        "label": {"field": "region", "type": "nominal"},
                    },
                },
                {
                    "id": "variant_risk",
                    "title": "Discrepancia por variante",
                    "subtitle": "Fuentes compatibles rechazadas por el agregado.",
                    "type": "horizontalBar",
                    "dataset": "variant_summary",
                    "sourceId": "variant",
                    "encodings": {
                        "x": {"field": "variant", "type": "nominal", "label": "Variante"},
                        "y": {
                            "field": "exact_variant_not_aggregate_pct",
                            "type": "quantitative",
                            "format": "percent",
                            "label": "Fracción rechazada",
                        },
                    },
                },
                {
                    "id": "magnitude_risk",
                    "title": "Discrepancia compatible por magnitud G",
                    "subtitle": "Tasa sobre la unión exacta compatible.",
                    "type": "line",
                    "dataset": "magnitude_overall",
                    "sourceId": "magnitude",
                    "encodings": {
                        "x": {"field": "magnitude_bin", "type": "ordinal", "label": "Magnitud G"},
                        "y": {
                            "field": "exact_variant_not_aggregate_pct_exact_union",
                            "type": "quantitative",
                            "format": "percent",
                            "label": "Fracción discrepante",
                        },
                    },
                },
            ],
            "tables": [
                {
                    "id": "region_detail",
                    "title": "Resultados por región",
                    "dataset": "region_summary",
                    "sourceId": "region",
                    "defaultSort": {"field": "exact_variant_not_aggregate_pct_exact_union", "direction": "desc"},
                    "columns": [
                        {"field": "region", "label": "Región"},
                        {"field": "galactic_l_center", "label": "l", "format": "number"},
                        {"field": "galactic_b_center", "label": "b", "format": "number"},
                        {"field": "density_stratum", "label": "Densidad"},
                        {"field": "acquisition_rows", "label": "Adquiridas", "format": "number"},
                        {"field": "any_exact_variant_keep", "label": "Unión exacta", "format": "number"},
                        {"field": "exact_variant_not_aggregate", "label": "Perdidas", "format": "number"},
                        {
                            "field": "exact_variant_not_aggregate_pct_exact_union",
                            "label": "% unión perdida",
                            "format": "percent",
                        },
                    ],
                },
                {
                    "id": "variant_detail",
                    "title": "Matriz resumida por variante",
                    "dataset": "variant_summary",
                    "sourceId": "variant",
                    "defaultSort": {"field": "exact_variant_not_aggregate_pct", "direction": "desc"},
                    "columns": [
                        {"field": "variant", "label": "Variante"},
                        {"field": "bit", "label": "Bit", "format": "number"},
                        {"field": "exact_variant_keep", "label": "Aceptadas", "format": "number"},
                        {"field": "exact_variant_not_aggregate", "label": "Perdidas", "format": "number"},
                        {"field": "exact_variant_not_aggregate_pct", "label": "% perdido", "format": "percent"},
                        {"field": "aggregate_not_exact_variant", "label": "Agregado fuera", "format": "number"},
                        {"field": "current_pool_sample_status", "label": "Estado"},
                    ],
                },
                {
                    "id": "wr_detail",
                    "title": "Controles WR afectados",
                    "dataset": "known_wr_affected",
                    "sourceId": "known_wr_affected",
                    "defaultSort": {"field": "wr_id", "direction": "asc"},
                    "columns": [
                        {"field": "wr_id", "label": "WR"},
                        {"field": "Spectral Type", "label": "Tipo espectral"},
                        {"field": "G", "label": "G", "format": "number"},
                        {"field": "BP_RP", "label": "BP−RP", "format": "number"},
                        {"field": "J_K", "label": "J−Ks", "format": "number"},
                    ],
                },
                {
                    "id": "containment_detail",
                    "title": "Garantía geométrica por variante",
                    "dataset": "geometric_containment",
                    "sourceId": "containment",
                    "defaultSort": {"field": "variant", "direction": "asc"},
                    "columns": [
                        {"field": "variant", "label": "Variante"},
                        {
                            "field": "analytic_contained_planes",
                            "label": "Planos contenidos",
                            "format": "number",
                        },
                        {"field": "analytic_total_planes", "label": "Planos", "format": "number"},
                        {
                            "field": "synthetic_exact_not_aggregate",
                            "label": "Testigos",
                            "format": "number",
                        },
                        {
                            "field": "constructive_counterexample_found",
                            "label": "Superset refutado",
                        },
                        {"field": "interpretation", "label": "Interpretación"},
                    ],
                },
                {
                    "id": "storage_detail",
                    "title": "Estimación de almacenamiento",
                    "dataset": "storage_estimate",
                    "sourceId": "storage",
                    "defaultSort": {"field": "estimated_gib", "direction": "asc"},
                    "columns": [
                        {"field": "scenario", "label": "Escenario"},
                        {"field": "primary_rows", "label": "Filas primarias", "format": "number"},
                        {"field": "secondary_rows", "label": "Filas secundarias", "format": "number"},
                        {"field": "estimated_gib", "label": "GiB estimados", "format": "number"},
                        {"field": "basis", "label": "Base"},
                    ],
                },
                {
                    "id": "strata_detail",
                    "title": "Discrepancia por estrato espacial",
                    "dataset": "stratified_summary",
                    "sourceId": "strata",
                    "defaultSort": {
                        "field": "exact_variant_not_aggregate_pct_exact_union",
                        "direction": "desc",
                    },
                    "columns": [
                        {"field": "dimension", "label": "Dimensión"},
                        {"field": "stratum", "label": "Estrato"},
                        {"field": "region_count", "label": "Teselas", "format": "number"},
                        {"field": "acquisition_rows", "label": "Adquiridas", "format": "number"},
                        {"field": "any_exact_variant_keep", "label": "Unión exacta", "format": "number"},
                        {"field": "exact_variant_not_aggregate", "label": "Perdidas", "format": "number"},
                        {
                            "field": "exact_variant_not_aggregate_pct_exact_union",
                            "label": "% discrepante",
                            "format": "percent",
                        },
                    ],
                },
                {
                    "id": "extinction_detail",
                    "title": "Discrepancia por proxy de extinción",
                    "dataset": "extinction_summary",
                    "sourceId": "extinction",
                    "defaultSort": {"field": "ag_gspphot_bin", "direction": "asc"},
                    "columns": [
                        {"field": "ag_gspphot_bin", "label": "Estrato ag_gspphot"},
                        {"field": "acquisition_rows", "label": "Adquiridas", "format": "number"},
                        {"field": "any_exact_variant_keep", "label": "Unión exacta", "format": "number"},
                        {"field": "exact_variant_not_aggregate", "label": "Perdidas", "format": "number"},
                        {
                            "field": "exact_variant_not_aggregate_pct_exact_union",
                            "label": "% discrepante",
                            "format": "percent",
                        },
                        {"field": "coverage_note", "label": "Interpretación"},
                    ],
                },
            ],
            "sources": [{"id": item["id"], "label": item["label"], "path": item["path"]} for item in sources],
            "blocks": [
                {"id": "title", "type": "markdown", "body": f"# {title}"},
                {
                    "id": "technical_summary",
                    "type": "markdown",
                    "sourceId": "headline",
                    "body": (
                        "## Resumen técnico\n\n"
                        f"La auditoría adquirió {acquired:,} fuentes en {len(regions)} teselas estratificadas. "
                        f"La unión exacta compatible admite {exact_keep:,}; el agregado pierde {compatible_missed:,} "
                        f"de ellas (**{exact_loss_rate:.2%}**) y conserva {aggregate_keep:,}. La mayor discrepancia "
                        f"regional es `{max_region['region']}` "
                        f"({max_region['exact_variant_not_aggregate_pct_exact_union']:.2%})."
                    ),
                },
                {
                    "id": "headline_strip",
                    "type": "metric-strip",
                    "cardIds": ["acquisition", "compatible_misses", "net_growth", "known_wr_rescued"],
                },
                {
                    "id": "decision",
                    "type": "markdown",
                    "body": (
                        "## Decisión recomendada\n\n"
                        f"{variant_decision} Crear un **pool paralelo** desde la acquisition envelope, con unión "
                        "exacta versionada y bitmask por fuente; conservar el pool actual como legado "
                        "`aggregated_mean_fit` y no sobrescribirlo. La cobertura lógica objetivo es completa, aunque "
                        "la ejecución puede priorizar regiones. No iniciar scoring definitivo sobre el legado."
                    ),
                },
                {"id": "policy_chart_block", "type": "chart", "chartId": "policy_counts"},
                {"id": "variant_chart_block", "type": "chart", "chartId": "variant_risk"},
                {"id": "variant_table_block", "type": "table", "tableId": "variant_detail"},
                {"id": "containment_table_block", "type": "table", "tableId": "containment_detail"},
                {"id": "density_chart_block", "type": "chart", "chartId": "density_risk"},
                {"id": "magnitude_chart_block", "type": "chart", "chartId": "magnitude_risk"},
                {"id": "region_table_block", "type": "table", "tableId": "region_detail"},
                {"id": "strata_table_block", "type": "table", "tableId": "strata_detail"},
                {"id": "extinction_table_block", "type": "table", "tableId": "extinction_detail"},
                {
                    "id": "controls",
                    "type": "markdown",
                    "sourceId": "known_wr",
                    "body": (
                        "## Controles WR\n\n"
                        f"La unión exacta recupera {int(controls['exact_union_not_current'])} controles WR dentro "
                        "de la envelope que el agregado rechaza. Es una prueba de regresión de la política, no una "
                        "estimación poblacional independiente."
                    ),
                },
                {"id": "wr_table_block", "type": "table", "tableId": "wr_detail"},
                {
                    "id": "scope",
                    "type": "markdown",
                    "body": (
                        "## Alcance, datos y definiciones\n\n"
                        f"Se evaluaron {len(regions)} cajas persistentes estratificadas por longitud y latitud "
                        "galácticas, densidad, magnitud, disco interior/exterior y una proxy incompleta de extinción. "
                        "`passes_any_exact_locus` mide geometría; `passes_any_exact_variant` añade calidad "
                        "fotométrica y condición astrométrica. Las fuentes Gaia desconocidas permanecen no "
                        "etiquetadas: se informan discrepancias y densidades admitidas, nunca FPR real."
                    ),
                },
                {
                    "id": "methods",
                    "type": "markdown",
                    "body": (
                        "## Métodos reproducibles\n\n"
                        "Cada consulta aplica únicamente la acquisition envelope ADQL y conserva texto, SHA-256, "
                        "CSV y Gaia job id. Después se evalúan literalmente los ocho loci junto con condiciones "
                        "fotométricas y astrométricas. El contrato de bits enlaza cada variante con su "
                        "`locus_run_id`, hash y colores requeridos."
                    ),
                },
                {
                    "id": "geometry",
                    "type": "markdown",
                    "sourceId": "plane",
                    "body": (
                        "## Diagnóstico geométrico\n\n"
                        "Promediar pendientes e interceptos y combinar umbrales no produce la envolvente de una "
                        "unión de bandas lineales. Los gráficos de frontera identifican las fuentes que quedan "
                        "dentro de algún locus exacto y fuera de la recta agregada."
                    ),
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": (
                        "## Limitaciones y robustez\n\n"
                        "Las teselas son estratificadas, no una muestra probabilística del cielo; la tasa puntual no "
                        "se extrapola como estimación celeste. `ag_gspphot` tiene cobertura incompleta y se usa sólo "
                        "como descriptor. Gaia lo define como extinción en banda G estimada por GSP-Phot y advierte "
                        "de degeneraciones temperatura-extinción en líneas de visión difíciles "
                        "([modelo de datos](https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_main_source_catalogue/ssec_dm_gaia_source.html), "
                        "[GSP-Phot](https://gea.esac.esa.int/archive/documentation/GDR3/Data_analysis/chap_cu8par/sec_cu8par_apsis/ssec_cu8par_apsis_gspphot.html)). "
                        "Los controles WR contribuyeron a definir los locus, por lo que validan regresiones de "
                        "implementación, no sensibilidad independiente."
                    ),
                },
                {
                    "id": "lineage",
                    "type": "markdown",
                    "sourceId": "legacy",
                    "body": (
                        "## Linaje histórico\n\n"
                        f"Se encontraron {int(lineage['csv_count'])} CSV "
                        f"({float(lineage['csv_bytes']) / 1e9:.2f} GB); "
                        f"{int(lineage['partial_completed_matches'])} enlazan con teselas completadas del registro "
                        "parcial, pero no conservan hashes de consulta, envelope, configuración ni locus. No son "
                        "evidencia primaria."
                    ),
                },
                {
                    "id": "storage",
                    "type": "markdown",
                    "body": (
                        "## Persistencia pre-filtro\n\n"
                        "La opción principal es un único Parquet de adquisición por tesela con bitmask y vistas "
                        "lógicas elegibles. Permite recalcular la política sin volver a Gaia y evita duplicación: "
                        f"la estimación es {mask_gib:.2f} GiB frente a {two_layer_gib:.2f} GiB para dos capas "
                        "físicas. El staging sólo se retira tras validar checksums, conteos y registro atómico."
                    ),
                },
                {"id": "storage_table_block", "type": "table", "tableId": "storage_detail"},
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "body": (
                        "## Próximos pasos\n\n"
                        "1. Congelar el contrato de bits y el manifiesto del nuevo build.\n"
                        "2. Construir el pool exacto en paralelo, con validación por tesela antes de retirar staging.\n"
                        "3. Mantener `aggregated_mean_fit` como artefacto legado de sólo auditoría.\n"
                        "4. Continuar después con OOF/calibración, sampler `none`, top-K, auditoría independiente e "
                        "integración en Streamlit."
                    ),
                },
                {
                    "id": "questions",
                    "type": "markdown",
                    "body": (
                        "## Preguntas de seguimiento\n\n"
                        f"La variante más afectada es `{max_variant['variant']}` "
                        f"({max_variant['exact_variant_not_aggregate_pct']:.2%}). En los primeros lotes del nuevo "
                        "build conviene fijar alertas por densidad, magnitud y cobertura de `ag_gspphot`."
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {name: _frame_records(tables[name]) for name in dataset_names},
        },
        "sources": sources,
    }
    path = report_dir / "artifact.json"
    _write_json(path, artifact)
    return path


def compare_legacy_acquisition(region: dict[str, Any], new_acquisition: pd.DataFrame) -> dict[str, object]:
    legacy_path = resolve_path(region["legacy_csv"])
    result: dict[str, object] = {
        "lineage_status": "insufficient_for_primary_analysis",
        "legacy_csv_path": str(legacy_path),
        "legacy_csv_sha256": _file_sha256(legacy_path) if legacy_path.exists() else None,
    }
    if not legacy_path.exists():
        result["error"] = "legacy_csv_missing"
        return result
    legacy = pd.read_csv(legacy_path)
    subset = legacy[
        legacy["ra"].between(float(region["ra_min"]), float(region["ra_max"]), inclusive="left")
        & legacy["dec"].between(float(region["dec_min"]), float(region["dec_max"]), inclusive="left")
    ].copy()
    new_ids = set(pd.to_numeric(new_acquisition["source_id"], errors="coerce").dropna().astype("int64"))
    legacy_ids = set(pd.to_numeric(subset["source_id"], errors="coerce").dropna().astype("int64"))
    result.update(
        {
            "legacy_subset_rows": len(subset),
            "new_rows": len(new_acquisition),
            "source_id_intersection": len(new_ids & legacy_ids),
            "source_id_union": len(new_ids | legacy_ids),
            "jaccard": len(new_ids & legacy_ids) / len(new_ids | legacy_ids) if new_ids | legacy_ids else np.nan,
        }
    )
    legacy_db = resolve_path(region["legacy_db"])
    if legacy_db.exists():
        with duckdb.connect(str(legacy_db), read_only=True) as con:
            tile_id = legacy_path.stem
            row = con.execute(
                "SELECT status, row_count, output_file, job_id, updated_at FROM prediction_pool_tiles WHERE tile_id = ?",
                [tile_id],
            ).fetchone()
        result["legacy_tile_registry"] = list(row) if row else None
    return result


def build_pilot_figures(
    sources: pd.DataFrame,
    *,
    exact_loci: dict[str, ExactLocus],
    envelope: dict[str, Any],
    report_dir: Path,
    magnitude_labels: list[str],
) -> list[Path]:
    paths: list[Path] = []
    summary = []
    for region, group in sources.groupby("region", sort=False):
        summary.append(
            {
                "region": region,
                "acquisition": len(group),
                "aggregate": int(group["aggregated_locus_keep"].sum()),
                "exact_union": int(group["passes_any_exact_variant"].sum()),
            }
        )
    chart = pd.DataFrame(summary).set_index("region")
    figure_width = max(11, len(chart) * 0.72)
    ax = chart[["acquisition", "aggregate", "exact_union"]].plot.bar(
        figsize=(figure_width, 5.5),
        color=["#9a9a9a", "#4e79a7", "#f28e2b"],
    )
    ax.set_title("Pilot source counts by filtering policy")
    ax.set_ylabel("Sources")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=55)
    ax.legend(title="Policy", labels=["Acquisition envelope", "Current aggregate", "Compatible exact-variant union"])
    ax.figure.tight_layout()
    path = report_dir / "policy_counts_by_region.png"
    ax.figure.savefig(path, dpi=180)
    plt.close(ax.figure)
    paths.append(path)

    magnitude = sources.copy()
    bins = [0, 12, 14, 16, 18, 30]
    magnitude["magnitude_bin"] = pd.cut(magnitude["G"], bins=bins, labels=magnitude_labels, right=False)
    mag = magnitude.groupby("magnitude_bin", observed=True).agg(
        exact_union=("passes_any_exact_variant", "sum"),
        missed=("exact_variant_not_aggregate", "sum"),
    )
    mag["rate"] = mag["missed"] / mag["exact_union"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(mag.index.astype(str), mag["rate"] * 100, color="#e15759", edgecolor="#333333")
    ax.set_title("Exact-locus sources rejected by the aggregate, by G magnitude")
    ax.set_ylabel("Percent of compatible exact union")
    ax.set_xlabel("G magnitude")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    path = report_dir / "discrepancy_rate_by_magnitude.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    discrepancy = sources[sources["exact_not_aggregate"]]
    plane_counts = []
    for fit in envelope["fits"]:
        name = f"{fit['x']}__{fit['y']}"
        agg_margin = f"aggregated_margin__{name}"
        exact_margin = f"best_exact_margin__{name}"
        if agg_margin in discrepancy and exact_margin in discrepancy:
            count = int((discrepancy[agg_margin].lt(0) & discrepancy[exact_margin].ge(0)).sum())
            plane_counts.append((name, count))
    for plane_name, _ in sorted(plane_counts, key=lambda item: item[1], reverse=True)[:3]:
        x_name, y_name = plane_name.split("__", 1)
        path = report_dir / f"plane_discrepancy__{plane_name}.png"
        _plot_plane_discrepancy(
            sources,
            x_name=x_name,
            y_name=y_name,
            exact_loci=exact_loci,
            envelope=envelope,
            path=path,
        )
        paths.append(path)

    variants = list(exact_loci)
    region_variant_rows = []
    for (region, group) in sources.groupby("region", sort=False):
        aggregate = group["aggregated_locus_keep"].fillna(False).astype(bool)
        for variant in variants:
            compatible = group[f"exact_variant_keep__{variant}"].fillna(False).astype(bool)
            denominator = int(compatible.sum())
            region_variant_rows.append(
                {
                    "region": region,
                    "variant": variant,
                    "loss_rate": int((compatible & ~aggregate).sum()) / denominator if denominator else np.nan,
                }
            )
    region_variant = pd.DataFrame(region_variant_rows).pivot(
        index="variant",
        columns="region",
        values="loss_rate",
    )
    region_order = list(dict.fromkeys(sources["region"].astype(str)))
    region_variant = region_variant.reindex(index=variants, columns=region_order)
    path = report_dir / "variant_by_region_loss_heatmap.png"
    _plot_rate_heatmap(
        region_variant,
        title="Compatible exact-union discrepancy by variant and region",
        xlabel="Region",
        ylabel="Variant",
        path=path,
    )
    paths.append(path)

    magnitude_variant_rows = []
    for magnitude_bin, group in magnitude.groupby("magnitude_bin", observed=True, sort=False):
        aggregate = group["aggregated_locus_keep"].fillna(False).astype(bool)
        for variant in variants:
            compatible = group[f"exact_variant_keep__{variant}"].fillna(False).astype(bool)
            denominator = int(compatible.sum())
            magnitude_variant_rows.append(
                {
                    "variant": variant,
                    "magnitude_bin": str(magnitude_bin),
                    "loss_rate": int((compatible & ~aggregate).sum()) / denominator if denominator else np.nan,
                }
            )
    magnitude_variant = pd.DataFrame(magnitude_variant_rows).pivot(
        index="variant",
        columns="magnitude_bin",
        values="loss_rate",
    )
    magnitude_variant = magnitude_variant.reindex(
        index=variants,
        columns=[label for label in magnitude_labels if label in magnitude_variant],
    )
    path = report_dir / "variant_by_magnitude_loss_heatmap.png"
    _plot_rate_heatmap(
        magnitude_variant,
        title="Compatible exact-union discrepancy by variant and G magnitude",
        xlabel="G magnitude",
        ylabel="Variant",
        path=path,
    )
    paths.append(path)

    compatible_masks = {
        variant: sources[f"exact_variant_keep__{variant}"].fillna(False).astype(bool)
        for variant in variants
    }
    overlap = pd.DataFrame(index=variants, columns=variants, dtype=float)
    for left in variants:
        for right in variants:
            intersection = int((compatible_masks[left] & compatible_masks[right]).sum())
            union = int((compatible_masks[left] | compatible_masks[right]).sum())
            overlap.loc[left, right] = intersection / union if union else np.nan
    path = report_dir / "variant_overlap_jaccard_heatmap.png"
    _plot_rate_heatmap(
        overlap,
        title="Jaccard overlap of compatible exact variants",
        xlabel="Variant",
        ylabel="Variant",
        path=path,
        value_label="Jaccard",
        value_max=1.0,
    )
    paths.append(path)

    spatial_rows = []
    for region, group in sources.groupby("region", sort=False):
        denominator = int(group["passes_any_exact_variant"].sum())
        spatial_rows.append(
            {
                "region": region,
                "l": float(group["region_galactic_l_center"].iloc[0]),
                "abs_b": abs(float(group["region_galactic_b_center"].iloc[0])),
                "density": len(group) / float(group["region_area_deg2"].iloc[0]),
                "loss_rate": int(group["exact_variant_not_aggregate"].sum()) / denominator if denominator else np.nan,
            }
        )
    spatial = pd.DataFrame(spatial_rows)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    scatter = ax.scatter(
        spatial["l"],
        spatial["loss_rate"] * 100,
        c=spatial["abs_b"],
        s=35 + 90 * spatial["density"] / spatial["density"].max(),
        cmap="viridis",
        alpha=0.85,
        edgecolors="#333333",
        linewidths=0.4,
    )
    for _, row in spatial.iterrows():
        ax.annotate(row["region"], (row["l"], row["loss_rate"] * 100), xytext=(3, 3), textcoords="offset points", fontsize=7)
    ax.set_title("Regional discrepancy across Galactic longitude")
    ax.set_xlabel("Galactic longitude l [deg]")
    ax.set_ylabel("Percent of compatible exact union rejected")
    ax.set_xlim(-5, 365)
    ax.set_ylim(bottom=0)
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Absolute Galactic latitude |b| [deg]")
    fig.tight_layout()
    path = report_dir / "spatial_discrepancy_by_galactic_longitude.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    extinction = sources[pd.to_numeric(sources["ag_gspphot"], errors="coerce").notna()].copy()
    if len(extinction) >= 20 and extinction["ag_gspphot"].nunique() >= 4:
        extinction["ag_quantile"] = pd.qcut(
            pd.to_numeric(extinction["ag_gspphot"], errors="coerce"),
            q=4,
            duplicates="drop",
        )
        extinction_summary = extinction.groupby("ag_quantile", observed=True).agg(
            exact_union=("passes_any_exact_variant", "sum"),
            missed=("exact_variant_not_aggregate", "sum"),
        )
        extinction_summary["loss_rate"] = extinction_summary["missed"] / extinction_summary["exact_union"]
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.bar(
            extinction_summary.index.astype(str),
            extinction_summary["loss_rate"] * 100,
            color="#59a14f",
            edgecolor="#333333",
        )
        ax.set_title("Discrepancy by Gaia GSP-Phot G-band extinction proxy")
        ax.set_xlabel("ag_gspphot quantile interval [mag]")
        ax.set_ylabel("Percent of compatible exact union rejected")
        ax.tick_params(axis="x", rotation=20)
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        path = report_dir / "discrepancy_by_ag_gspphot.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def _plot_rate_heatmap(
    matrix: pd.DataFrame,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    path: Path,
    value_label: str = "Discrepancy rate",
    value_max: float | None = None,
) -> None:
    """Render a compact annotated matrix without adding a plotting dependency."""
    values = matrix.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    vmax = value_max if value_max is not None else (float(finite.max()) if finite.size else 1.0)
    vmax = max(vmax, 1e-9)
    fig_width = max(7.5, 0.68 * len(matrix.columns) + 3.5)
    fig_height = max(5.0, 0.5 * len(matrix.index) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(values, aspect="auto", cmap="magma", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(matrix.columns)), labels=matrix.columns, rotation=55, ha="right")
    ax.set_yticks(range(len(matrix.index)), labels=matrix.index)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for row in range(len(matrix.index)):
        for column in range(len(matrix.columns)):
            value = values[row, column]
            label = "—" if not np.isfinite(value) else f"{value:.1%}"
            color = "white" if np.isfinite(value) and value < vmax * 0.65 else "black"
            ax.text(column, row, label, ha="center", va="center", fontsize=7, color=color)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(value_label)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_plane_discrepancy(
    sources: pd.DataFrame,
    *,
    x_name: str,
    y_name: str,
    exact_loci: dict[str, ExactLocus],
    envelope: dict[str, Any],
    path: Path,
) -> None:
    x = _signed_log1p(sources[x_name])
    y = _signed_log1p(sources[y_name])
    fig, ax = plt.subplots(figsize=(8.5, 6))
    base = ~sources["exact_not_aggregate"]
    ax.scatter(x[base], y[base], s=7, alpha=0.12, color="#6f6f6f", label="Other acquired sources")
    ax.scatter(
        x[sources["exact_not_aggregate"]],
        y[sources["exact_not_aggregate"]],
        s=16,
        alpha=0.75,
        color="#e15759",
        edgecolors="none",
        label="Exact union yes / aggregate no",
    )
    finite_x = x[np.isfinite(x)]
    grid = np.linspace(float(finite_x.quantile(0.01)), float(finite_x.quantile(0.99)), 250)
    aggregate_fit = next(fit for fit in envelope["fits"] if fit["x"] == x_name and fit["y"] == y_name)
    slope = (float(aggregate_fit["slope_min"]) + float(aggregate_fit["slope_max"])) / 2
    intercept = (float(aggregate_fit["intercept_min"]) + float(aggregate_fit["intercept_max"])) / 2
    threshold = float(aggregate_fit["threshold_max"])
    predicted = intercept + slope * grid
    ax.plot(grid, predicted, color="#4e79a7", linewidth=2.2, label="Current aggregate")
    ax.plot(grid, predicted + threshold, color="#4e79a7", linewidth=1.2, linestyle="--")
    ax.plot(grid, predicted - threshold, color="#4e79a7", linewidth=1.2, linestyle="--")
    colors = plt.cm.tab10.colors
    plotted = 0
    for variant, locus in exact_loci.items():
        plane = next((plane for plane in locus.planes if plane.x == x_name and plane.y == y_name), None)
        if plane is None:
            continue
        line = plane.intercept + plane.slope * grid
        color = colors[plotted % len(colors)]
        ax.plot(grid, line, color=color, linewidth=0.9, alpha=0.7, label=variant)
        ax.plot(grid, line + plane.threshold, color=color, linewidth=0.55, alpha=0.35, linestyle=":")
        ax.plot(grid, line - plane.threshold, color=color, linewidth=0.55, alpha=0.35, linestyle=":")
        plotted += 1
    ax.set_title(f"Aggregate vs exact loci: {x_name} / {y_name}")
    ax.set_xlabel(f"signed_log1p({x_name})")
    ax.set_ylabel(f"signed_log1p({y_name})")
    ax.legend(loc="best", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _signed_log1p(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return np.sign(numeric) * np.log1p(np.abs(numeric))


def _sky_box_area_deg2(tile: SkyTile) -> float:
    """Return the spherical area of an axis-aligned RA/Dec box in square degrees."""
    ra_width = math.radians(tile.ra_max - tile.ra_min)
    dec_band = math.sin(math.radians(tile.dec_max)) - math.sin(math.radians(tile.dec_min))
    return abs(ra_width * dec_band) * (180.0 / math.pi) ** 2


def _output_dir(config: dict[str, Any], key: str, run_id: str) -> Path:
    return resolve_path(str(config["outputs"][key]).format(run_id=run_id))


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": _file_sha256(path),
    }


def _json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return sha256(payload).hexdigest()


def _frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a bounded dataframe to JSON-safe records without stringifying numbers."""
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")
