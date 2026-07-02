"""Typer CLI entry points for catalogue, modelling, history, and explorer workflows."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import typer

from wr_detector.db import audit_reference_database
from wr_detector.features import export_color_locus_datasets, export_reference_datasets, export_simbad_negative_datasets
from wr_detector.modeling import (
    cleanup_unreferenced_model_artifacts,
    normalize_training_history_paths,
    reduce_negative_variants,
    run_model_benchmark,
    sync_training_history,
    train_second_layer_validators,
    train_models,
)
from wr_detector.pipelines.prediction_pool import audit_prediction_pool, build_prediction_pool
from wr_detector.pipelines.reference import build_reference
from wr_detector.pipelines.simbad_negative import build_simbad_negative


app = typer.Typer(help="Wolf-Rayet Detector project CLI.")


EXPLORER_THEMES = {
    "dracula": {
        "primaryColor": "#bd93f9",
        "backgroundColor": "#282a36",
        "secondaryBackgroundColor": "#343746",
        "textColor": "#f8f8f2",
    },
    "nebula": {
        "primaryColor": "#9d7bff",
        "backgroundColor": "#161226",
        "secondaryBackgroundColor": "#221b38",
        "textColor": "#ece9f7",
    },
    "slate": {
        "primaryColor": "#4da3ff",
        "backgroundColor": "#0f131a",
        "secondaryBackgroundColor": "#171c26",
        "textColor": "#e6e9ef",
    },
}

DEFAULT_EXPLORER_THEME = "dracula"


def explorer_theme_options(theme: str) -> list[str]:
    if theme not in EXPLORER_THEMES:
        raise ValueError(f"Unknown explorer theme: {theme}. Available: {', '.join(EXPLORER_THEMES)}")
    options = ["--theme.base", "dark"]
    for key, value in EXPLORER_THEMES[theme].items():
        options.extend([f"--theme.{key}", value])
    return options


def streamlit_model_explorer_command(config: Path, port: int, theme: str = DEFAULT_EXPLORER_THEME) -> list[str]:
    app_path = Path(__file__).resolve().parent / "apps" / "model_explorer.py"
    return [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(port),
        *explorer_theme_options(theme),
        "--",
        "--config",
        str(config),
    ]


@app.command("build-reference")
def build_reference_command(
    config: Path = typer.Option(Path("configs/reference.yaml"), "--config", "-c"),
) -> None:
    result = build_reference(config)
    typer.echo("Reference pipeline completed.")
    for key, value in result.items():
        typer.echo(f"{key}: {value}")


@app.command("audit-reference")
def audit_reference_command(
    db: Path = typer.Option(Path("data/databases/wr_reference.duckdb"), "--db"),
) -> None:
    result = audit_reference_database(db)
    typer.echo("Reference DB audit")
    for table, count in result["counts"].items():
        typer.echo(f"{table}: {count}")
    typer.echo(f"duplicate_source_ids: {result['duplicate_source_ids']}")


@app.command("export-reference-datasets")
def export_reference_datasets_command(
    config: Path = typer.Option(Path("configs/filters.yaml"), "--config", "-c"),
) -> None:
    counts = export_reference_datasets(config)
    typer.echo("Reference datasets exported.")
    for name, count in counts.items():
        typer.echo(f"{name}: {count}")


@app.command("build-simbad-negative")
def build_simbad_negative_command(
    config: Path = typer.Option(Path("configs/simbad_negative.yaml"), "--config", "-c"),
) -> None:
    result = build_simbad_negative(config)
    typer.echo("SIMBAD negative pipeline completed.")
    for key, value in result.items():
        typer.echo(f"{key}: {value}")


@app.command("export-simbad-negative-datasets")
def export_simbad_negative_datasets_command(
    config: Path = typer.Option(Path("configs/filters.yaml"), "--config", "-c"),
) -> None:
    counts = export_simbad_negative_datasets(config)
    typer.echo("SIMBAD negative datasets exported.")
    for name, count in counts.items():
        typer.echo(f"{name}: {count}")


@app.command("export-color-locus")
def export_color_locus_command(
    config: Path = typer.Option(Path("configs/filters.yaml"), "--config", "-c"),
) -> None:
    result = export_color_locus_datasets(config)
    typer.echo("Color-locus datasets exported.")
    for key, value in result.items():
        if key != "candidate_fits":
            typer.echo(f"{key}: {value}")


@app.command("build-prediction-pool")
def build_prediction_pool_command(
    config: Path = typer.Option(Path("configs/prediction_pool.yaml"), "--config", "-c"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print planned tiles and sample ADQL without launching Gaia jobs."),
    max_tiles: int | None = typer.Option(None, "--max-tiles", help="Limit the number of sky tiles for smoke tests."),
    row_limit: int | None = typer.Option(None, "--row-limit", help="Add TOP N to every Gaia tile query."),
) -> None:
    result = build_prediction_pool(config, dry_run=dry_run, max_tiles=max_tiles, row_limit=row_limit)
    typer.echo("Prediction pool build prepared." if dry_run else "Prediction pool build completed.")
    for key, value in result.items():
        if key == "sample_adql":
            typer.echo("sample_adql:")
            typer.echo(value)
        else:
            typer.echo(f"{key}: {value}")


@app.command("audit-prediction-pool")
def audit_prediction_pool_command(
    db: Path = typer.Option(Path("data/databases/prediction_pool.duckdb"), "--db"),
) -> None:
    result = audit_prediction_pool(db)
    typer.echo("Prediction pool audit")
    typer.echo(f"db_path: {result['db_path']}")
    typer.echo(f"db_size_bytes: {result['db_size_bytes']}")
    typer.echo("counts:")
    for table, count in result["counts"].items():
        typer.echo(f"  {table}: {count}")
    typer.echo("tile_status:")
    for status, count in result["tile_status"].items():
        typer.echo(f"  {status}: {count}")
    typer.echo("exclusions:")
    for label, count in result["exclusions"].items():
        typer.echo(f"  {label}: {count}")


@app.command("benchmark-models")
def benchmark_models_command(
    config: Path = typer.Option(Path("configs/models.yaml"), "--config", "-c"),
) -> None:
    results = run_model_benchmark(config)
    typer.echo("Model benchmark completed.")
    typer.echo(f"experiments: {len(results)}")
    if not results.empty:
        best = results.iloc[0]
        typer.echo(
            "best: "
            f"{best['dataset_variant']} / {best['feature_set']} / {best['model']} "
            f"(holdout_f2={best['holdout_f2_wr']:.3f}, "
            f"precision={best['holdout_precision_wr']:.3f}, recall={best['holdout_recall_wr']:.3f})"
        )


@app.command("reduce-negatives")
def reduce_negatives_command(
    config: Path = typer.Option(Path("configs/models.yaml"), "--config", "-c"),
    variants: list[str] | None = typer.Option(None, "--variant", help="Dataset variant to reduce. Repeat to select multiple."),
    negative_ratios: list[str] | None = typer.Option(None, "--negative-ratio", help="Negative:WR ratio to export, e.g. 10, 20, 50, all_train_negatives. Repeat to select multiple."),
) -> None:
    summary, audit = reduce_negative_variants(config, variants=variants, negative_ratios=negative_ratios, verbose=True)
    typer.echo("Negative reduction completed.")
    typer.echo(f"variants: {len(summary)}")
    typer.echo(f"audit_rows: {len(audit)}")


@app.command("train-models")
def train_models_command(
    config: Path = typer.Option(Path("configs/models.yaml"), "--config", "-c"),
    variants: list[str] | None = typer.Option(None, "--variant", help="Dataset variant. Repeat to select multiple."),
    models: list[str] | None = typer.Option(None, "--model", help="Model key from config. Repeat to select multiple."),
    samplers: list[str] | None = typer.Option(None, "--sampler", help="Sampler key from config. Repeat to select multiple."),
    feature_sets: list[str] | None = typer.Option(None, "--feature-set", help="Feature-set key from config. Repeat to select multiple."),
    negative_ratios: list[str] | None = typer.Option(None, "--negative-ratio", help="Negative:WR ratio key. Repeat to select multiple."),
    n_iter: int | None = typer.Option(None, "--n-iter", help="Override BayesSearchCV iterations."),
    run_id: str | None = typer.Option(None, "--run-id", help="Stable training run id. Auto-generated when omitted."),
    resume_run: bool = typer.Option(False, "--resume-run", help="Resume a run by skipping completed configurations in its run CSV."),
) -> None:
    results = train_models(
        config,
        variants=variants,
        models=models,
        samplers=samplers,
        feature_sets=feature_sets,
        negative_ratios=negative_ratios,
        n_iter=n_iter,
        run_id=run_id,
        resume_run=resume_run,
        verbose=True,
    )
    typer.echo("Model training completed.")
    typer.echo(f"experiments: {len(results)}")
    if not results.empty:
        best = results.iloc[0]
        typer.echo(
            "best: "
            f"{best['dataset_variant']} / {best['feature_set']} / {best['model']} / {best['sampler']} "
            f"(holdout_f2={best['holdout_f2_wr']:.3f}, "
            f"precision={best['holdout_precision_wr']:.3f}, recall={best['holdout_recall_wr']:.3f}, "
            f"status={best['selection_status']})"
        )


@app.command("explore-models")
def explore_models_command(
    config: Path = typer.Option(Path("configs/models.yaml"), "--config", "-c"),
    port: int = typer.Option(8501, "--port", help="Local Streamlit server port."),
    theme: str = typer.Option(
        DEFAULT_EXPLORER_THEME,
        "--theme",
        help=f"Color theme: {', '.join(EXPLORER_THEMES)}.",
    ),
) -> None:
    if importlib.util.find_spec("streamlit") is None:
        raise typer.BadParameter('Streamlit is not installed. Install it with: pip install -e ".[viz]"')
    if theme not in EXPLORER_THEMES:
        raise typer.BadParameter(f"Unknown theme: {theme}. Available: {', '.join(EXPLORER_THEMES)}")
    typer.echo(f"Starting Model Explorer at http://localhost:{port} (theme: {theme})")
    subprocess.run(streamlit_model_explorer_command(config, port, theme), check=True)


@app.command("train-second-layer")
def train_second_layer_command(
    config: Path = typer.Option(Path("configs/second_layer.yaml"), "--config", "-c"),
    variants: list[str] | None = typer.Option(None, "--variant", help="Dataset variant. Repeat to select multiple."),
    feature_sets: list[str] | None = typer.Option(None, "--feature-set", help="Second-layer feature-set key. Repeat to select multiple."),
    methods: list[str] | None = typer.Option(None, "--method", help="Second-layer method key. Repeat to select multiple."),
    subtypes: list[str] | None = typer.Option(None, "--subtype", help="WR broad subtype such as WN or WC. Repeat to select multiple."),
    negative_ratios: list[str] | None = typer.Option(None, "--negative-ratio", help="Negative:WR ratio key. Repeat to select multiple."),
    run_id: str | None = typer.Option(None, "--run-id", help="Stable second-layer run id. Auto-generated when omitted."),
) -> None:
    results = train_second_layer_validators(
        config,
        variants=variants,
        feature_sets=feature_sets,
        methods=methods,
        subtypes=subtypes,
        negative_ratios=negative_ratios,
        run_id=run_id,
        verbose=True,
    )
    typer.echo("Second-layer validation completed.")
    typer.echo(f"experiments: {len(results)}")
    accepted = int(results["status"].eq("accepted").sum()) if "status" in results else 0
    typer.echo(f"accepted: {accepted}")
    if accepted:
        best = results[results["status"].eq("accepted")].iloc[0]
        typer.echo(
            "best: "
            f"{best['dataset_variant']} / {best['feature_set']} / {best['subtype']} / {best['method']} "
            f"(holdout_ap={best['holdout_average_precision']:.3f}, "
            f"positive_retention={best['holdout_positive_retention']:.3f}, "
            f"calibration_negative_pass={best['threshold_calibration_negative_pass_rate']:.3f})"
        )


@app.command("sync-second-layer-history")
def sync_second_layer_history_command(
    config: Path = typer.Option(Path("configs/second_layer.yaml"), "--config", "-c"),
    run_id: str = typer.Option(..., "--run-id", help="Second-layer run id (directory name under the layer runs dir)."),
    csv: Path | None = typer.Option(None, "--csv", help="Results CSV. Defaults to the run's results CSV."),
    replace_run: bool = typer.Option(False, "--replace-run", help="Replace an existing run with the same id."),
) -> None:
    import pandas as pd

    from wr_detector.modeling.history import sync_second_layer_history
    from wr_detector.modeling.second_layer import load_second_layer_config, second_layer_results_path

    layer_config = load_second_layer_config(config)
    csv_path = csv or second_layer_results_path(layer_config, run_id)
    if not Path(csv_path).exists():
        raise typer.BadParameter(f"Second-layer results CSV not found: {csv_path}")
    results = pd.read_csv(csv_path)
    result = sync_second_layer_history(
        layer_config["models_config"],
        results,
        run_id=run_id,
        source_csv=csv_path,
        config_path=str(config),
        replace_run=replace_run,
    )
    typer.echo("Second-layer history synchronized.")
    for key, value in result.items():
        typer.echo(f"{key}: {value}")


@app.command("sync-training-history")
def sync_training_history_command(
    config: Path = typer.Option(Path("configs/models.yaml"), "--config", "-c"),
    run_id: str | None = typer.Option(None, "--run-id", help="Stable run id. Auto-generated when omitted."),
    run_name: str | None = typer.Option(None, "--run-name", help="Human-readable training run name."),
    notes: str | None = typer.Option(None, "--notes", help="Optional notes stored in the DB."),
    replace_run: bool = typer.Option(False, "--replace-run", help="Replace an existing run with the same id."),
) -> None:
    result = sync_training_history(
        config,
        run_id=run_id,
        run_name=run_name,
        notes=notes,
        replace_run=replace_run,
    )
    typer.echo("Training history synchronized.")
    for key, value in result.items():
        typer.echo(f"{key}: {value}")


@app.command("clean-model-artifacts")
def clean_model_artifacts_command(
    config: Path = typer.Option(Path("configs/models.yaml"), "--config", "-c"),
    apply: bool = typer.Option(False, "--apply", help="Delete unreferenced artifacts. Without this flag, only reports what would be deleted."),
    remove_db_backed_sidecars: bool = typer.Option(
        False,
        "--remove-db-backed-sidecars",
        help="Also delete prediction CSVs, feature-importance CSVs and model JSON sidecars after they have been synchronized to DuckDB.",
    ),
) -> None:
    report = cleanup_unreferenced_model_artifacts(config, apply=apply, remove_db_backed_sidecars=remove_db_backed_sidecars)
    removable = report[~report["keep"]]
    size_mb = removable["size_bytes"].sum() / 1024 / 1024
    typer.echo("Model artifact cleanup report.")
    typer.echo(f"total_files: {len(report)}")
    typer.echo(f"unreferenced_files: {len(removable)}")
    typer.echo(f"unreferenced_size_mb: {size_mb:.2f}")
    if "reason" in removable:
        typer.echo("reasons:")
        for reason, count in removable["reason"].value_counts().items():
            typer.echo(f"  {reason}: {count}")
    typer.echo(f"deleted: {bool(apply)}")
    if not removable.empty:
        typer.echo("first_unreferenced:")
        for path in removable["path"].head(10):
            typer.echo(f"  {path}")


@app.command("normalize-training-history-paths")
def normalize_training_history_paths_command(
    config: Path = typer.Option(Path("configs/models.yaml"), "--config", "-c"),
) -> None:
    result = normalize_training_history_paths(config)
    typer.echo("Training history paths normalized.")
    typer.echo(f"db_path: {result['db_path']}")
    for table, rows in result["updated"].items():
        typer.echo(f"{table}: {rows}")


if __name__ == "__main__":
    app()
