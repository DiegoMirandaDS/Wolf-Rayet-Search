# Agent Context

## Project

Wolf-Rayet Detector builds reproducible datasets and model artifacts for finding Galactic Wolf-Rayet (WR) candidates. The original repository is `https://github.com/DiegoMirandaDS/Wolf-Rayet-Detector`.

Historical notebooks and scripts live under `legacy/` and should be inspected only for context. Reproducible code lives under `src/wr_detector`, configs under `configs/`, notebooks under `notebooks/`, generated data under `data/`, and generated reports/models under `reports/`.

## Agent Rules

- Do not modify `legacy/` except to inspect historical context.
- Keep generated parquet, raw snapshots, DuckDB files, model artifacts and reports out of version control unless explicitly requested.
- Keep `Avances.md` local and ignored by Git.
- Prefer reproducible CLI/config/tested Python over notebook-only logic.
- Treat notebooks as audit, EDA and visualization artifacts, not as sources of truth.
- When changing design, modelling, splitting, filtering, feature, or artifact behavior, document the decision in this file. If the decision matters to scientific users, also update `README.md` and/or `docs/modeling_decisions.md`.
- Before handing off, make docs match code/configs. Do not leave stale command examples, model lists, dataset variants or notebook descriptions.

## Phase 1: Reference And Negative Pipelines

Phase 1 created reproducible catalogue pipelines:

- GWRC/Crowther catalogue snapshot ingestion.
- Gaia DR3 source selection from explicit aliases.
- Gaia, 2MASS and WISE photometry enrichment.
- VizieR fallback matching for missing 2MASS/WISE rows.
- DuckDB reference database plus processed parquet exports.
- SIMBAD non-WR negative sample with known WR exclusions.
- Dataset variants are built from photometric family plus astrometric subset.

Photometric families:

- `strict`: Gaia `G/BP/RP`, 2MASS `J/H/Ks`, WISE `W1/W2`, requiring A-quality 2MASS/WISE required bands.
- `relaxed`: same required photometry, accepting A or B quality in required 2MASS/WISE bands.

Astrometric subsets:

- `photometry`: no parallax or parallax-over-error cut.
- `parallax_soft`: `parallax > 0`.
- `poe_1`, `poe_2`, `poe_3`, `poe_5`: `parallax > 0` plus increasing `parallax_over_error` thresholds.

## Phase 2: Color-Locus Cut

The color-locus pipeline is the source of truth; the notebook is for audit and visualization.

- The Phase 2 reference base decision is `poe_3` unless a later explicit decision changes it.
- Color-locus cuts use only intra-mission colors: Gaia (`G_BP`, `G_RP`, `BP_RP`) and 2MASS (`J_H`, `J_K`, `H_K`).
- WISE contributes `W1_W2` as an intra-mission feature, but WISE does not define a Phase 2 regression plane because W3/W4 are not used.
- Cross-mission colors such as Gaia-2MASS, Gaia-WISE or 2MASS-WISE must not be used for Phase 2 color-locus cuts.
- The export keeps all rows and adds diagnostics. It does not silently delete rows.
- Model training excludes RLR/color-locus outliers through `color_locus_keep`, while source exports retain all diagnostics.

## Phase 3: Modelling Before Prediction Pool

Treat WR detection as a candidate-ranking problem, not ordinary balanced classification. The prediction pool has tens of millions of rows, so low false-positive behavior matters more than threshold accuracy.

Current baseline training run:

- `run_20260608T194455984398Z_d1dee0800bd4`

Current default training grid in `configs/models.yaml`:

- Variants: `strict` and `relaxed` crossed with `photometry`, `parallax_soft`, `poe_2`, and `poe_3`.
- `poe_5` is excluded from the default sweep.
- Feature sets: `colors_parallax`, `colors_parallax_error`.
- Models: `random_forest`, `hist_gradient_boosting`, `xgboost`.
- Samplers: `smote`, `smote_enn`.
- Negative ratio: `10x` training negatives per WR by default.
- Total default sweep: `8 variants * 2 feature sets * 3 models * 2 samplers = 96` configurations.

Split policy:

- Holdout is stable by `source_id` hash and stratified by target class.
- Approximately 20% of WR and 20% of negatives enter holdout before negative reduction.
- All non-holdout WR go to train.
- Non-holdout negatives are sampled to `10x` WR for train.
- Non-selected non-holdout negatives become `threshold_calibration`.
- If split logic changes, regenerate reduced datasets with `reduce-negatives` before retraining.

Evaluation policy:

- Optimize ranking with average precision in `BayesSearchCV`.
- Report top-K WR recovery, average precision, PR/ROC curves, recall at fixed FPR, train/CV/holdout stability diagnostics and feature importance.
- Overfit alerts are multicomponent diagnostics: keep `cv_train_gap_f2` for continuity, but read overfit risk primarily from `overfit_warning_flag`, `overfit_risk_score`, `train_cv_gap_average_precision`, `cv_holdout_drop_average_precision`, `holdout_to_cv_average_precision_ratio` and component flags.
- Threshold precision floors flag risky operating points; they are not the only model-selection criterion.
- Prefer models that recover WR in short candidate lists while maintaining low-FPR behavior and plausible feature reliance.
- Use `threshold_calibration` as a false-positive stress-test pool; it contains negatives only.
- Second-layer validation with autoencoders, Deep SVDD or other one-class models should be treated as candidate re-ranking/compatibility scoring over top first-stage predictions. Start with enriched tabular features and non-deep one-class baselines; use WN/WC subtype-aware validation when `Spectral Type` is preserved, and treat WO as a stress-test group rather than a trainable subtype.
- The implemented second-layer baseline is `wr-detector train-second-layer --config configs/second_layer.yaml`; it fits WN/WC one-class validators with Gaussian mixture, one-class SVM, Isolation Forest and robust covariance, then evaluates holdout retention and negative pass rates.
- Gaia DR3 H-alpha is available for enrichment through `gaiadr3.astrophysical_parameters` columns such as `ew_espels_halpha` and `classlabel_espels`, but it is not persisted in local datasets yet. Do not make H-alpha required until a full coverage audit is implemented for negatives and prediction-pool rows.

## Model Explorer

The Streamlit Model Explorer (`wr-detector explore-models`) was rebuilt as a multipage app (2026-06-11):

- Entry point: `src/wr_detector/apps/model_explorer.py`; page/chart/widget code in `src/wr_detector/apps/explorer_ui/`; all query logic stays in tested modules `src/wr_detector/modeling/explorer.py` and `src/wr_detector/modeling/cases.py`. Pages must not embed SQL or business logic.
- Charts are Altair with `width="container"` and bounded heights; theming comes from Streamlit theme flags via `wr_detector.cli.EXPLORER_THEMES` presets (`--theme dracula|nebula|slate`, default `dracula`), not custom CSS.
- `wr_detector.modeling.cases` joins `model_predictions` with `wr_reference.duckdb` and `simbad_negative.duckdb` (resolved via `configs/paths.yaml`) for case-level review: per-source identity, WR broad subtype recovery, SIMBAD false-positive composition and cross-model case overlap. It degrades gracefully when the reference DBs are absent.
- Per-model case loads and run-wide overlap aggregations run as SQL in DuckDB (read-only); only aggregated or per-model frames reach pandas/Streamlit.
- Model detail renders PR/ROC/confusion live from synchronized predictions (`wr_detector.modeling.cases.precision_recall_points`/`roc_points`); saved matplotlib PNGs stay on disk as artifacts and are listed by path only.
- Validation layers (second layer now, third layer later) are integrated read-only through `wr_detector.modeling.layers.VALIDATION_LAYERS`: each layer is a spec pointing at its config's `run_dir_template` and results CSV. To add a future layer, append a `ValidationLayer` spec — no page changes needed unless its schema diverges. Layer runs are read from the history DB tables (`<layer>_runs`/`<layer>_results`) first, with CSV discovery as fallback for unsynced runs.

## Validation-Layer Rules

- Layer training is linked to **data lineage, not first-layer run ids**: every second-layer result row records `dataset_path`, `dataset_sha256`, `models_config_path`, `models_config_sha256`, `holdout_fraction`, `require_color_locus_keep` and `color_locus_excluded_rows`. Pair a layer run with a first-layer run only when the dataset hashes match.
- Future layers that consume upstream outputs (e.g. top-K candidates of a specific first-layer model) must record the upstream `run_id`/`result_id` they consumed, and operational layer pairings must be registered explicitly before prediction-pool scoring.
- Color-locus outliers are excluded from all modelling before the holdout split (`load_modeling_dataset` with `require_color_locus_keep`), so train/holdout/threshold_calibration in every reduced dataset are already clean. **Every layer (second, third, ...) must train and evaluate only on color-locus-kept rows**; the second layer enforces this with `apply_color_locus_keep` as a defensive guard.
- Second-layer results auto-sync to `second_layer_runs`/`second_layer_results` in `training_history.duckdb` (`outputs.auto_sync_training_history`, default on). Backfill CSV-only runs with `wr-detector sync-second-layer-history --run-id <id>`.
- First-layer threshold selection already uses the full `threshold_calibration` negative pool via `make_threshold_selection_scores`; probability calibration is deliberately not applied while the task is ranking.

## Run IDs And Artifacts

Model notebooks that inspect training results must select `TRAINING_RUN_ID` and load from `data/databases/training_history.duckdb`.

New training runs:

- Generate or accept `--run-id`.
- Write run-specific artifacts under `reports/modeling/runs/{run_id}/`.
- Update `reports/modeling/runs/{run_id}/model_training_results.csv` after every completed configuration.
- Are resumable with `--resume-run`, which skips completed configurations from the run CSV.
- Sync to `data/databases/training_history.duckdb` at the end when `auto_sync_training_history` is enabled.
- Keep `.joblib` and `.png` artifacts on disk; CSV/JSON sidecars can be compacted into DuckDB.

`reports/tables/model_training_results.csv` is a latest/global export for compatibility. DuckDB history is the canonical experiment store.

## Prediction Pool

Prediction-pool construction is separate from model training:

- `build-prediction-pool` queries Gaia in sky tiles and writes tile parquet plus `data/databases/prediction_pool.duckdb`.
- Known WR/SIMBAD sources are excluded by `source_id`.
- The current pool is large; do not train directly on it.
- Apply selected trained models only after model run selection and audit.
- Consumers should use `prediction_pool_sources` for source rows and `prediction_pool_effective_tiles` for the operational list of completed parquet tiles to score. Do not iterate raw `prediction_pool_tiles` without filtering; skipped parents with `error_message = subdivided_into_subtiles` are coverage-terminal parents replaced by completed subtiles.
- Use `prediction_pool_tile_coverage` to audit registered tiles. It labels rows as `effective_tile`, `subdivided_parent`, or `incomplete`.

## Notebook Purposes

- `00_Available_photometry_analysis.ipynb`: audit photometry availability and dataset-design tradeoffs.
- `01_reference_catalog_audit.ipynb`: audit reference database provenance, export sizes, match provenance and feature coverage.
- `02_color_eda_linear_cuts.ipynb`: audit robust color-locus regression assumptions and diagnostics.
- `03_negative_reduction_audit.ipynb`: audit reduced modelling datasets, train/calibration/holdout sizes and negative-reduction representativeness.
- `04_model_training_results.ipynb`: inspect trained runs by `TRAINING_RUN_ID`, rank models, compare top-K recovery, metrics and artifacts.
- `04_overfitting_and_tree_diagnostics.ipynb`: inspect overfitting, tree/boosting complexity, feature importance and saved validation artifacts by `TRAINING_RUN_ID`.
- `05_prediction_pool_audit.ipynb`: audit prediction-pool coverage, exclusions and distribution comparison before applying a model.

## Useful Commands

Full reproducible data preparation:

```powershell
wr-detector build-reference --config configs/reference.yaml
wr-detector audit-reference --db data/databases/wr_reference.duckdb
wr-detector export-reference-datasets --config configs/filters.yaml
wr-detector build-simbad-negative --config configs/simbad_negative.yaml
wr-detector export-simbad-negative-datasets --config configs/filters.yaml
wr-detector export-color-locus --config configs/filters.yaml
```

Current modelling run:

```powershell
wr-detector reduce-negatives --config configs/models.yaml
wr-detector train-models --config configs/models.yaml --run-id run_YYYYMMDD_science_v1
wr-detector train-models --config configs/models.yaml --run-id run_YYYYMMDD_science_v1 --resume-run
wr-detector train-second-layer --config configs/second_layer.yaml
```

Optional targeted or larger negative-ratio runs:

```powershell
wr-detector reduce-negatives --config configs/models.yaml --negative-ratio 20 --negative-ratio 50
wr-detector train-models --config configs/models.yaml --run-id run_YYYYMMDD_ratios --negative-ratio 20 --negative-ratio 50
```

History and cleanup:

```powershell
wr-detector sync-training-history --config configs/models.yaml --run-id run_YYYYMMDD_science_v1 --replace-run
wr-detector clean-model-artifacts --config configs/models.yaml --remove-db-backed-sidecars
wr-detector clean-model-artifacts --config configs/models.yaml --remove-db-backed-sidecars --apply
wr-detector normalize-training-history-paths --config configs/models.yaml
```

Prediction pool:

```powershell
wr-detector build-prediction-pool --config configs/prediction_pool.yaml --dry-run
wr-detector build-prediction-pool --config configs/prediction_pool.yaml
wr-detector audit-prediction-pool --db data/databases/prediction_pool.duckdb
```
