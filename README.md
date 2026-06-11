# Wolf-Rayet Detector

Wolf-Rayet Detector is a reproducible Python pipeline for building a curated positive sample of known Galactic Wolf-Rayet (WR) stars, constructing a controlled SIMBAD non-WR negative sample, training candidate-ranking models, and preparing a Gaia-scale prediction pool for later candidate search.

The project is currently in a pre-prediction modelling stage: the goal is to select a scientifically defensible model and operating policy before scoring tens of millions of Gaia sources.

Historical exploratory work lives in `legacy/`. Reproducible code lives in `src/wr_detector`, configuration in `configs/`, notebooks in `notebooks/`, generated data in `data/`, and generated model/report artifacts in `reports/`.

## Scientific Scope

The current pipeline does not claim discovery of new WR stars by itself. It builds:

- a positive reference set from the Crowther/GWRC Galactic WR catalogue;
- Gaia DR3, 2MASS and WISE photometric/astrometric features;
- a SIMBAD non-WR negative sample with known WR exclusions;
- reproducible dataset variants for model comparison;
- model-selection diagnostics designed for rare-object candidate ranking;
- a tiled Gaia prediction pool that can be scored after model selection.

WR detection is treated as a **ranking problem**. In a Gaia-scale pool, even a small false-positive rate can generate many candidates, so top-K recovery, average precision and low-FPR behavior are more important than ordinary accuracy.

## Setup

```powershell
python -m pip install -e ".[dev]"
```

With Conda/Mamba:

```powershell
mamba env create -f environment.yml
mamba activate wolf-rayet-detector
python -m pip install -e .
```

Install optional XGBoost support for the current model sweep:

```powershell
python -m pip install -e ".[dev,modeling]"
```

## Data Pipelines

### Reference WR Catalogue

```powershell
wr-detector build-reference --config configs/reference.yaml
wr-detector audit-reference --db data/databases/wr_reference.duckdb
wr-detector export-reference-datasets --config configs/filters.yaml
```

`build-reference` downloads the GWRC/Crowther table, keeps catalogue rows with explicit Gaia DR3 aliases, queries Gaia DR3, enriches with 2MASS and WISE photometry, uses VizieR fallback matching where needed, and writes `data/databases/wr_reference.duckdb`.

### SIMBAD Negative Sample

```powershell
wr-detector build-simbad-negative --config configs/simbad_negative.yaml
wr-detector export-simbad-negative-datasets --config configs/filters.yaml
```

`build-simbad-negative` queries configured SIMBAD non-WR object types, keeps Gaia DR3-identified sources, excludes known WR objects, enriches with Gaia/2MASS/WISE, and exports the same variant matrix as the WR reference sample.

### Color-Locus Annotation

```powershell
wr-detector export-color-locus --config configs/filters.yaml
```

The color-locus step fits robust intra-mission color-color relations on WR reference variants and annotates both WR and SIMBAD-negative rows.

Important design choices:

- Gaia color planes use `G_BP`, `G_RP`, `BP_RP`.
- 2MASS color planes use `J_H`, `J_K`, `H_K`.
- WISE contributes `W1_W2` as a feature, but WISE does not define a Phase 2 color-color regression because W3/W4 are not used.
- Cross-mission colors such as Gaia-2MASS, Gaia-WISE and 2MASS-WISE are not used for Phase 2 color-locus cuts.
- The export keeps all rows and adds diagnostics; model training later filters with `color_locus_keep`.

## Dataset Variants

Variants combine a photometric family with an astrometric subset.

Photometric families:

- `strict`: Gaia `G/BP/RP`, 2MASS `J/H/Ks`, WISE `W1/W2`, requiring A-quality 2MASS/WISE required bands.
- `relaxed`: same required photometry, accepting A or B quality in required 2MASS/WISE bands.

Astrometric subsets:

- `photometry`: no parallax or parallax-over-error restriction.
- `parallax_soft`: `parallax > 0`.
- `poe_1`, `poe_2`, `poe_3`, `poe_5`: `parallax > 0` plus increasing `parallax_over_error` thresholds.

The current default modelling sweep uses:

- `strict_photometry`
- `strict_parallax_soft`
- `strict_poe_2`
- `strict_poe_3`
- `relaxed_photometry`
- `relaxed_parallax_soft`
- `relaxed_poe_2`
- `relaxed_poe_3`

`poe_5` is exported but excluded from the default model sweep because it is more restrictive and reduces the scarce WR positive sample.

## Modelling Workflow

### Negative Reduction

```powershell
wr-detector reduce-negatives --config configs/models.yaml
```

The current split policy is:

1. Split the full WR + SIMBAD-negative variant by stable `source_id` hash.
2. Stratify by target class so approximately 20% of WR and 20% of negatives enter holdout.
3. Keep all non-holdout WR in train.
4. Sample non-holdout negatives to `10x` WR for train using color-quantile stratification.
5. Store remaining non-holdout negatives as `threshold_calibration`.

This gives:

- `train`: all train WR plus representative reduced negatives;
- `threshold_calibration`: negatives only, used to stress-test false positives and threshold behavior;
- `holdout`: WR and negatives never used for fitting or calibration.

Larger negative-ratio experiments are available but not default:

```powershell
wr-detector reduce-negatives --config configs/models.yaml --negative-ratio 20 --negative-ratio 50 --negative-ratio all_train_negatives
```

### Training

```powershell
wr-detector train-models --config configs/models.yaml --run-id run_YYYYMMDD_science_v1
```

Resume an interrupted run:

```powershell
wr-detector train-models --config configs/models.yaml --run-id run_YYYYMMDD_science_v1 --resume-run
```

Current default grid from `configs/models.yaml`:

- 8 dataset variants;
- 2 feature sets: `colors_parallax`, `colors_parallax_error`;
- 3 models: Random Forest, HistGradientBoosting, XGBoost;
- 2 samplers: SMOTE, SMOTE-ENN;
- 1 negative ratio: `10x`;
- total: 96 configurations.

Feature sets:

- `colors_parallax`: `BP_RP`, `G_BP`, `G_RP`, `J_H`, `J_K`, `H_K`, `W1_W2`, `parallax`.
- `colors_parallax_error`: the same features plus `parallax_error` and `parallax_over_error`.

Model search uses `BayesSearchCV` optimized by average precision. SMOTE-style samplers live inside the imbalanced-learn pipeline, so resampling occurs only inside cross-validation folds. Overfit warnings are stored as multicomponent diagnostics: the legacy F2 gap is retained, and new runs also report train-CV average-precision gap, CV-holdout average-precision drop, holdout/CV average-precision ratio, component warning flags and an `overfit_risk_score`.

### Run IDs And Artifacts

Every training run has a `run_id`. Run-specific artifacts are written to:

```text
reports/modeling/runs/{run_id}/
```

The run CSV is updated after every completed configuration:

```text
reports/modeling/runs/{run_id}/model_training_results.csv
```

The canonical experiment history is:

```text
data/databases/training_history.duckdb
```

The compatibility/latest export is:

```text
reports/tables/model_training_results.csv
```

Heavy `.joblib` model files and figures remain on disk and are referenced from DuckDB. Prediction CSVs, feature-importance CSVs and JSON sidecars can be compacted into DuckDB and cleaned after sync:

```powershell
wr-detector clean-model-artifacts --config configs/models.yaml --remove-db-backed-sidecars
wr-detector clean-model-artifacts --config configs/models.yaml --remove-db-backed-sidecars --apply
```

### Interactive Model Explorer

Install the optional visualization extra and launch the local Streamlit explorer:

```powershell
pip install -e ".[viz]"
wr-detector explore-models --config configs/models.yaml
```

The explorer reads `data/databases/training_history.duckdb` in read-only mode. Use it to switch runs, filter dataset variants, compare models by ranking metrics, inspect saved curves and review feature importance without editing notebook cells.

## Model Selection Policy

The default model-selection view prioritizes:

1. WR recovered in top candidate budgets, especially top 50, top 100 and top 500.
2. Holdout average precision.
3. Recall at fixed false-positive rates: 0.1%, 0.5% and 1%.
4. Precision-recall curve shape.
5. Train/CV/holdout stability, overfit risk flags and overfitting gaps.
6. Feature importance and physical plausibility.

Thresholded precision, recall and F-score are reported, but a single thresholded F-score is not the only decision criterion. Threshold precision floors are warnings about an operating point, not automatic rejection of a useful ranking model.

Second-layer validation with autoencoders, Deep SVDD or other one-class models is treated as an experimental re-ranking layer over top first-stage candidates. It should start with enriched tabular features already available in the prediction pool, including magnitude/flux errors, `W3/W4`, proper motions and color-locus diagnostics, and should be evaluated against simpler one-class baselines before becoming operational.

Train the current non-deep second-layer validators:

```powershell
wr-detector train-second-layer --config configs/second_layer.yaml
wr-detector train-second-layer --config configs/second_layer.yaml --variant strict_poe_3 --feature-set current_colors --method gaussian_mixture --subtype WN --run-id run_YYYYMMDD_second_layer
```

This fits subtype-aware one-class validators for WN/WC and reports holdout positive retention, negative pass rates and calibration-negative pass rates. The layer is for compatibility scoring and candidate re-ranking, not a hard rejection gate.

See `docs/modeling_decisions.md` for the current modelling rationale.

## Prediction Pool

Build or audit the Gaia-scale prediction pool:

```powershell
wr-detector build-prediction-pool --config configs/prediction_pool.yaml --dry-run
wr-detector build-prediction-pool --config configs/prediction_pool.yaml
wr-detector audit-prediction-pool --db data/databases/prediction_pool.duckdb
```

The pool is tiled on sky coordinates, excludes known reference/SIMBAD sources by `source_id`, and writes parquet tiles plus `data/databases/prediction_pool.duckdb`. It is not used for training. It should be scored only after selecting and auditing a training run.

Prediction-pool tile consumption is centralized through DuckDB views:

- `prediction_pool_sources`: all source rows from completed parquet tiles.
- `prediction_pool_effective_tiles`: completed parquet tiles that should be iterated during scoring.
- `prediction_pool_tile_coverage`: registered tile coverage, including completed effective tiles, subdivided parent tiles and incomplete tiles.

If a large tile fails and is subdivided, the parent tile is marked `skipped` with `subdivided_into_subtiles`; the completed subtiles appear in `prediction_pool_effective_tiles` and their rows appear in `prediction_pool_sources`. Scoring code should iterate `prediction_pool_effective_tiles`, not raw `prediction_pool_tiles`.

## Notebooks

Notebooks are audit and visualization artifacts. They should not contain source-of-truth pipeline logic.

- `notebooks/00_Available_photometry_analysis.ipynb`: audits photometric completeness and variant-design tradeoffs.
- `notebooks/01_reference_catalog_audit.ipynb`: audits reference catalogue provenance, table counts, crossmatch logs, export sizes and feature coverage.
- `notebooks/02_color_eda_linear_cuts.ipynb`: audits robust color-locus regression assumptions, residuals, outlier flags and configured intra-mission planes.
- `notebooks/03_negative_reduction_audit.ipynb`: audits reduced modelling datasets, train/calibration/holdout dimensions and negative-reduction representativeness.
- `notebooks/04_model_training_results.ipynb`: audits a selected `TRAINING_RUN_ID` from DuckDB; use `wr-detector explore-models` for routine interactive comparison.
- `notebooks/04_overfitting_and_tree_diagnostics.ipynb`: audits overfitting, tree/boosting complexity, feature importance and saved validation artifacts for a selected `TRAINING_RUN_ID`.
- `notebooks/05_prediction_pool_audit.ipynb`: audits prediction-pool coverage, failed tiles, known-source exclusions and distribution comparison before model scoring.

## Useful Outputs

- Raw reference snapshots: `data/raw/reference/`
- Reference DB: `data/databases/wr_reference.duckdb`
- SIMBAD negative DB: `data/databases/simbad_negative.duckdb`
- Prediction-pool DB: `data/databases/prediction_pool.duckdb`
- Processed WR variants: `data/processed/reference/`
- Processed SIMBAD-negative variants: `data/processed/simbad_negative/`
- Reduced modelling datasets: `data/processed/modeling/`
- Training history: `data/databases/training_history.duckdb`
- Run artifacts: `reports/modeling/runs/{run_id}/`

## Development Checks

```powershell
pytest
python -m compileall src\wr_detector
```
