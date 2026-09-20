# Wolf-Rayet Search

Wolf-Rayet Search is a reproducible Python project for ranking Galactic
Wolf-Rayet (WR) candidates from Gaia DR3, 2MASS and WISE data.

The project connects four pieces that are often treated separately: catalogue
construction, a physically constrained colour-locus, rare-object model
selection and a Gaia-scale prediction pool. The output is a ranked list for
astronomical follow-up, not an automatic claim that a source is a new WR star.

This research project began in 2024 and is being developed toward a scientific
manuscript. The ranked candidate set has been shared with collaborating
astronomers for archival-spectrum searches and spectral assessment. Until that
review is complete, every listed source remains a candidate rather than a
confirmed Wolf-Rayet star.

## Technical handoff

The concise project narrative, equations, modelling decisions, ranked-model
table, limitations and current pool status are collected in the
[technical report](deliverables/technical_report/InformeTecnico.pdf). Practical
review instructions are provided in
[`InstruccionesDeUso.pdf`](deliverables/technical_report/InstruccionesDeUso.pdf).
Its figures and evidence snapshot are reproducible with:

```powershell
python -m wr_detector.reporting.project_summary --figures-only
```

| Colour-locus decision | Model evaluation |
|---|---|
| ![Relaxed colour locus](reports/public/figures/color_locus_relaxed.png) | ![Model performance](reports/public/figures/model_performance.png) |

The binary data required to browse the completed experiments is deliberately
kept out of Git history. Reviewers can download
`wolf_rayet_search_review_bundle.zip` from the GitHub Release associated with
this delivery and extract it in the repository root.

The bundle contains the three DuckDB review databases and three representative
first-stage models. Second-layer metrics remain visible in Model Explorer, but
the second-layer `.joblib` files and their reduced training datasets are omitted:
their audit did not improve the operational first-stage ranking enough to justify
shipping executable model artifacts.

For a review-only installation, use an isolated Python 3.11 environment and the
hashed dependency lock. It includes Streamlit and preserves the
scikit-learn/XGBoost versions used to serialize the bundled models:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --require-hashes -r requirements-review.txt
python -m pip install -e . --no-deps --no-build-isolation
wr-detector explore-models --config configs/models.yaml
```

The launcher binds Streamlit to `127.0.0.1`, keeps CORS and XSRF protection
enabled, and disables anonymous usage telemetry. Only load `.joblib` files from
the official project Release.

## Current status

The reference, negative-sample, colour-locus and modelling pipelines are
implemented. The first Gaia-scale prediction pool was also built, but an
independent audit found that its local `aggregated_mean_fit` filter does not
retain every source accepted by at least one model-compatible dataset variant:

- 44,469 Gaia sources were evaluated in 18 stratified sky regions;
- the aggregate rejected 1,960 of 26,369 sources accepted by at least one
  compatible variant: 7.43%;
- regional discrepancies reached 18.90%;
- two known WR regression controls were affected;
- seven variants showed misses in Gaia data, and constructive colour
  counterexamples refuted a superset guarantee for all eight variants.

The existing 58,037,788-row pool is therefore registered as a read-only legacy
build. It remains useful for auditing and engineering checks, but it is not
approved for definitive candidate scoring. The replacement prediction-pool
build is running in parallel under the internal build id `exact_union`. It
persists the broad acquisition layer and the compatible-variant bitmask for
every source. No definitive prediction scores or candidate list are reported
until that build finishes and passes its final audit.

## Scientific workflow

### Phase 1 — Known WR reference catalogue

The reference pipeline starts from the Galactic WR catalogue, keeps records
with explicit Gaia DR3 aliases and enriches them with:

- Gaia DR3 astrometry and `G/BP/RP` photometry;
- 2MASS `J/H/Ks`;
- WISE `W1/W2`, while retaining `W3/W4` when available;
- VizieR fallback matches when the Gaia crossmatch tables do not provide the
  required 2MASS or WISE row.

The result is stored in a DuckDB database with catalogue snapshots, match
provenance and processed Parquet variants.

```powershell
wr-detector build-reference --config configs/reference.yaml
wr-detector audit-reference --db data/databases/wr_reference.duckdb
wr-detector export-reference-datasets --config configs/filters.yaml
```

### Phase 2 — Controlled non-WR sample

The negative sample is built from configured SIMBAD object types. Known WR
`source_id` values are excluded before export, and the same Gaia/2MASS/WISE
feature construction is used for positives and negatives.

```powershell
wr-detector build-simbad-negative --config configs/simbad_negative.yaml
wr-detector export-simbad-negative-datasets --config configs/filters.yaml
```

This is a controlled comparison sample, not a complete representation of every
non-WR population in the Galaxy. That distinction matters when interpreting
precision and false-positive estimates.

### Phase 3 — WR colour-locus

Robust colour-colour regressions are fitted on WR reference variants. Only
intra-survey colours define the locus:

- Gaia: `G_BP`, `G_RP`, `BP_RP`;
- 2MASS: `J_H`, `J_K`, `H_K`.

`W1_W2` is retained as a model feature, but WISE does not define a regression
plane because `W3/W4` are not required. Cross-survey colours are deliberately
excluded from the locus to reduce sensitivity to epoch, calibration and
crossmatch systematics.

The export keeps every row and adds locus diagnostics. Training filters through
`color_locus_keep`; the source artefacts remain auditable.

```powershell
wr-detector export-color-locus --config configs/filters.yaml
```

### Phase 4 — Dataset variants and model training

Each dataset variant combines a photometric family and an astrometric subset.

| Component | Definition |
|---|---|
| `strict` | Required 2MASS and WISE bands have quality A |
| `relaxed` | Required 2MASS and WISE bands have quality A or B |
| `photometry` | No parallax restriction |
| `parallax_soft` | `parallax > 0` |
| `poe_2` | `parallax > 0` and `parallax_over_error >= 2` |
| `poe_3` | `parallax > 0` and `parallax_over_error >= 3` |

The default sweep uses the eight `strict`/`relaxed` combinations with
`photometry`, `parallax_soft`, `poe_2` and `poe_3`.

The split is stable by `source_id` hash and stratified by class:

1. approximately 20% of WR and 20% of negatives enter holdout;
2. all remaining WR enter training;
3. training negatives are reduced to the configured ratio, currently 10:1;
4. unused non-holdout negatives form `threshold_calibration`.

The holdout is never used for fitting. `threshold_calibration` contains
negatives only and is used as a false-positive stress test.

```powershell
wr-detector reduce-negatives --config configs/models.yaml
wr-detector train-models --config configs/models.yaml --run-id run_YYYYMMDD_science_v1
```

Before a full sweep, run the compact comparison that isolates the effect of
synthetic resampling on the two broad photometric families:

```powershell
wr-detector train-models `
  --config configs/models.yaml `
  --run-id run_YYYYMMDD_exact_union_pilot_v1 `
  --variant strict_photometry `
  --variant relaxed_photometry `
  --feature-set colors_parallax_error `
  --model xgboost `
  --model hist_gradient_boosting `
  --sampler none `
  --sampler smote
```

This is eight configurations. The default unfiltered command is the complete
144-configuration sweep.

Photometric-provenance sensitivity can be isolated without changing negatives,
features or the split. `gaia_native_ir` keeps only WR whose 2MASS and WISE
matches both come from the Gaia cross-match tables:

```powershell
wr-detector train-models `
  --config configs/models.yaml `
  --run-id run_provenance_native `
  --variant relaxed_photometry `
  --feature-set colors_parallax_error `
  --model xgboost `
  --sampler none `
  --train-positive-cohort gaia_native_ir `
  --evaluation-positive-cohort gaia_native_ir
```

The two cohort flags affect positive rows only. Every result records the cohort
contract, source-id hash and pre/post counts. Use the same
`--evaluation-positive-cohort` in compared runs so their holdout populations
are identical.

Interrupted runs are resumable:

```powershell
wr-detector train-models `
  --config configs/models.yaml `
  --run-id run_YYYYMMDD_science_v1 `
  --resume-run
```

Model search optimizes average precision. Selection prioritizes candidate
recovery within practical follow-up budgets, low-FPR behaviour and stability
across train, cross-validation and holdout. Ordinary accuracy is not an
appropriate objective for a pool containing tens of millions of sources.

The current grid compares Random Forest, HistGradientBoosting and XGBoost with
`none`, `SMOTE` and `SMOTE-ENN`. `none` preserves the observed training rows
and compensates class imbalance inside the estimator: balanced subsample
weights for Random Forest, class weights for HistGradientBoosting and
`scale_pos_weight = N_negative/N_WR` for XGBoost. Sampled configurations use
neutral estimator weights so imbalance is not corrected twice.

### Phase 5 — Model inspection and optional validation audit

The Streamlit explorer reads experiment history from DuckDB and provides
run-level comparison, case review, subtype recovery, contaminant analysis and
validation-layer diagnostics. Its compact active-model context stays
synchronized across pages without exposing long result ids in the sidebar.
The active-model picker first narrows to a dataset variant and then offers its
short, searchable configuration list. On Compare, table checks select up to
eight models for the charts; a separate `Set active` action can explicitly
promote one checked row. Checking a row never changes the active model.
Case review can switch between top-K review states and conventional
operating-threshold TP/FP/FN/TN states. Its photometric plot supports
color-magnitude and color-color exploration with intra-mission colors, plus a
Galactic polar sky view. The top-down Galactic-plane view includes only
positive, quality-filtered parallaxes and labels its `1/parallax` distances
and spiral arms as approximations.
An optional WN/WC one-class validation experiment was also evaluated. It did
not improve fixed-budget recovery enough to become an operational layer, so the
first-stage score remains the candidate ranking. Its aggregate retention and
negative-pass metrics remain available under Validation Layers; its model
binaries and reduced datasets are not part of the reviewer bundle.

```powershell
wr-detector explore-models --config configs/models.yaml
```

### Phase 6 — Gaia prediction pool

The acquisition stage queries Gaia in sky tiles using a broad colour envelope.
The corrected operational policy is:

```text
broad acquisition envelope
→ persist every acquired source before eligibility filtering
→ evaluate every compatible dataset variant locally
→ expose the combined eligible pool and model subsets as logical views
→ score each model only on sources carrying its variant bit
```

The production configuration versions the bit contract, locus lineage,
acquisition envelope, resume behaviour and storage paths. Scoring checks these
contracts before applying any selected model.
The three-region Gaia smoke contains 9,099 unique acquisition rows, measures
436.5 compressed bytes per row with the retained audit columns, and resumes by
skipping all three validated tiles.

The subsequent 18-region build validation passes all physical, mask, checksum
and resume checks. The selected all-bands path is estimated at about 154
million rows and 62.6 GiB after retaining quality, cross-match, astrometric,
variability and H-alpha audit fields. Only 249/347 finite relaxed WR controls
have both Gaia 2MASS and AllWISE best-neighbour paths, while the reference
dataset permits VizieR fallback; this is an explicit scope limitation, not an
envelope loss.

The old configuration is intentionally blocked for mutation because it still
uses `aggregated_mean_fit`. The new acquisition-first configuration is separate:

```powershell
# Validate the new envelope, WR coverage, masks and ADQL locally
wr-detector build-prediction-pool-exact-union `
  --config configs/prediction_pool_exact_union.yaml `
  --dry-run

# Bounded, disposable smoke build; this is not the full scientific pool
wr-detector build-prediction-pool-exact-union `
  --config configs/prediction_pool_exact_union_smoke.yaml `
  --max-tiles 3 `
  --row-limit 10000

# Start or resume the production build (the same command is always used)
wr-detector build-prediction-pool-exact-union `
  --config configs/prediction_pool_exact_union.yaml `
  --confirm-full-build

# Run after every build session; definitive scoring requires a passing audit
wr-detector audit-prediction-pool-exact-union-build `
  --config configs/prediction_pool_exact_union.yaml

# Inspect the legacy query plan without launching Gaia jobs
wr-detector build-prediction-pool --config configs/prediction_pool.yaml --dry-run

# Audit the existing pool
wr-detector audit-prediction-pool --db data/databases/prediction_pool.duckdb

# Reproduce the 18-region compatibility audit from its persistent acquisitions
wr-detector audit-prediction-pool-exact-union `
  --config configs/prediction_pool_locus_audit.yaml `
  --skip-download
```

After the production build audit passes and a training run has been reviewed,
list its synchronized model results and select the operational `result_id`
values explicitly:

```powershell
wr-detector list-prediction-pool-models `
  --config configs/prediction_pool_scoring.yaml `
  --model-run-id run_v3_main

# Validate contracts and estimate the selected work without writing scores
wr-detector score-prediction-pool `
  --config configs/prediction_pool_scoring.yaml `
  --scoring-run-id score_run_v3_main_v1 `
  --model-run-id run_v3_main `
  --result-id RESULT_ID `
  --dry-run

# First score and audit one tile
wr-detector score-prediction-pool `
  --config configs/prediction_pool_scoring.yaml `
  --scoring-run-id score_run_v3_main_v1 `
  --model-run-id run_v3_main `
  --result-id RESULT_ID `
  --max-tiles 1

wr-detector audit-prediction-pool-scoring `
  --config configs/prediction_pool_scoring.yaml `
  --scoring-run-id score_run_v3_main_v1

# Resume the same scoring run over every completed production tile
wr-detector score-prediction-pool `
  --config configs/prediction_pool_scoring.yaml `
  --scoring-run-id score_run_v3_main_v1 `
  --model-run-id run_v3_main `
  --result-id RESULT_ID
```

Repeat `--result-id` to apply more than one reviewed model. The scorer never
selects a model automatically. It verifies the model hash, exact locus,
variant bit, feature list and pool schema before inference; reads only
completed acquisition tiles; and writes atomic, resumable score Parquet per
model and tile. Sources compatible with the variant but missing a model feature
are counted in the manifest and are not imputed.

Do not delete the legacy DuckDB or its 675 Parquet tiles. Do not run the
non-dry-run legacy builder. The production prediction-pool configuration was opened
only after the Gaia smoke validated the final query columns, measured storage,
checksums, manifests and resume behavior.

## Repository layout

```text
configs/                    Versioned pipeline and model configuration
notebooks/                  EDA and audit notebooks; never source-of-truth logic
src/wr_detector/
  catalogs/                 Gaia, GWRC, SIMBAD and VizieR access
  pipelines/                Reference, negative and prediction-pool pipelines
  modeling/                 Splits, reduction, training, evaluation and history
  analysis/                 Reproducible audit calculations
  apps/                     Streamlit Model Explorer
tests/                      Unit and integration tests
data/                       Local generated data; excluded from Git
reports/                    Local run artefacts; excluded from Git
```

DuckDB is the canonical store for experiment history and pool metadata.
Parquet is used for large source tables. Notebooks read those artefacts for
inspection; they do not redefine filtering or training behaviour.

## Full development installation

The review-only installation is described under Technical handoff. The commands
below install every development, training, reporting and notebook dependency and
are intended for contributors who need to reproduce the complete pipeline.
Python 3.11 or newer is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,modeling,viz,report,notebooks]"
```

Alternatively:

```powershell
mamba env create -f environment.yml
mamba activate wolf-rayet-detector
python -m pip install -e ".[dev,modeling,viz,report,notebooks]"
```

Gaia archive credentials are optional for public queries. Authenticated
long-running jobs use a local credentials file:

```powershell
Copy-Item .env.example .env
```

Edit `.env` so `GAIA_CREDENTIALS_FILE` points to your local Gaia credentials
file. Neither `.env` nor the credential file is versioned.

## Reproducing the implemented pipeline

Run the stages in this order:

```powershell
wr-detector build-reference --config configs/reference.yaml
wr-detector audit-reference --db data/databases/wr_reference.duckdb
wr-detector export-reference-datasets --config configs/filters.yaml

wr-detector build-simbad-negative --config configs/simbad_negative.yaml
wr-detector export-simbad-negative-datasets --config configs/filters.yaml

wr-detector export-color-locus --config configs/filters.yaml
wr-detector reduce-negatives --config configs/models.yaml
wr-detector train-models --config configs/models.yaml --run-id run_YYYYMMDD_science_v1
wr-detector train-second-layer --config configs/second_layer.yaml
```

Large catalogue queries and model grids can take considerable time. Every
long-running stage writes run- or tile-level state so work can be inspected or
resumed.

## Main artefacts

Large generated artefacts stay outside Git. The small, stable handoff material
under `reports/public/` is the deliberate exception:

| Artefact | Local path |
|---|---|
| WR reference database | `data/databases/wr_reference.duckdb` |
| SIMBAD negative database | `data/databases/simbad_negative.duckdb` |
| Training history | `data/databases/training_history.duckdb` |
| Legacy prediction pool | `data/databases/prediction_pool.duckdb` |
| Prediction pool (build id `exact_union`) | `data/databases/prediction_pool_exact_union_v1.duckdb` |
| Prediction scoring registry | `data/databases/prediction_pool_scoring.duckdb` |
| Reduced modelling datasets | `data/processed/modeling/` |
| Model runs | `reports/modeling/runs/{run_id}/` |
| Prediction scores | `data/processed/prediction_pool_scores/{scoring_run_id}/` |
| Variant-compatibility audit | `reports/analysis/prediction_pool_locus_audit/` |
| Technical handoff PDF | `deliverables/technical_report/` |
| Public figures and evidence snapshot | `reports/public/` |
| Optional reviewer bundle | `dist/wolf_rayet_search_review_bundle.zip` |

Run IDs, dataset hashes, configuration hashes and source-level predictions are
used to keep comparisons tied to the data that produced them.

## Audit notebooks

Three compact notebooks document the scientific checks in pipeline order:

- `01_reference_and_color_locus.ipynb`: catalogue provenance, negative
  composition, photometric retention and the signed-log colour-locus decision;
- `02_model_selection_and_validation.ipynb`: split policy, ranking metrics,
  leading model trade-offs, curves, feature importance and second-layer scope;
- `03_prediction_pool_status.ipynb`: legacy-filter failure, replacement-pool
  architecture, live build snapshot and the path to definitive scoring.

They are executed audit artefacts, not sources of pipeline logic. Rebuild all
three with:

```powershell
python -m wr_detector.reporting.review_notebooks --execute
```

## Development checks

```powershell
python -m pytest
python -m compileall src\wr_detector
```

## Immediate roadmap

The next work should be completed in this order:

1. finish the running prediction-pool acquisition and pass its coverage, checksum,
   manifest and mask audit;
2. select one or more synchronized `result_id` values from `run_v3_main` using
   an explicit follow-up budget and the stability diagnostics;
3. validate scoring with `--dry-run`, then score and audit one tile;
4. resume full scoring and review the resulting short lists by subtype and
   contaminant class;
5. audit Gaia BP/RP, RVS and H-alpha coverage before adding spectral evidence
   as optional enrichment.

The legacy pool remains available throughout this transition and is never
overwritten.
