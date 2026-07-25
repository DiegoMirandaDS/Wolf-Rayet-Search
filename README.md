# Wolf-Rayet Detector

Wolf-Rayet Detector is a reproducible Python project for ranking Galactic
Wolf-Rayet (WR) candidates from Gaia DR3, 2MASS and WISE data.

The project connects four pieces that are often treated separately: catalogue
construction, a physically constrained colour-locus, rare-object model
selection and a Gaia-scale prediction pool. The output is a ranked list for
astronomical follow-up, not an automatic claim that a source is a new WR star.

## Current status

The reference, negative-sample, colour-locus and modelling pipelines are
implemented. The first Gaia-scale prediction pool was also built, but an
independent audit found that its local `aggregated_mean_fit` filter is not a
superset of the exact model-variant loci:

- 44,469 Gaia sources were evaluated in 18 stratified sky regions;
- the aggregate rejected 1,960 of 26,369 sources accepted by the compatible
  exact union: 7.43%;
- regional discrepancies reached 18.90%;
- two known WR regression controls were affected;
- seven variants showed misses in Gaia data, and constructive colour
  counterexamples refuted a superset guarantee for all eight variants.

The existing 58,037,788-row pool is therefore registered as a read-only legacy
build. It remains useful for auditing and engineering checks, but it is not
approved for definitive candidate scoring. The next pool will be built in
parallel from the broad acquisition envelope and will persist the exact
compatible-variant mask for every source.

No full-sky exact-union build has been launched yet.

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
`SMOTE` and `SMOTE-ENN`. Adding an unresampled `none` baseline is part of the
next modelling revision.

### Phase 5 — Validation and model inspection

The optional second layer evaluates whether top first-stage candidates remain
compatible with WN or WC reference distributions. Its Gaussian-mixture,
one-class SVM, Isolation Forest and robust-covariance models are re-ranking
tools, not hard rejection gates.

```powershell
wr-detector train-second-layer --config configs/second_layer.yaml
```

The Streamlit explorer reads experiment history from DuckDB and provides
run-level comparison, case review, subtype recovery, contaminant analysis and
validation-layer diagnostics.

```powershell
python -m pip install -e ".[viz]"
wr-detector explore-models --config configs/models.yaml
```

### Phase 6 — Gaia prediction pool

The acquisition stage queries Gaia in sky tiles using a broad colour envelope.
The corrected operational policy is:

```text
broad acquisition envelope
→ evaluate every compatible exact variant
→ retain a source if at least one variant accepts it
→ persist the compatible-variant bitmask
→ score each model only on sources carrying its variant bit
```

The bit contract, lineage requirements, resume behaviour and storage decision
are specified in
[`docs/prediction_pool_exact_union_design.md`](docs/prediction_pool_exact_union_design.md).

The old configuration is intentionally blocked for mutation because it still
uses `aggregated_mean_fit`. These commands are safe:

```powershell
# Inspect the legacy query plan without launching Gaia jobs
wr-detector build-prediction-pool --config configs/prediction_pool.yaml --dry-run

# Audit the existing pool
wr-detector audit-prediction-pool --db data/databases/prediction_pool.duckdb

# Reproduce the 18-region exact-union audit from its persistent acquisitions
wr-detector audit-prediction-pool-exact-union `
  --config configs/prediction_pool_locus_audit.yaml `
  --skip-download
```

Do not delete the legacy DuckDB or its 675 Parquet tiles. Do not run the
non-dry-run legacy builder. A separate full-build configuration will be
published only after the exact-union storage path and manifest contract are
integrated and pass tile-level resume tests.

## Repository layout

```text
configs/                    Versioned pipeline and model configuration
docs/                       Scientific decisions, audits and pool design
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
legacy/                     Historical material kept for context
```

DuckDB is the canonical store for experiment history and pool metadata.
Parquet is used for large source tables. Notebooks read those artefacts for
inspection; they do not redefine filtering or training behaviour.

## Installation

Python 3.11 or newer is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,modeling,viz]"
```

Alternatively:

```powershell
mamba env create -f environment.yml
mamba activate wolf-rayet-detector
python -m pip install -e ".[dev,modeling,viz]"
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

Generated artefacts stay outside Git:

| Artefact | Local path |
|---|---|
| WR reference database | `data/databases/wr_reference.duckdb` |
| SIMBAD negative database | `data/databases/simbad_negative.duckdb` |
| Training history | `data/databases/training_history.duckdb` |
| Legacy prediction pool | `data/databases/prediction_pool.duckdb` |
| Reduced modelling datasets | `data/processed/modeling/` |
| Model runs | `reports/modeling/runs/{run_id}/` |
| Exact-union audit | `reports/analysis/prediction_pool_locus_audit/` |

Run IDs, dataset hashes, configuration hashes and source-level predictions are
used to keep comparisons tied to the data that produced them.

## Audit notebooks

The numbered notebooks document the scientific checks in pipeline order:

- `00`: photometric availability and dataset-family trade-offs;
- `01`: reference-catalogue provenance and feature coverage;
- `02`: colour-locus fits, residuals and outliers;
- `03`: negative reduction and split composition;
- `04`: model ranking and overfitting diagnostics;
- `05`: legacy prediction-pool coverage;
- `06`: PCA/UMAP exploratory structure;
- `07`: bounded aggregate-versus-exact pilot;
- `08`: expanded 18-region exact-union audit and rebuild decision.

## Development checks

```powershell
python -m pytest
python -m compileall src\wr_detector
```

## Immediate roadmap

The next work should be completed in this order:

1. integrate exact-union evaluation and immutable acquisition Parquet into the
   full prediction-pool builder;
2. validate a small resumable build, checksums, manifests and model-bit
   filtering before launching full-sky queries;
3. add an unresampled `none` baseline and top-K/candidates-per-WR metrics;
4. finish the OOF threshold-selection audit;
5. select the operational first-layer and validation-layer pairing;
6. build the new pool in parallel, then begin definitive scoring.

The legacy pool remains available throughout this transition and is never
overwritten.
