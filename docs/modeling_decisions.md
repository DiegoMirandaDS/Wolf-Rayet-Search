# Modeling Decision Log

## Candidate Detection Is A Ranking Problem

The original project reported classification metrics on a much less imbalanced test set than the current reproducible pipeline. In the current data, the holdout can contain roughly hundreds of SIMBAD non-WR objects per known WR. Under that prevalence, a classifier can have high ROC-AUC and still show low threshold precision if the false-positive rate is not extremely small.

For this reason, Phase 3 modelling treats WR detection primarily as a candidate-ranking problem:

- optimize model ranking with average precision;
- inspect precision-recall curves rather than accuracy;
- report precision@K and recall@K for candidate-list budgets;
- report recall at fixed false-positive-rate levels;
- choose operating thresholds after ranking, based on follow-up budget and acceptable false positives.

Accuracy is retained only as a diagnostic. It is not a selection objective, because a model can achieve very high accuracy by predicting nearly everything as non-WR.

## Training Design

- The default training grid uses `strict` and `relaxed` crossed with `photometry`, `parallax_soft`, `poe_2`, and `poe_3`; only `poe_5` is excluded from the default model sweep.
- The active feature sets are `colors_parallax` and `colors_parallax_error`.
- The production comparison includes Random Forest, HistGradientBoosting and XGBoost when optional XGBoost is installed.
- The default sweep has 96 configurations: 8 variants, 2 feature sets, 3 models, 2 samplers and 1 negative ratio.
- `BayesSearchCV` optimizes average precision.
- SMOTE-style samplers remain inside the pipeline so resampling happens only inside CV folds.
- Reduced training negatives are sampled representatively in color space.
- The default reduction target is `10x` negatives per WR. Larger ratios such as `20x`, `50x`, and `all_train_negatives` remain available through `--negative-ratio` for follow-up stability checks.
- Negatives not selected for training are retained as `threshold_calibration` rows so thresholds are calibrated against a more realistic negative prevalence.
- `threshold_calibration` negatives are also audited as a stress-test pool. They are not used inside `BayesSearchCV`, so they remain useful for checking false-positive behavior after model selection. They contain negatives only, so they cannot measure recall by themselves; they measure how many plausible non-WR objects receive high WR scores.
- The holdout split is stable by `source_id` hash and stratified by target class. This keeps a reproducible 20% WR holdout and 20% negative holdout before negative reduction, avoiding random loss of scarce WR positives while keeping metrics comparable across negative-ratio experiments.
- Rows rejected by the robust linear-regression color-locus cut are excluded from model fitting through `color_locus_keep`. The color-locus parquet exports still keep all source rows and diagnostic columns.
- Overfitting is now audited with separate stability diagnostics instead of a single thresholded F-score status. Training rows keep the legacy `cv_train_gap_f2`, and also store `train_cv_gap_average_precision`, `cv_holdout_drop_average_precision`, `holdout_to_cv_average_precision_ratio`, per-signal warning flags and `overfit_risk_score`.

If holdout splitting, color-locus filtering, required features or negative-reduction policy changes, regenerate the reduced modelling datasets before retraining:

```powershell
wr-detector reduce-negatives --config configs/models.yaml
```

The reason for class-stratified holdout is scarcity of positives. A purely global hash split is reproducible, but it does not guarantee a stable WR count in holdout for every variant. Stratifying the same stable hash by target class preserves reproducibility while making the test set scientifically easier to compare.

## Model Selection

The main comparison should prioritize:

- holdout average precision;
- holdout precision@K and recall@K;
- holdout recall at fixed FPR;
- precision-recall curve shape;
- overfitting and instability diagnostics across train, CV, holdout and ranking metrics;
- feature importance and physical plausibility.

Threshold precision floors are useful for flagging unusable operating points, but they should not be the only way to judge whether a model is useful for candidate discovery.

The `selection_status` column remains a compact primary status for compatibility. Overfit risk must be read from `overfit_warning_flag` and the component flags, because a threshold or precision floor failure can coexist with overfit risk.

## Current Operating Policy

The Streamlit Model Explorer is the routine interactive surface for ranking trained models as candidate generators. It is launched with:

```powershell
wr-detector explore-models --config configs/models.yaml
```

The default operational ranking is:

1. maximize known WR recovered in the top 100 holdout candidates;
2. break ties with holdout average precision;
3. prefer stronger top-50 recovery when manual inspection budget is small;
4. inspect recall at FPR 0.5% as a low-false-positive control;
5. reject or demote only when train/CV/holdout diagnostics show clear instability.

This is intentionally different from maximizing a single thresholded F-score. A fixed threshold is still reported through the confusion matrix, precision, recall and F-score, but it is not the only product of the model. For WR discovery, a high-quality ranked list can be useful even when the default precision floor is not met at the automatically selected threshold.

The Model Explorer presents the repeatable decision views interactively:

- best model for broad candidate recovery, using top-100 and top-500 recovery;
- best model for short-list inspection, using top-50 recovery and average precision;
- top-3 overfitting diagnostics, using train/CV/holdout gaps and saved validation artifacts.

Model notebooks remain audit and narrative review artifacts. They can inspect a specific `TRAINING_RUN_ID`, but routine run/model switching should use the Model Explorer so results are generated from DuckDB filters rather than manually edited notebook markdown.

## Second-Layer Validation Research

A second, more expensive validation layer is scientifically reasonable, but it should be evaluated as a compatibility or re-ranking layer over the top candidates from the first-stage ranking models, not as a hard rejection gate.

Implemented second-layer baseline:

- `wr-detector train-second-layer --config configs/second_layer.yaml`
- Trains subtype-aware one-class validators for WN and WC.
- Supported baseline methods are Gaussian mixture, one-class SVM, Isolation Forest and robust covariance.
- The validators fit only known WR positives from the requested subtype and evaluate against holdout positives, holdout negatives and `threshold_calibration` negatives.
- The output is `reports/modeling/second_layer/runs/{run_id}/second_layer_validation_results.csv`, plus a latest compatibility export at `reports/tables/second_layer_validation_results.csv`.
- The layer reports positive retention, negative pass rates, holdout average precision and ROC-AUC. These metrics are for compatibility scoring and re-ranking, not automatic rejection.

Recommended feature priorities:

1. Current stable modelling features: `BP_RP`, `G_BP`, `G_RP`, `J_H`, `J_K`, `H_K`, `W1_W2`, `parallax`, `parallax_error`, `parallax_over_error`.
2. Prediction-pool features already available but not yet used by training: Gaia/2MASS/WISE magnitudes, fluxes, flux errors, magnitude errors, `W3`, `W4`, `W3_error`, `W4_error`, `pmra`, `pmdec`, source quality strings, and color-locus diagnostics.
3. Derived features to add before deep validation: additional intra- and near/mid-IR colors involving `W3/W4`, signal-to-noise ratios from flux/error columns, magnitude-error features, quality flags encoded as categorical indicators, proper-motion amplitude, and color-locus residual/plane-count diagnostics.
4. High-value external enrichment for a later phase: Gaia XP spectra or ESP-ELS class probabilities, image cutouts from WISE/unWISE or H-alpha surveys, extinction/dust context, Galactic coordinates and local sky-density context.

Autoencoders and Deep SVDD are viable experiments only if the feature space is enriched beyond the current small color set and the models are regularized aggressively. With the current 8-10 tabular features, they are likely to duplicate simpler one-class baselines or overfit the small positive sample. The preferred experimental order is:

1. subtype-aware non-deep baselines: robust covariance, Gaussian mixture, one-class SVM and Isolation Forest;
2. small tabular autoencoder with strong bottleneck, dropout/weight decay and early stopping;
3. Deep SVDD by broad subtype only after the autoencoder and non-deep baselines have a validated holdout/calibration advantage;
4. image or spectral deep models only after Gaia XP, WISE image, H-alpha or comparable high-dimensional inputs are added.

Known WR subtype labels should be preserved from GWRC `Spectral Type` into modelling exports. WN and WC can be audited separately; WO and WN/WC are too scarce for separate model training and should be treated as stress-test categories.

## H-Alpha And NIR/MIR Feature Availability

Current local datasets do not persist H-alpha photometry or Gaia ESP-ELS H-alpha measurements. They do preserve `source_id`, coordinates and enough Gaia/2MASS/WISE context to cross-match or enrich later.

Gaia DR3 provides H-alpha information through `gaiadr3.astrophysical_parameters`, including `ew_espels_halpha` and `classlabel_espels`. A 2026-06-11 Gaia@AIP spot audit found:

- GWRC Gaia-identified WR: 430/442 have rows in `astrophysical_parameters`; 339/442 have non-null `ew_espels_halpha`.
- `relaxed_photometry` reduced positives: 269/331 have non-null `ew_espels_halpha`.
- `strict_photometry` reduced positives: 234/292 have non-null `ew_espels_halpha`.
- `relaxed_poe_3` reduced positives: 201/208 have non-null `ew_espels_halpha`.
- `strict_poe_3` reduced positives: 170/173 have non-null `ew_espels_halpha`.
- A small 100-row negative sample from each of `strict/relaxed photometry` and `strict/relaxed poe_3` showed roughly 91-93% non-null `ew_espels_halpha`; this is an estimate only and should be replaced by a full enrichment audit before making H-alpha mandatory.

Conclusion: Gaia ESP-ELS H-alpha is promising and probably would not heavily reduce the `poe_3` WR-positive datasets. It should be added as an optional enrichment/audit feature first, then promoted to a required feature only after measuring full negative and prediction-pool coverage. IPHAS/VPHAS+ H-alpha photometry is scientifically useful but has footprint and optical-extinction limitations, so it should remain optional enrichment rather than a default required field for Gaia-scale scoring.

## Training History Storage

The project keeps `reports/tables/model_training_results.csv` as a compatibility/latest export, but the Model Explorer and model notebooks should read the canonical experiment history from `data/databases/training_history.duckdb` by `run_id`.

The database stores:

- one row per training run in `training_runs`;
- all model metrics and selected hyperparameters in `model_results`;
- model and figure paths in `model_artifacts`;
- feature-importance rows in `feature_importance`;
- prediction rows in `model_predictions`;
- JSON sidecar metadata in `model_metadata`.

Paths stored in the database are relative to the project root. Large `.joblib` and `.png` artifacts remain as files. Prediction CSVs, feature-importance CSVs and model JSON sidecars are compacted into DuckDB and can be deleted with `clean-model-artifacts --remove-db-backed-sidecars --apply`. The cleanup command runs as a dry-run unless `--apply` is passed.

New training runs create a run id before the first model is fitted. Artifacts are written under `reports/modeling/runs/{run_id}/`, including an incremental `model_training_results.csv` updated after every completed configuration. Interrupted terminals can be resumed without duplicating completed configurations:

```powershell
wr-detector reduce-negatives --config configs/models.yaml
wr-detector train-models --config configs/models.yaml --run-id run_YYYYMMDD_science_v1
wr-detector train-models --config configs/models.yaml --run-id run_YYYYMMDD_science_v1 --resume-run
```

The existing baseline artifacts are registered in DuckDB as `run_20260608T194455984398Z_d1dee0800bd4` without moving the files.

## Next Improvements

- Preserve GWRC `Spectral Type` in modelling exports and audit missed WR objects from the recommended top-100 list by subtype and photometric quality.
- Add threshold-calibration false-positive stress metrics to the training row, so overfit risk can include score behavior on non-selected negatives.
- Add hard-negative mining by retraining with negatives that the best model ranks highly.
- Add calibrated probabilities only after selecting the ranking model.
- Add holdout permutation importance before treating any feature as physically meaningful.

## Prediction-Pool Scoring Handoff

After model selection, scoring should read prediction-pool work units from `prediction_pool_effective_tiles`, not directly from raw `prediction_pool_tiles`. The raw tile table can include skipped parent tiles that were replaced by completed subtiles. `prediction_pool_tile_coverage` is the audit view for distinguishing `effective_tile`, `subdivided_parent`, and `incomplete`; `prediction_pool_sources` is the all-row view over completed parquet files.
