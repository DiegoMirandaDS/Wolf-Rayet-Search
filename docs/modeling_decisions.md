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
- The default sweep has 144 configurations: 8 variants, 2 feature sets, 3
  models, 3 samplers and 1 negative ratio.
- `BayesSearchCV` optimizes average precision.
- `none` is the non-synthetic baseline. It uses estimator-native imbalance
  handling: Random Forest `balanced_subsample`, HistGradientBoosting class
  weights and XGBoost `scale_pos_weight = N_negative/N_WR`.
- SMOTE-style samplers remain inside the pipeline so resampling happens only
  inside CV folds. Their estimator weights are neutral; combining SMOTE with
  balanced estimator weights would correct the same imbalance twice.
- Reduced training negatives are sampled representatively in color space.
- The default reduction target is `10x` negatives per WR. Larger ratios such as `20x`, `50x`, and `all_train_negatives` remain available through `--negative-ratio` for follow-up stability checks.
- Negatives not selected for training are retained as `threshold_calibration` rows so thresholds are calibrated against a more realistic negative prevalence.
- `threshold_calibration` negatives are also audited as a stress-test pool. They are not used inside `BayesSearchCV`, so they remain useful for checking false-positive behavior after model selection. They contain negatives only, so they cannot measure recall by themselves; they measure how many plausible non-WR objects receive high WR scores.
- The holdout split is stable by `source_id` hash and stratified by target class. This keeps a reproducible 20% WR holdout and 20% negative holdout before negative reduction, avoiding random loss of scarce WR positives while keeping metrics comparable across negative-ratio experiments.
- Rows rejected by the robust linear-regression color-locus cut are excluded from model fitting through `color_locus_keep`. The color-locus parquet exports still keep all source rows and diagnostic columns.
- Overfitting is now audited with separate stability diagnostics instead of a single thresholded F-score status. Training rows keep the legacy `cv_train_gap_f2`, and also store `train_cv_gap_average_precision`, `cv_holdout_drop_average_precision`, `holdout_to_cv_average_precision_ratio`, per-signal warning flags and `overfit_risk_score`.
- Every new first-layer result records the reduced dataset, reference, negative,
  config and model SHA-256 values; the locus identifier; feature list; split
  policy; Git worktree fingerprint; sampler strategy; effective positive class
  weight; and threshold-selection population. This prevents models trained
  under different data or locus contracts from being treated as interchangeable.
- Top-K reporting includes `candidates_per_wr_at_K`, the direct manual-review
  cost for each recovered holdout WR.

The first rerun after introducing `none` should be an eight-configuration
pilot: strict/relaxed photometry, `colors_parallax_error`, XGBoost and
HistGradientBoosting, each with `none` and SMOTE. The 144-configuration sweep
is justified only after this isolates whether synthetic resampling improves
ranking and low-FPR behavior over native estimator weighting.

If holdout splitting, color-locus filtering, required features or negative-reduction policy changes, regenerate the reduced modelling datasets before retraining:

```powershell
wr-detector reduce-negatives --config configs/models.yaml
```

The reason for class-stratified holdout is scarcity of positives. A purely global hash split is reproducible, but it does not guarantee a stable WR count in holdout for every variant. Stratifying the same stable hash by target class preserves reproducibility while making the test set scientifically easier to compare.

## 2MASS/WISE Provenance Sensitivity (2026-07-25)

Some WR reference rows obtain missing 2MASS or WISE photometry through a
coordinate-based VizieR fallback. Coordinates are not model features and Gaia
parallax is unchanged, but the fallback can still alter the photometric
measurement domain. This was tested as a positive-cohort sensitivity check,
not as a new dataset definition.

The versioned `gaia_native_ir` cohort requires both
`twomass_matches.match_method = gaia_xmatch` and
`wise_matches.match_method = gaia_xmatch`. In `relaxed_photometry` this gives:

- 249 of 347 WR before the color-locus decision;
- 238 of 331 WR after `color_locus_keep`;
- 184 native WR in the reduced training split and 54 in holdout before
  feature-completeness filtering.

Two compact XGBoost/`none` runs used the same reduced-dataset hash, ten
`colors_parallax_error` features, negative rows, split and eight-step Bayesian
search:

- `run_v3_provenance_all`: all eligible WR for training;
- `run_v3_provenance_native`: only `gaia_native_ir` WR for training;
- both evaluated on the same complete native holdout: 53 WR and 6,558
  negatives.

| Metric on common native holdout | All eligible WR | Native-only WR |
|---|---:|---:|
| Average precision | 0.519 | 0.489 |
| ROC-AUC | 0.9812 | 0.9811 |
| WR in top 10 | 8 | 8 |
| WR in top 50 | 30 | 28 |
| WR in top 100 | 39 | 38 |
| Recall at FPR 0.5% | 0.660 | 0.585 |

A paired class-stratified bootstrap gave an average-precision difference
(`native-only - all`) of -0.031, with a 95% interval of approximately
[-0.076, 0.011]. The result is therefore not evidence of a statistically clear
degradation, but it gives no indication that discarding fallback WR improves
the native domain. On the full complete holdout, including the 15 fallback WR,
the all-WR model also had higher AP (0.514 versus 0.480) and recovered more WR
in the top 50 (34 versus 28).

Operational decision: keep all 347 `relaxed_photometry` WR available to the
main sweep, subject to each variant's locus and feature requirements. Preserve
`gaia_native_ir` as an auditable sensitivity cohort rather than making it the
default. This pilot uses one variant, model, sampler and stable split, so it is
a scope check rather than a universal claim about match quality.

## Prediction-Pool Locus Gate Pilot (2026-07-21)

The main modelling pipeline remains conditioned on `color_locus_keep`; the holdout split is not moved before the locus. A separate pilot tested whether the local aggregate filter used after Gaia acquisition was a safe superset of all exact variant loci.

The aggregate is constructed by taking the midpoint of exact slopes and intercepts and the maximum threshold plane by plane. This construction has no geometric guarantee of containing the union of the original bands. Five predeclared Gaia boxes were acquired using only the current global color envelope, then evaluated with the literal aggregate implementation and all eight exact loci.

Results:

- 9,102 sources were acquired inside the envelope.
- 333 pass at least one exact locus but fail the aggregate; 327 remain compatible after variant quality and astrometric conditions.
- The aggregate keeps 5,858 sources; the exact compatible union keeps 6,151 (+293 net, +5.0%).
- The acquisition envelope alone keeps all 9,102 (+55.4% versus the aggregate).
- Two known WR controls within the envelope, WR 122-15 (WN6) and WR 157 (WN5o(+B1II)), pass an exact compatible variant but fail the aggregate.
- The discrepancy is dominated by the `J_H__H_K` and `J_K__H_K` planes.
- The pilot's aggregate result reconciles with the current pool in all five boxes before the later known-source exclusion step.

Decision: the current prediction pool is not approved for definitive scoring of the planned model grid. Do not reconstruct or mass-score it yet. The provisional replacement is the versioned logical union of exact compatible variants, while preserving the locus-conditioned pipeline. Run a wider independent audit before reconstruction; use acquisition-envelope-only as the maximum-sensitivity fallback, recognizing its larger volume.

The 386 historical raw CSVs are not primary evidence: 385 link to completed tiles in the partial-build registry, but the query text/hash and configuration/envelope/locus hashes were not persisted; one file is `test_simple`. The pilot therefore uses new persistent queries with ADQL, Gaia job id and SHA-256 lineage. Reproduce with `wr-detector pilot-prediction-pool-locus --config configs/prediction_pool_locus_pilot.yaml`.

## Expanded Exact-Union Spatial Audit (2026-07-22)

The reconstruction gate was evaluated in 18 persistent Gaia boxes stratified by Galactic longitude/latitude, acquisition density, magnitude, inner/outer disk and approximate extinction. Unknown Gaia sources remain unlabelled, so the reported quantities are discrepancy rates and admitted-source densities, not false-positive rates.

Results:

- 44,469 sources were acquired inside the broad envelope.
- The aggregate keeps 24,570; the compatible exact union keeps 26,369.
- The aggregate rejects 1,960 compatible exact-union sources (7.43% of that union) and admits 161 sources outside the compatible union, for +1,799 net sources (+7.32%) under the exact union.
- Regional loss ranges from 1.96% to 18.90% outside the very small Galactic-centre box; the largest rates occur in dense/high-extinction inner-disk fields. High-density strata lose 12.48%, versus 4.03–4.29% in the low/medium density strata. Inner-disk fields lose 13.94%, versus 2.94% in the outer disk and 2.37% off the plane.
- The discrepancy rises with magnitude: 4.11% for `G<12`, 6.48% for `16<=G<18`, and 10.90% for `G>=18`.
- Seven model variants have observed losses. `relaxed_poe_2` has zero observed losses in the sample, but it is not analytically contained plane by plane. A deterministic Sobol search that respects all envelope bounds and the Gaia/2MASS color identities finds 25 explicit `relaxed_poe_2` color witnesses that pass the exact locus and fail the aggregate; witnesses exist for all eight variants. Therefore no variant is certified complete in the legacy pool.
- The two known-WR regression controls remain affected: WR 122-15 and WR 157.

Decision: create a complete logical replacement as a **new parallel pool**, and preserve the current build as immutable legacy `aggregated_mean_fit`. A prioritized regional execution order is acceptable, but a priority-only reconstruction is not the final scientific dataset. The legacy pool may be used only for audits or exploratory engineering, not definitive model scoring, including for `relaxed_poe_2`.

The legacy configuration is now guarded in code: `configs/prediction_pool.yaml`
is marked `legacy_read_only`, and a non-dry-run `build-prediction-pool` call
raises before opening or modifying its DuckDB. This avoids presenting the old
aggregate constructor as the corrected reconstruction path. The full-sky
exact-union command will be published only with separate output locations,
immutable acquisition storage and validated tile manifests/resume behaviour.

The preferred persistence design is one immutable Zstandard acquisition Parquet per terminal tile containing `compatible_variant_mask`, `compatible_variant_count` and `passes_any_exact_variant`, with eligible and model-specific logical views. Audit-based planning estimates are 14.77 GiB for eligible-only, 38.48 GiB for physical acquisition plus eligible layers, and 23.88 GiB for acquisition-with-mask plus logical views. These are stratified-sample extrapolations, not capacity guarantees. See `docs/prediction_pool_exact_union_design.md` and reproduce with `wr-detector audit-prediction-pool-exact-union --config configs/prediction_pool_locus_audit.yaml`.

## Acquisition-First Envelope v1 (2026-07-24)

The replacement builder now persists every source returned by a versioned
intra-mission acquisition envelope before applying model eligibility. The
envelope uses only `G_BP`, `G_RP`, `BP_RP`, `J_H`, `J_K`, `H_K` and `W1_W2`.
It is derived from all 347 finite `relaxed_photometry` WR reference rows,
including locus outliers, with at least 0.15 mag or 5% of the observed span
added on each side. The deterministic dry-run coverage gate accepts 347/347
finite reference rows.

Exact locus, A/B quality and astrometric rules are not server-side acquisition
cuts. They are local, versioned masks. Known sources remain physically
available with exclusion labels and are removed only by the operational logical
view. This design prioritizes recalculating future locus variants without
another Gaia query.

The wider policy invalidated the previous storage estimate as a guarantee,
especially because low-quality 2MASS/WISE rows are acquired rather than
discarded in ADQL. The production configuration was opened only after the
final-column smoke established bytes per row, checksums and resume behavior.
The legacy 58-million-row build remains immutable.

The 2026-07-24 three-region Gaia smoke acquired 9,099 unique sources. With the
final retained fields it wrote 3,971,509 compressed bytes (436.5 bytes/source).
The exact compatible union contains 3,868 rows; four eligible known sources are
excluded only from the logical operational view. A repeated identical run
skipped all three completed tiles with unchanged counts and checksums.

The subsequent 18-region acquisition-first validation contains 65,210 unique
sources, passes all physical/mask/checksum/resume invariants and estimates
approximately 154 million rows. Applying the final smoke compression rate gives
an estimated 62.6 GiB of Parquet for the current all-bands join.
However, archive-path controls show 347/347 relaxed WR with Gaia core
photometry, 343/347 with the Gaia 2MASS path and only 249/347 with the Gaia
AllWISE path. All 249 with both paths are inside the envelope. The missing 98
are a provenance mismatch caused by reference-side VizieR fallback, not an
envelope failure.

Decision: use the smaller all-bands path for the current exact-union build and
record the Gaia/VizieR provenance mismatch as a scope limitation. The final
query keeps quality, cross-match, astrometric, variability and H-alpha audit
fields and measures 436.5 compressed bytes/source in the smoke, implying about
62.6 GiB at the 154-million-row estimate. The full-build gate is open after
query, checksum and resume validation. A future WISE-optional family requires
a distinct immutable acquisition lane; that alternative measures 187,450 rows
in the 18 boxes versus 65,210 currently and extrapolates to roughly 443 million
rows. See
`docs/prediction_pool_prebuild_verification_20260724.md`.

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
- one compact active-model context synchronized across comparison, detail,
  cases, statistics and candidate-stack views;
- multi-row Compare checks that control only the comparison charts, plus an
  explicit action to promote one checked model to the cross-page active model;
- Case review separates review-budget diagnostics from operating-threshold
  confusion classes. A known WR outside top-K is a missed review-budget case,
  not automatically a threshold false negative. Photometric case views use
  intra-mission colors only. Galactic plane positions use qualified positive
  parallaxes with a disclosed inverse-parallax approximation; sources without
  acceptable distances remain available in the RA/Dec-derived polar sky view;
- a lineage-compatible first-stage/validation-layer audit that keeps long
  result ids in provenance views rather than routine selectors.

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

The 2026-07-25 `run_v3_second_layer` joint audit does not justify a hard
second-layer gate. For the leading `relaxed_photometry` XGBoost/`none` result,
enriched Gaussian mixture retained 45/48 WR and 36/52 negatives inside the
first-stage top 100, then recovered 46 WR at pass-first K=100 versus 48 in the
original ranking. At input top 500 it retained 54/60 WR and 169/440 negatives,
but recovered 58 WR at pass-first K=500 versus 60 originally. Enriched
one-class SVM removed more negatives but lost more WR. Current-color Isolation
Forest preserved the most WR but passed too many negatives to demonstrate a
useful operational gain. Keep the layer audit-only and do not reject
prediction-pool candidates with it until a lineage-registered stack improves
fixed-budget recovery without material subtype loss.

For the external reviewer handoff, retain the synchronized second-layer result
rows in `training_history.duckdb` so Validation Layers remains auditable, but do
not distribute the 64 second-layer `.joblib` files or the four reduced Parquet
datasets. Candidate Stack remains an optional local reproduction path when
those artifacts are present.

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

## Validation-Layer Lineage And Color-Locus Policy

Validation layers (second layer today, any future third layer) are trained independently of first-layer model runs on purpose: a one-class validator learns what a WR subtype looks like, which does not depend on which first-stage classifier was selected. The reproducibility link between layers is therefore **data lineage, not run identity**:

- Every second-layer result row records `dataset_path`, `dataset_sha256`, `models_config_path`, `models_config_sha256`, `holdout_fraction`, `require_color_locus_keep` and `color_locus_excluded_rows`.
- A layer run may be paired with a first-layer run only when both consumed reduced datasets with the same hashes; otherwise splits may be inconsistent (train/holdout leakage across layers).
- When a future layer consumes upstream *outputs* (for example, training or evaluating on the top-K candidates of a specific first-layer model), that layer must additionally record the upstream `run_id`/`result_id` it consumed. The application step that pairs layers into an operational candidate pipeline should be registered explicitly (a candidate-stack record) before scoring the prediction pool.
- Second-layer results auto-sync to the canonical history DB (`second_layer_runs` / `second_layer_results` in `data/databases/training_history.duckdb`); `wr-detector sync-second-layer-history --run-id <id>` backfills CSV-only runs.

Color-locus outliers (rows with `color_locus_keep = false`, both WR and negatives) are excluded from all modelling: the exclusion happens once in `load_modeling_dataset` before the holdout split, so every reduced dataset and every split (train, holdout, threshold_calibration) already excludes them. **Every validation layer must keep this property.** The second layer additionally applies a defensive `apply_color_locus_keep` guard and records how many rows it had to exclude (expected: 0). Future layers must follow the same rule: train and evaluate only on color-locus-kept rows.

Threshold selection in the first layer is not a thresholded-accuracy afterthought: the operating threshold is selected against train-positive out-of-fold scores combined with the full `threshold_calibration` negative pool (`make_threshold_selection_scores`), so the large non-sampled negative population already disciplines the operating point. Probability calibration (Platt/isotonic) is intentionally not applied while the problem is treated as ranking.

## Next Improvements

- Preserve GWRC `Spectral Type` in modelling exports and audit missed WR objects from the recommended top-100 list by subtype and photometric quality.
- Add score-distribution summaries for threshold-calibration negatives to the
  training row, so overfit risk can include behavior on non-selected negatives.
- Add hard-negative mining by retraining with negatives that the best model ranks highly.
- Add calibrated probabilities only after selecting the ranking model.
- Add holdout permutation importance before treating any feature as physically meaningful.

## Prediction-Pool Scoring Handoff

After model selection, scoring should read prediction-pool work units from `prediction_pool_effective_tiles`, not directly from raw `prediction_pool_tiles`. The raw tile table can include skipped parent tiles that were replaced by completed subtiles. `prediction_pool_tile_coverage` is the audit view for distinguishing `effective_tile`, `subdivided_parent`, and `incomplete`; `prediction_pool_sources` is the all-row view over completed parquet files.

The exact-union scorer is now implemented through
`wr-detector score-prediction-pool --config
configs/prediction_pool_scoring.yaml`. Operational model selection remains a
human-reviewed decision: the command requires explicit `result_id` values from
`wr-detector list-prediction-pool-models`. Each model is applied only where its
stable variant bit is active. Hash, locus, schema, feature-completeness and
resume checks are mandatory; no missing model feature is imputed during
Gaia-scale inference.
## Case Review visual diagnostics

- Case Review keeps review-budget states and operating-threshold confusion
  states separate. A known WR outside top-K is not automatically a
  threshold-based false negative.
- Interactive density or deterministic sampling is a rendering optimization
  for background/true-negative sources only. It does not change evaluation,
  filters, predictions or the complete source frame used by PNG export.
- The all-sky diagnostic uses the standard Mollweide equations, with Galactic
  longitude zero centered and positive longitude increasing to the left.
- Top-down Galactocentric positions use `distance_kpc = 1 / parallax_mas` only
  for positive parallaxes that meet the selected parallax/error floor and
  distance cap. The disk and spiral arms are schematic context, not inferred
  source-arm assignments.

## Relative model comparison across holdouts

- Cross-dataset model selection uses relative metrics whenever positive
  holdout sizes differ. `Recall @100` is `holdout_wr_at_100 / wr_holdout` and
  `Recall @50` is defined analogously. Raw `WR@100` remains visible as
  descriptive context but is not a primary ranking criterion.
- The Model Explorer ranking score is
  `0.35 * Recall@100 + 0.25 * AP + 0.15 * Precision@100 +
  0.10 * Recall@50 + 0.10 * threshold precision +
  0.05 * threshold recall`, minus `0.05 * sqrt(holdout FPR)`.
  Threshold precision, recall, FP and FPR are holdout evaluations at the
  already selected threshold; the threshold-calibration negative pass rate is
  a separate stress-test metric and is not substituted for holdout FPR.
- Historical runs with missing fields remain readable. Derivable ratios are
  reconstructed from stored counts and denominators. The composite score
  normalizes by available weight and applies a small coverage discount; it
  does not fabricate unavailable metrics.
- Overview selects a diversity-aware Top 10 seeded by Recall@100, AP,
  Precision@100, calibrated-threshold balance, lowest FPR and global balance,
  then fills from the composite ranking while limiting near-identical
  model/sampler/family/feature-set signatures.
