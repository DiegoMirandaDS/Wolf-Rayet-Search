# Code Audit - 2026-06-24

## Scope

This audit reviewed reproducible source code under `src/wr_detector`, configuration files, public Markdown documentation, audit notebooks under `notebooks/`, and the Streamlit Model Explorer. The historical `legacy/` directory and generated `data/` and `reports/` artifacts were intentionally excluded.

The test suite passed before fixes (`56 passed`) and after targeted fixes. No evidence was found for a major scientific-design failure in the current Phase 1-3 pipeline.

## Corrected Findings

### Benchmark configuration drift

`wr_detector.modeling.benchmark._benchmark_one_model` read `config["threshold"]["min_precision"]`, but the current project config stores threshold-selection policy under `selection`. This made `wr-detector benchmark-models --config configs/models.yaml` incompatible with the current default config.

Resolution: the benchmark now reads `threshold` for backward compatibility and falls back to `selection`. The regression test now uses the modern `selection` block.

### Run-scoped artifact cleanup gap

Current training writes run-specific artifacts under `reports/modeling/runs/{run_id}/`, but `cleanup_unreferenced_model_artifacts` only scanned legacy top-level artifact directories. As a result, run-scoped prediction and feature-importance sidecars could remain on disk even after being compacted into DuckDB.

Resolution: cleanup now scans run-specific `models`, `figures`, and `reports` subdirectories. A regression test verifies detection and deletion of an orphan run-scoped sidecar.

### Model Explorer mojibake risk

Several Model Explorer labels used non-ASCII separators, arrows, and symbols. They are valid Unicode in source, but they rendered as mojibake in some local terminal views and can make diffs harder to inspect.

Resolution: Streamlit labels, captions, model short labels, and layer titles now use ASCII separators. The change is display-only and does not affect query logic.

### Notebook language and stale outputs

Notebooks 03, 04, 04B, 05, and 06 contained Spanish Markdown, plot titles, and display messages. Their executed outputs also preserved stale Spanish text.

Resolution: notebook sources were rewritten in scientific English and outputs were cleared for the edited notebooks. This keeps notebooks reproducible and avoids mixed-language audit artifacts.

### Code documentation coverage

Several reproducible modules had no module-level documentation, even though they encode important catalogue, feature, modelling, history, and prediction-pool responsibilities.

Resolution: concise module docstrings were added to undocumented source modules. The docstrings state purpose and operational responsibility only; implementation details remain in code and tests.

## Scientific And Structural Review

### Data lineage

The first-layer and second-layer lineage policy is coherent: first-layer runs are identified by `run_id`, while validation-layer compatibility is linked to reduced-dataset and config hashes. This avoids an incorrect dependency between one-class validators and a specific first-stage classifier when the validator only learns WR subtype compatibility.

Residual recommendation: add an explicit candidate-stack registry before prediction-pool scoring if a future layer consumes first-stage top-K outputs. That registry should record the upstream `run_id`, `result_id`, model artifact path, validation-layer run, and scoring policy.

### Color-locus policy

The Phase 2 rule is implemented consistently: intra-mission colors define robust color-locus diagnostics, exports retain all rows, and modelling filters with `color_locus_keep`. Cross-mission colors are available as features in later layers but are not used for Phase 2 locus cuts.

Residual recommendation: keep `server_side_locus_filter` disabled unless the ADQL predicate is updated to match the local aggregate-plane rule. The current prediction-pool config already keeps it disabled, which is the scientifically safer choice.

### Negative reduction and thresholds

The split and reduction policy is coherent for rare-object ranking: holdout is assigned before negative reduction, training negatives are reduced reproducibly, and the remaining non-holdout negatives form `threshold_calibration`.

Residual recommendation: persist threshold-calibration stress metrics in each first-layer training row, not only in notebooks or second-layer evaluation. Useful metrics include calibration negatives above the selected threshold, top-K calibration score quantiles, and calibration pass rate at fixed WR recall.

### Model Explorer

The Explorer has the right separation of concerns: Streamlit pages render UI, while SQL and business logic live in `wr_detector.modeling.explorer` and `wr_detector.modeling.cases`. Case review, subtype recovery, recurrent contaminants, and validation layers are scientifically useful for candidate-ranking decisions.

Residual recommendations:

- Add a run-level lineage panel showing config hash, dataset hashes when available, color-locus filtering status, and negative ratio coverage.
- Add threshold-calibration stress views once those metrics are persisted in first-layer history.
- Add a read-only dimensionality page only after notebook 06 demonstrates stable interpretive value across more than one run.
- Keep saved PNG artifacts as paths only; live PR/ROC/confusion charts from synchronized predictions are the better source of truth.

### Notebooks

The notebook sequence is coherent: early notebooks audit input data and color-locus assumptions, modelling notebooks inspect synchronized history, and the prediction-pool notebook audits coverage before scoring.

Residual recommendations:

- Keep notebooks output-clean in Git unless a specific rendered audit snapshot is intentionally versioned.
- Move repeated notebook helpers into tested modules only when they affect decisions outside exploratory review.
- Do not promote PCA/UMAP coordinates to model features without a separate validation study; their current role should remain visual audit and case triage.

## No Major Issues Found

No high-severity bug was found in the core scientific flow: catalogue construction, color-locus filtering, holdout-before-reduction, ranking-oriented model evaluation, and second-layer lineage are internally consistent with the documented project policy.
