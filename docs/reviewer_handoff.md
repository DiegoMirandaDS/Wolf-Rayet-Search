# Reviewer handoff

This repository is source-first. The code, configurations, tests, public
figures, technical report and compact audit notebooks belong in Git. Large
DuckDB, Parquet and model binaries do not: storing them as ordinary Git blobs
would permanently inflate every clone and make later artifact replacement
awkward.

## Release bundle

The binary companion is published as
`wolf_rayet_search_review_bundle.zip` in the project GitHub Release. Download
it from the Release associated with this delivery and extract it into the
repository root while preserving its internal paths. The full prediction pool
is still being built and is not part of this handoff.

The bundle is a binary companion to the source repository. It includes:

- `training_history.duckdb`, restricted to the complete `run_v3_main` and
  `run_v3_second_layer` records used in this delivery;
- `wr_reference.duckdb` and an app-facing `simbad_negative.duckdb` projection
  containing the identity, classification, astrometry and photometry fields
  used by Model Explorer;
- three representative first-stage models from `run_v3_main`;
- compact CSV exports of the first- and second-layer result tables.

The training-history database contains metrics and source-level predictions for
all 144 first-stage configurations, so Model Explorer can compare the complete
run even though only three first-stage `.joblib` files are shipped. The selected
models cover the broad-shortlist, AP-leading and accepted/stable profiles.

The second-layer audit remains visible under Validation Layers through its
synchronized DuckDB rows. Its 64 `.joblib` files and four reduced Parquet
datasets are intentionally excluded: the experiment did not improve the
operational first-stage ranking enough to justify distributing those executable
artifacts. Candidate Stack requires those omitted files and is not part of the
review-bundle workflow.

## Integrity

The Release notes provide the SHA-256 of the complete ZIP. The archive also
contains `REVIEW_BUNDLE_MANIFEST.json`, with a SHA-256 and byte count for every
payload file.

## Open the project after extraction

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --require-hashes -r requirements-review.txt
python -m pip install -e . --no-deps --no-build-isolation
wr-detector explore-models --config configs/models.yaml
```

The app can browse run comparison, case review, subtype diagnostics and
validation-layer results from the bundled databases. Re-training and mass
prediction are not required for review. The launcher listens only on
`127.0.0.1`, keeps CORS and XSRF protection enabled and disables Streamlit usage
telemetry.

The three bundled `.joblib` files are Python serialized objects. Load only the
copies downloaded from the official project Release and verify the Release
SHA-256 before extraction.
