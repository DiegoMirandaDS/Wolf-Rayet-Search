# Exact-union pre-build verification — 2026-07-24

## Decision

The exact-union storage, mask, checksum and resume implementation passes its
18-region validation. The project selected the current all-bands acquisition
path for this build. The modelling reference and Gaia acquisition do not use
identical infrared-photometry provenance, so that limitation remains explicit.

The reference pipeline allows VizieR fallback for missing 2MASS/WISE rows.
The prediction query currently requires Gaia DR3 `best_neighbour` rows. This
does not change the color envelope, but it changes which sources have the
features required to enter it.

## Physical build validation

The new acquisition-first builder was run over the same 18 spatially
stratified boxes used by the expanded locus audit:

- 18/18 tiles completed;
- 65,210 acquisition rows;
- 65,210 distinct `source_id` values;
- 65,210 distinct `source_hash_v1` values;
- 28,858 rows pass at least one exact compatible variant;
- 11 eligible known sources are excluded from the operational view;
- 28,847 operational eligible rows;
- two sources have multiple 2MASS PSC alternatives; both alternatives are
  persisted as JSON and one is chosen by the versioned deterministic rule;
- 17,199,908 compressed Parquet bytes, or 263.76 bytes/source.

An identical rerun completed zero tiles and skipped all 18. The reusable build
audit passed:

```powershell
wr-detector audit-prediction-pool-exact-union-build `
  --config configs/prediction_pool_exact_union_validation.yaml
```

It verifies Parquet and manifest SHA-256 values, physical and registered row
counts, uniqueness, bit population counts, union flags, component-mask
intersection, exclusions and effective tile counts.

Relative to the earlier 44,469-row audit acquisition, removing the server-side
A/B quality cut and expanding the WR bounds increases the sample by 46.64%.
A direct stratified extrapolation gives approximately 154 million rows. The
final query-complete smoke retains additional quality, cross-match,
astrometric, variability and H-alpha fields and measures 436.5 compressed
bytes/source, revising the Parquet estimate to approximately 62.6 GiB.

## WR acquisition-path coverage

All 347 finite `relaxed_photometry` WR rows are geometrically inside the
versioned envelope. Archive-path verification gives a different result:

| Required archive path | WR available |
|---|---:|
| Gaia core photometry | 347/347 |
| Gaia 2MASS best-neighbour path | 343/347 |
| Gaia AllWISE best-neighbour path | 249/347 |
| Both infrared paths and all envelope bounds | 249/347 |

Every one of the 249 WR with both archive paths is inside the envelope. The
missing 98 are therefore not color-boundary failures. Their local modelling
rows rely on fallback photometry not reproduced by the current prediction
query.

The 2MASS clean-to-PSC join can occasionally produce multiple alternatives.
Replacing it with `best_neighbour.original_ext_source_id` was tested and
rejected because it also reduced WR coverage to 249/347. The accepted
implementation keeps the clean-to-PSC join, chooses a canonical counterpart by
quality/error/identifier and stores every alternative.

## Cost of broader acquisition lanes

Counts inside the same 18 boxes:

| Physical acquisition requirement | Rows | Relative to current |
|---|---:|---:|
| Gaia + 2MASS + WISE bounds | 65,210 | 1.00× |
| Gaia + 2MASS bounds, WISE optional | 187,450 | 2.87× |
| Gaia-color bounds only | 522,361 | 8.01× |

Using the current stratified scaling and bytes per row only for planning:

- current all-bands path: about 154 million rows / 62.6 GiB with the final
  retained columns;
- WISE-optional Gaia+2MASS path: about 443 million rows / up to 108.8 GiB;
- Gaia-only path: about 1.23 billion rows / up to 303 GiB.

Null-heavy optional columns may compress better, so the last two disk values
are conservative approximations rather than capacity guarantees.

## Recorded scientific choice

Two defensible paths were evaluated:

1. Keep the approximately 154-million-row all-bands pool, but rebuild
   training/reference variants using the same Gaia best-neighbour provenance.
   This is smaller but excludes 98/347 current relaxed WR controls.
2. Make WISE optional in the physical Gaia+2MASS acquisition layer
   (approximately 443 million rows), keep the current WISE-compatible logical
   view, and train a no-WISE first-stage model for fallback rows. External WISE
   enrichment can then be applied only to ranked candidates. Four WR without
   the Gaia 2MASS path remain documented exceptions or require a much larger
   Gaia-only lane.

The first path was selected for the current build: it covers the seven
intra-mission colors used by the existing model families and remains feasible
to acquire and audit quickly. The 98 WR controls without both Gaia archive
paths remain a documented provenance limitation. A future no-WISE family
would require a distinct acquisition lane and a new immutable pool identifier;
the current build must not be silently reinterpreted as that broader pool.
