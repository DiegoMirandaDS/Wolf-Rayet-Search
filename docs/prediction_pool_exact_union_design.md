# Versioned exact-union prediction pool

## Decision

The operational eligibility policy is:

```text
broad acquisition envelope
→ evaluate every compatible exact variant
→ keep a source when at least one exact variant accepts it
→ persist the accepting variants
→ score each model only on sources whose variant bit is active
```

The `aggregated_mean_fit` regression is not part of the new eligibility path.
The existing 58,037,788-row pool remains a read-only legacy build registered in
`configs/prediction_pool_build_registry.yaml`; it must not be overwritten or
used for definitive scoring.

## Source-level contract

`src/wr_detector/pipelines/exact_variant_union.py` is the reusable source of
truth. For each source it produces:

- `passes_any_exact_variant`: logical OR of all compatible exact variants.
- `compatible_variant_count`: population count of the active variant bits.
- `compatible_variant_mask`: unsigned 64-bit mask with one stable bit per
  variant.
- `source_hash_v1`: deterministic hash over the canonical source payload used
  to detect conflicting resume rows; its exact field order and null encoding
  must be frozen before the full build.

A variant is active only if all three components pass: its exact color locus,
its photometric-quality condition, and its astrometric condition. The colors
required by the exact locus are recorded in the variant contract. Missing
required values fail that variant rather than being imputed during pool
construction.

### Stable bit schema

The initial registry reserves bits by photometric family so adding a variant
does not move existing bits:

| Bit | Variant |
|---:|---|
| 0 | `strict_photometry` |
| 1 | `strict_parallax_soft` |
| 2 | `strict_poe_1` |
| 3 | `strict_poe_2` |
| 4 | `strict_poe_3` |
| 5 | `strict_poe_5` |
| 6–7 | reserved for strict variants |
| 8 | `relaxed_photometry` |
| 9 | `relaxed_parallax_soft` |
| 10 | `relaxed_poe_1` |
| 11 | `relaxed_poe_2` |
| 12 | `relaxed_poe_3` |
| 13 | `relaxed_poe_5` |
| 14–15 | reserved for relaxed variants |

The manifest stores the ordered mapping, its canonical SHA-256, and a version
of the form `compatible_variant_mask_v1_<hash-prefix>`. Canonical hashing sorts
by bit, so evaluation order cannot change the schema. Adding or changing a
variant changes the schema hash/version, but existing assignments stay fixed.
Bits must never be silently reused; an incompatible registry change requires a
new major schema version.

Every variant-manifest entry records:

- variant name and bit;
- exact `locus_run_id` and locus artifact SHA-256;
- required colors and transform;
- photometric-quality rule;
- astrometric rule;
- exact plane coefficients and thresholds.

The historical color-locus exports predate explicit run identifiers. The audit
therefore assigns a deterministic transitional identifier
`legacy_exact_<variant>_<sha-prefix>` and keeps the full file SHA-256. A future
locus export must provide its native `locus_run_id` rather than deriving one.

## Build and tile lineage

Each future pool build must have an immutable build manifest containing:

- `pool_build_id`, creation/completion timestamps and status;
- code commit, package/environment fingerprint and configuration versions;
- acquisition-envelope definition and SHA-256;
- bitmask schema version, mapping and SHA-256;
- exact `locus_run_id` and artifact hash for every variant;
- canonical ADQL text and SHA-256 per tile;
- Gaia asynchronous job ID and archive release/table identifiers;
- pre-locus acquired rows;
- accepted rows per variant and by the exact union;
- written rows and known-source exclusions;
- artifact paths, byte sizes and SHA-256 checksums;
- parent/subtile relationships and retry history.

Counts are checked at two levels. A tile manifest reconciles its physical
Parquet; the build manifest is the sum of effective terminal tiles. Known
sources are excluded after exact compatibility is calculated so both the
scientific acceptance count and the written count remain auditable.

## Pre-filter persistence

The preferred layout is one compressed Zstandard Parquet acquisition artifact
per terminal tile. It contains the acquired source columns plus the exact mask,
count and union flag. Eligible pools and model-specific subsets are DuckDB views
or scanner predicates, not duplicate source rows:

```sql
SELECT * FROM acquisition_sources
WHERE passes_any_exact_variant;

SELECT * FROM acquisition_sources
WHERE compatible_variant_mask & (1::UBIGINT << :model_variant_bit) <> 0;
```

This layout retains the ability to recompute eligibility without querying Gaia
again. When a locus changes, a new derived mask artifact may temporarily sit
beside the immutable acquisition artifact; it replaces the prior operational
mask only after validation.

The two-layer alternative (`acquisition` plus a physically duplicated
`eligible` Parquet) is simpler for some consumers but costs more disk and creates
an extra reconciliation surface. Keeping only eligible rows is smallest but
cannot recover a source after a locus change. The expanded audit measures all
three layouts and writes `storage_estimate.csv`; these are planning estimates,
not capacity guarantees.

The production path is implemented by
`wr_detector.pipelines.prediction_pool_exact_union`. Its v1 acquisition
configuration expands all finite `relaxed_photometry` WR color limits with a
versioned margin and removes server-side locus, quality and astrometric cuts.
It stores separate `exact_locus_variant_mask`, `photometry_variant_mask`,
`astrometry_variant_mask` and `compatible_variant_mask` columns. This lets
future policies be audited from one physical acquisition layer.

Because the expanded v1 envelope includes locus outliers and no longer requires
A/B quality in ADQL, the earlier 23.88 GiB extrapolation is not a capacity
guarantee for this broader snapshot. The full-build gate stays closed until a
bounded Gaia smoke build measures its source density and compressed bytes per
row. See `docs/acquisition_envelope_v1.md`.

Staging deletion is permitted only after:

1. the Parquet is readable and its schema matches the contract;
2. row counts and unique `source_id` counts match the tile manifest;
3. the file checksum is persisted;
4. variant, union, exclusion and written counts reconcile;
5. the tile record is committed atomically as completed;
6. a resume simulation confirms the completed tile is skipped.

## Scoring contract

A model is paired with one declared dataset variant. Before inference, the
scoring reader resolves that variant's bit from the build manifest and applies
the bit predicate. It must not infer compatibility from the model filename or
re-evaluate a different locus. The scorer records the pool build ID, mask schema
hash, variant, bit, locus run/hash and model result ID in its output manifest.

The operational implementation is
`wr_detector.pipelines.prediction_pool_scoring` and its configuration is
`configs/prediction_pool_scoring.yaml`. A scoring application requires an
immutable `scoring_run_id`, a synchronized first-layer `model_run_id` and one
or more explicitly reviewed `result_id` values. Automatic best-model selection
is deliberately not supported.

For every selected model and terminal tile, the scorer:

1. verifies the model artifact SHA-256 and exact feature list;
2. matches the model's dataset variant, locus run and reference hash to the
   pool contract;
3. reads only completed acquisition Parquet whose checksum still matches;
4. pushes the stable variant-bit predicate into DuckDB before inference;
5. excludes known reference/control sources;
6. streams bounded Arrow batches rather than loading a whole tile;
7. scores only rows with every required model feature present;
8. writes a compressed Parquet atomically plus a checksummed manifest;
9. registers compatible, feature-missing, scored and threshold-positive counts
   in `prediction_pool_scoring.duckdb`.

Completed work units are skipped only when the input, output, manifest, model
and schema contracts all still match. A corrupt completed record is never
silently overwritten. Reuse the same scoring run to resume valid work; use a
new run ID when the selected models or pool contract changes.

Full scoring is blocked until the completed tile set equals the tile set
declared by the production pool configuration. `--max-tiles` is the bounded
validation path used after training: dry-run first, score one tile, audit it,
then resume the same application without the bound.

## Mandatory validation

Automated tests cover the following invariants:

- the union contains every source accepted by any compatible exact variant;
- a source with a false union flag is never eligible;
- mask bits, per-variant booleans and the count agree;
- evaluation order does not affect the result;
- adding a variant changes the schema hash/version without moving old bits;
- astrometric rules are applied exactly;
- model scoring receives only rows with its bit active;
- physical pre/post counts match tile manifests;
- resume merges neither duplicate sources nor duplicate tiles and rejects
  conflicting completed artifacts.

The expanded spatial audit is evidence about discrepancy rate and admitted
source density. Unknown Gaia sources remain unlabeled, so its aggregate-only
acceptances must not be described as false positives.
