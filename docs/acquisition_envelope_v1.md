# Acquisition envelope v1

## Purpose

The acquisition layer is intentionally broader than any operational model
variant. It is the physical Gaia snapshot that lets the project recalculate
color-locus, photometric-quality and astrometric policies locally without
repeating the archive query.

It is not an all-Gaia download. It covers only the seven intra-mission colors
currently supported by the project:

- Gaia: `G_BP`, `G_RP`, `BP_RP`;
- 2MASS: `J_H`, `J_K`, `H_K`;
- WISE: `W1_W2`.

No Gaia–2MASS, Gaia–WISE or 2MASS–WISE color defines the acquisition boundary.
W3 and W4 are retained as optional features when available, but they are not
required and do not define the envelope.

## Reference population and margins

`configs/prediction_pool_exact_union.yaml` derives the bounds from all 347
finite rows in the `relaxed_photometry` WR color-locus export. It deliberately
includes rows marked as locus outliers. Every observed minimum and maximum is
expanded on both sides by the larger of:

- 0.15 mag;
- 5% of that color's observed WR span.

The dry-run coverage gate currently reports 347/347 finite WR reference rows
inside the envelope. A changed reference export, margin, color list or source
artifact changes the canonical envelope SHA-256 and requires a new immutable
`pool_build_id`.

Observed v1 acquisition bounds:

| Color | Minimum | Maximum |
|---|---:|---:|
| `G_BP` | -4.999606 | 0.261615 |
| `G_RP` | -0.182364 | 2.385318 |
| `BP_RP` | -0.371638 | 6.862254 |
| `J_H` | -0.146000 | 2.848000 |
| `J_K` | -0.134600 | 4.410601 |
| `H_K` | -0.082000 | 2.175000 |
| `W1_W2` | -2.148700 | 2.156700 |

The bounds contain some extreme WR colors and are consequently broader than
the earlier audit envelope. They must be measured with the dedicated smoke
build before the previous 23.88 GiB storage extrapolation is reused.

## Gaia smoke result

The three explicit smoke regions were acquired and validated on 2026-07-24:

| Region | Acquired | Exact compatible union |
|---|---:|---:|
| Dense inner plane, `l≈30` | 1,498 | 973 |
| Outer plane, `l≈150` | 2,862 | 1,285 |
| High latitude, `l≈180, b≈30` | 4,739 | 1,610 |
| **Total** | **9,099** | **3,868** |

With the final audit-column selection, the acquisition Parquet files occupy
3,971,509 bytes, or approximately 436.5 bytes per acquired row. Twelve known
sources were retained with exclusion
labels; four of them pass the exact compatible union, leaving 3,864 rows in the
operational eligible view.

Repeating the identical command completed zero tiles and skipped all three
validated artifacts. Counts remained 9,099 rows and 9,099 distinct
`source_id` values, confirming idempotent resume behavior for the smoke build.

The same regions contained 5,605 acquisitions under the earlier envelope and
A/B server-side cut. The wider v1 policy therefore acquired 62.3% more rows in
this small sample. The later 18-region validation supersedes this preliminary
factor: 65,210 versus 44,469 rows (+46.64%), implying approximately 154 million
rows. At the final measured 436.5 bytes per row, the working storage estimate
is 62.6 GiB for Parquet; 80 GiB including margin is the planning target.

The selected build deliberately uses the current all-bands path. Only 249/347 finite
relaxed WR controls have both Gaia 2MASS and AllWISE best-neighbour paths,
whereas reference construction permits VizieR fallback. Making WISE optional
would raise the 18-region acquisition from 65,210 to 187,450 rows and implies
about 443 million rows / up to 108.8 GiB. See
`docs/prediction_pool_prebuild_verification_20260724.md`. The limitation is
recorded and the existing training/reference lineage is not misrepresented as
complete Gaia best-neighbour coverage.

## Deferred policies

The Gaia ADQL requires the magnitudes needed to calculate the seven colors but
does not apply:

- an exact or aggregate color locus;
- a 2MASS/WISE quality cut;
- a parallax cut;
- a parallax-over-error cut.

Those decisions are evaluated locally for all twelve stable strict/relaxed
astrometric variants. Each acquisition row stores compact masks for exact
locus, photometric quality, astrometry and final compatibility.

Known WR and known SIMBAD non-WR sources remain in the physical acquisition
layer with an exclusion label. The operational `prediction_pool_sources` view
removes them, so they cannot be scored as discoveries while remaining
available for lineage and regression checks.

## Safe execution

First inspect the contract without contacting Gaia:

```powershell
wr-detector build-prediction-pool-exact-union `
  --config configs/prediction_pool_exact_union.yaml `
  --dry-run
```

The already completed smoke acquisition can be reproduced with:

```powershell
wr-detector build-prediction-pool-exact-union `
  --config configs/prediction_pool_exact_union_smoke.yaml `
  --max-tiles 3 `
  --row-limit 10000
```

Repeat the same command to verify that all three validated tiles are skipped.

The production configuration has `full_build_enabled: true` after the final
query-complete smoke reconciled row counts, checksums, resume behavior and
measured bytes per row. Start or resume the full build with:

```powershell
wr-detector build-prediction-pool-exact-union `
  --config configs/prediction_pool_exact_union.yaml `
  --confirm-full-build
```

The legacy pool and its configuration are separate and remain read-only.
