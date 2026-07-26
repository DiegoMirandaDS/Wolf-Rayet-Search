# Configuration map

There are 15 tracked YAML files. They have four roles:

| Role | Files |
|---|---|
| Shared inputs | `paths.yaml`, `catalogs.yaml`, `filters.yaml` |
| Reproducible pipelines | `reference.yaml`, `simbad_negative.yaml`, `models.yaml`, `second_layer.yaml`, `prediction_pool_scoring.yaml` |
| Current prediction pool | `prediction_pool_exact_union.yaml` |
| Small prediction-pool overlays | `prediction_pool_exact_union_smoke.yaml`, `prediction_pool_exact_union_validation.yaml` |
| Audits and lineage | `prediction_pool_locus_pilot.yaml`, `prediction_pool_locus_audit.yaml`, `prediction_pool_build_registry.yaml` |
| Immutable legacy pool | `prediction_pool.yaml` |

The exact-union production config is the source of truth. Smoke and validation
configs use `extends` and override only build identifiers, output paths, limits
and tile lists. Mapping values merge recursively; lists are replaced.

`prediction_pool_scoring.yaml` is the handoff between the exact-union pool and
explicitly selected model results from the canonical training-history DuckDB.
It controls only scoring storage, streaming batch size and the complete-pool
safety gate; model selection stays on the CLI through reviewed `result_id`
values.

`prediction_pool.yaml` is retained only to describe and audit the old
`aggregated_mean_fit` build. Code rejects attempts to mutate it.

Generated Parquet, CSV, DuckDB, reports and model artifacts do not belong in
this directory or in Git.
