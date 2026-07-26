# Chart map

Evidence snapshot: `reports\public\project_summary_evidence.json`

| Figure | Question | Chart family | Main source |
|---|---|---|---|
| `pipeline_overview.png` | How does a source move through the project? | Process flow | Code/configuration contract |
| `catalog_and_locus_retention.png` | How much of each class remains by stage? | Horizontal bars | GWRC/SIMBAD DuckDB + relaxed Parquet |
| `negative_sample_composition.png` | Which contaminant groups form the controlled negative sample? | Ranked bars | `simbad_negative_sources` |
| `color_locus_relaxed.png` | Where do WR and negatives sit relative to the robust locus? | Density + scatter + model band | relaxed color-locus Parquet |
| `sky_distribution.png` | Where are the labelled samples on the sky? | Mollweide density + points | reference/negative DuckDB |
| `model_performance.png` | What trade-offs appear across the 144-model grid? | Scatter + grouped bars | training history `run_v3_main` |
| `precision_recall_roc.png` | How do comparable models behave under class imbalance? | PR/ROC lines | holdout predictions in training history |
| `feature_importance.png` | Which features does the leading broad model use? | Horizontal bars | stored model importance |
| `prediction_pool_audit_and_status.png` | Why was the legacy pool rejected and what is the new build status? | Stacked bars | 18-region audit + exact-union build registry |

Palette policy: one blue root for context, orange for WR/focal results, pink for exclusions or discrepancies, and neutral grey scaffolding.
