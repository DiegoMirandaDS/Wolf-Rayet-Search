# Model Explorer Style Reference

## Product Posture

The Model Explorer is a scientific model-audit and candidate-review tool for Wolf-Rayet detection. Design for repeated technical use: dense information, explicit filters, reproducible provenance, and traceable model decisions. Avoid landing-page composition, oversized hero elements, decorative cards, gradients, illustrations, and generic dashboard chrome.

The visual tone should feel like a dark, compact analysis console: restrained, legible, data-forward, and practical. Every page should answer an operational question: which model is best, why it is risky, what sources it recovers, what contaminants it creates, and whether validation layers support the candidate-ranking workflow.

## App Structure

- Entry point: `src/wr_detector/apps/model_explorer.py`.
- Shared UI helpers: `src/wr_detector/apps/explorer_ui/ui.py`.
- Chart builders: `src/wr_detector/apps/explorer_ui/charts.py`.
- Cached Streamlit data access: `src/wr_detector/apps/explorer_ui/data.py`.
- Pages: `src/wr_detector/apps/explorer_ui/pages/`.
- Query and domain logic: `src/wr_detector/modeling/explorer.py`, `src/wr_detector/modeling/cases.py`, and `src/wr_detector/modeling/layers.py`.

Pages must compose widgets, charts, tables, and short explanatory captions. They must not embed SQL, mutate datasets, train models, score prediction-pool rows, or reimplement ranking/case logic already available in tested modeling modules.

## Navigation

Use the existing Streamlit multipage navigation pattern with `st.Page` and Material icons:

- Overview: `:material/dashboard:`
- Compare models: `:material/leaderboard:`
- Model detail: `:material/query_stats:`
- Case review: `:material/travel_explore:`
- Statistics: `:material/insights:`
- Validation layers: `:material/layers:`

The sidebar owns global run selection and history provenance. Keep the run selector label as `Training run` and show the history DB path in a sidebar caption.

## Themes And Colors

Global app theming comes from `wr_detector.cli.EXPLORER_THEMES`, passed to Streamlit through CLI flags. Do not replace this with custom CSS themes.

Theme presets:

| Theme | Primary | Background | Secondary background | Text |
| --- | --- | --- | --- | --- |
| `dracula` | `#bd93f9` | `#282a36` | `#343746` | `#f8f8f2` |
| `nebula` | `#9d7bff` | `#161226` | `#221b38` | `#ece9f7` |
| `slate` | `#4da3ff` | `#0f131a` | `#171c26` | `#e6e9ef` |

Default theme: `dracula`.

Chart color conventions:

- WR class: `#f28e2b` orange.
- Negative class: `#4e79a7` blue.
- Train split: `#59a14f` green.
- CV split: `#edc948` yellow.
- Holdout split and operating point highlights: `#e15759` red.
- Single blue metric bars: `#4e79a7`.
- Feature-importance uncertainty rules: `#e6e9ef` with opacity.
- ROC/random baseline: `#888` with dashed stroke.
- Categorical model/method series: Altair `tableau10`.
- Continuous heatmaps and confusion matrices: Altair `viridis`.

Use Streamlit badges for status:

- `accepted`: `:green-badge[accepted]`.
- `overfit_warning`: `:orange-badge[overfit warning]`.
- precision-floor or threshold failures: red badge.
- unknown or miscellaneous status: gray badge.
- case detail WR badge: orange badge.
- case detail negative badge: blue badge.

## Typography And Copy

Use Streamlit native typography:

- `st.title` for page titles.
- `st.subheader` for sections.
- `st.caption` for provenance, caveats, and concise interpretation.
- `st.markdown` only for compact rich labels or badges.
- `st.code(..., language="json")` for hyperparameters.

Copy should be specific, technical, and concise. Prefer domain labels such as `WR recovered @100`, `Recall @FPR 0.5%`, `calibration negatives passing`, `Data lineage`, and `Artifact files on disk`. Avoid marketing copy, onboarding text, and visible instructions that explain obvious UI mechanics.

## Layout Patterns

Use wide layout with `st.set_page_config(page_title="WR Model Explorer", page_icon="*", layout="wide")`.

Preferred layout primitives:

- KPI rows through `ui.kpi_row`, implemented as `st.metric(..., border=True)`.
- Dense controls in `st.columns` with explicit ratios.
- Optional filters in `st.expander("Filters", expanded=False)`.
- Alternate analytical views in `st.tabs`.
- Detailed provenance and artifact paths in collapsed expanders.
- Side-by-side analysis with `st.columns(..., gap="large")`.

Avoid nested cards, custom panels, hero sections, large whitespace, decorative image areas, and full-page text explanations. Keep page sections unframed unless Streamlit native tables, metrics, charts, tabs, or expanders provide the frame.

## Tables And DataFrames

Use configured `st.dataframe` calls:

- `hide_index=True`.
- `width="stretch"`.
- Use `height` only when the table is intentionally scrollable, such as case lists.
- Hide technical join keys such as `result_id` or `source_id` with column config `None` when they are needed only for selection state.
- Use `on_select="rerun"` and `selection_mode="single-row"` when table selection drives cross-page state or detail panes.

Column formatting:

- Ranking index: `NumberColumn("#", width="small")`.
- Scores/AP/AUC/thresholds: `NumberColumn(..., format="%.4f")`.
- Photometry and compact numeric case columns: `NumberColumn(..., format="%.3f")`.
- Risk scores: `NumberColumn(..., format="%.2f")`.
- Percent fields stored as 0 to 1: `ProgressColumn(..., min_value=0.0, max_value=1.0, format="percent")`.
- Object names, model labels, statuses, SIMBAD types, and spectral types: `TextColumn`.

Keep model labels compact using `ui.short_label`: model abbreviation, sampler, dataset variant, and `+err` when parallax-error features are included.

## Chart Rules

Every chart builder should return `alt.Chart | None` and let the page render an `st.info` empty state when no chartable data exists.

Required chart conventions:

- Set `.properties(width="container", height=...)`.
- Use bounded heights. Current patterns are 160-560 for dynamic bar charts, about 280-340 for compact distributions and curves, 380 for validation scatter, and 420 for color-magnitude review.
- Include explicit tooltips with domain labels and numeric formats.
- Put legends at the bottom for grouped charts unless the chart is intentionally legend-free.
- Use log x scales only when review budgets span orders of magnitude.
- Use symlog count scales when negatives vastly outnumber WR.
- Use reversed magnitude y axes for magnitude charts.
- Use `.interactive()` only where exploration is useful, such as scatter and color-magnitude charts.

Do not use saved matplotlib PNGs as primary app visuals. The app can list artifact paths, but live PR, ROC, confusion, ranking, and case-review charts should come from synchronized predictions and database rows.

## Widget Patterns

Use these Streamlit controls consistently:

- `selectbox` for run, model, split, view mode, ranking metric, and layer choices.
- `multiselect` for filters.
- `toggle` for binary filter switches like `Accepted only`.
- `slider` for bounded top-N model comparison.
- `number_input` for review budgets such as `Budget K`.
- `button` only for direct navigation actions in detail panes.

Synchronize model selection with `ui.SELECTED_MODEL_KEY` so Compare, Model detail, Case review, and Statistics stay aligned.

## Empty States And Degraded Modes

Every page must handle missing data gracefully:

- No training runs: show an error with the history DB path.
- Selected run has no results: show a warning.
- Filters match no rows: show a warning.
- Missing synchronized predictions: show an info state; do not read sidecar CSVs in pages.
- Missing WR/SIMBAD reference DBs: degrade subtype and false-positive composition features instead of failing.
- CSV-only validation layer run: show a warning with the sync command.

## Validation-Layer UI Rules

Validation layers are compatibility scorers and re-rankers, not hard rejection gates. Keep the explanatory copy aligned with that meaning.

Layer pages should:

- Read layer specs from `VALIDATION_LAYERS`.
- Load runs from history DB first, CSV discovery second.
- Show lineage columns in a collapsed `Data lineage` expander when present.
- Warn when a run predates lineage tracking or exists only as CSV.
- Use retention vs negative pass-rate scatter as the primary visual: ideal validators sit top-left.
- Display result tables with WR retention and negative pass rates as progress columns.

Add future validation layers by adding a `ValidationLayer` spec. Avoid page-specific branching unless the layer result schema materially differs.

## Scientific App Logic

The app must preserve project modeling decisions:

- Treat WR detection as candidate ranking, not ordinary balanced classification.
- Emphasize top-K WR recovery, average precision, recall at fixed FPR, threshold-calibration negative pass rate, and overfit diagnostics.
- Use `threshold_calibration` as the false-positive stress-test pool.
- Pair validation layers with first-layer runs by dataset hashes and data lineage, not by name alone.
- Keep color-locus outliers excluded from all modeling and validation-layer training/evaluation.
- Do not make Gaia H-alpha required until full coverage has been audited for negatives and prediction-pool rows.

## Test And Review Expectations

For UI changes, run focused tests when feasible:

```powershell
pytest tests/test_model_explorer.py tests/test_cases.py tests/test_layers.py
```

For CLI/theme wiring, also run:

```powershell
pytest tests/test_cli.py
```

For manual smoke checks, launch:

```powershell
wr-detector explore-models --config configs/models.yaml --theme dracula
```

Check at least Overview, Compare models, Model detail, Case review, Statistics, and Validation layers for layout overflow, empty-state handling, broken charts, and stale copy.
