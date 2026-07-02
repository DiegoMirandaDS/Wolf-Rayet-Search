---
name: wr-detector-ui-style
description: Project-specific UI and app-logic guidelines for the Wolf-Rayet Detector Streamlit Model Explorer. Use when creating, reviewing, or changing frontend code under src/wr_detector/apps, explorer_ui pages, Altair charts, Streamlit widgets, UI copy, layout, theme handling, case-review screens, validation-layer screens, or any user-facing app workflow.
---

# WR Detector UI Style

Use this skill before changing the Model Explorer or adding any new app screen for this project. The app is a dense scientific operations tool for model audit and candidate review, not a marketing site or general dashboard template.

## Required Reference

Read `references/model-explorer-style.md` before implementation when the task affects:

- Streamlit page layout, navigation, filters, widgets, tables, cards, captions, or status badges.
- Altair charts, colors, legends, heights, axis labels, chart interactivity, or chart tooltips.
- Theme presets in `wr_detector.cli.EXPLORER_THEMES`.
- Data loading, case review, validation-layer integration, or app/business logic placement.

## Core Rules

- Keep the UI operational, compact, and evidence-first. Prioritize fast comparison, auditability, and case-level review over decorative presentation.
- Preserve the multipage Streamlit structure: entry point in `src/wr_detector/apps/model_explorer.py`; page, widget, and chart code under `src/wr_detector/apps/explorer_ui/`.
- Keep SQL, joins, ranking logic, and business logic out of pages. Pages compose views; tested modules under `src/wr_detector/modeling/` own data semantics.
- Use Altair for charts. Charts must use `width="container"`, bounded heights, Streamlit theme inheritance, explicit tooltips, and clear legends when grouped.
- Use Streamlit theme flags and `EXPLORER_THEMES` for global colors. Do not add broad custom CSS or one-off decorative styling.
- Favor configured tables over raw tables: `hide_index=True`, `width="stretch"`, and `st.column_config` with percentage progress bars and metric number formats.
- Treat validation layers as read-only audit layers. Add future layers through `wr_detector.modeling.layers.VALIDATION_LAYERS` unless the schema truly diverges.
- Keep generated plots and model artifacts listed by path only; live charts in the app should be computed from synchronized database rows.

## Implementation Checklist

Before handing off UI work:

- Confirm the page still loads through `wr-detector explore-models --config configs/models.yaml --theme dracula`.
- Check that charts fit at desktop widths and do not overflow.
- Check empty-state behavior for missing run results, missing predictions, missing reference DBs, and unsynced layer runs.
- Confirm no page introduced direct SQL, dataset mutation, model training, prediction-pool scoring, or notebook-only logic.
- Update `AGENTS.md` if the UI or app-logic convention changed in a way future agents must follow.
