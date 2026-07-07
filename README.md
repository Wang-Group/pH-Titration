# pH-Titration Workspace

This repository workspace contains the code, data, review-analysis copies, and reproducibility helpers used for the pH-titration manuscript and revision work.

## Main folders

- `ph4github/`
  Main public-facing code and data package. This is the folder most appropriate to treat as the core GitHub package.
- `ph4github_analysiscopy/`
  Reviewer-analysis working copy with extracted text, audit outputs, and intermediate review artifacts.
- `ph4github_reprocopy/`
  Historical reproducibility-focused working copy prepared during the revision audit.

## Root-level assets

- `train_set_big_new1.json`
- `validation_set_big_new1.json`
- `test_set_big_new1.json`
- `volume_regressor_best_big_discrete_new1.pth`
- `volume_regressor_best_big_discrete_new1_trained-1.pth`

These remain at the workspace root because some review and timing scripts reference them directly.

## Reviewer-response related additions

The revision work added a small set of reusable scripts and notes:

- `ph4github/repro_support/`
  Lightweight checks and reproducibility helpers.
- `ph4github/review_support/`
  Expert-rule baseline, timing benchmark, and theoretical titration-curve scripts.
- `ph4github/tools/export_notebook_cells.py`
  Utility to export each notebook cell into a separate `.py` file.
- `ph4github/main_code3_cells/`
  Generated per-cell Python exports from `main_code3.ipynb`.

For a short inventory, see [REVIEW_RESPONSE_ASSETS.md](REVIEW_RESPONSE_ASSETS.md).

## Recommendation for GitHub presentation

If you want a cleaner external presentation, treat `ph4github/` as the main package and keep the other two copies as supporting internal revision folders. The `ph4github` subfolder now contains its own README, requirements, reproducibility notes, and helper scripts.
