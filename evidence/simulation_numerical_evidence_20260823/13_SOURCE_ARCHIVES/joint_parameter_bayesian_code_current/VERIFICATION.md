# Verification record

Verified on Windows, Python 3.11.9, on 2026-08-11.

## Static checks

- All Python files passed `python -m compileall`.
- `ONE_CLICK.ipynb` passed JSON parsing.

## Particle-filter checks

- All three controller variants completed a closed-loop smoke task.
- All three inference variants replayed the same fixed trajectory.
- Control summaries, McNemar tests, continuous paired tests, curve summaries, confusion matrices, PNG/SVG/PDF figures, CSV files, and JSON files were generated successfully.

## PyMC check

Tested with the exact versions in `requirements_tested.txt`.

- `pymc_pka_only_k3` completed with `pm.sample_smc`.
- `pymc_pka_conc_k3` completed with `pm.sample_smc`.
- `pymc_pka_conc_variable_k` completed all K=1,2,3 models, extracted finite marginal-likelihood estimates, calculated model posterior probabilities, and exported results.
- A one-task, 20-draw, one-chain end-to-end smoke run completed successfully.

That intentionally tiny PyMC test took approximately 188 seconds after environment setup, while the three matched particle-filter replays each took less than 0.3 seconds. The quick PyMC settings are only an installation/interface check. Use the standard or full profile for evidence, and interpret the number of fitted tasks explicitly.
