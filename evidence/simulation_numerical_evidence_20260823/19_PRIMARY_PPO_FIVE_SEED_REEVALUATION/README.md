# Five primary PPO checkpoints: independent re-evaluation

This block archives the re-evaluation completed on 6 September 2026: five saved
primary-protocol PPO checkpoints, each tested on the same five locked sets of
3,000 tasks. There are **75,000 model-task evaluations of 15,000 unique tasks**.
No models were retrained and no checkpoints were selected using these results.

## Results and statistical units

| Training seed | Success (%) across the five benchmark sets, mean ± sample SD |
|---|---:|
| 101 | 91.68 ± 0.41 |
| 202 | 90.46 ± 0.64 |
| 303 (previously validation-selected) | 93.95 ± 0.63 |
| 404 | 92.59 ± 0.57 |
| 555 | 90.29 ± 0.43 |

Across the **five training-seed means**, success was **91.79 ± 1.53%**
(91.7933333333 ± 1.5253560604% before display rounding; 92 ± 2% with a
one-significant-digit SD). This SD is not calculated over the 25 cells or
75,000 binary outcomes. Within each table row, SD is calculated over five
benchmark sets. All SDs use the sample definition (`ddof=1`).

PPO success exceeded the matching imitation-policy success in all 25
training-seed × benchmark-set comparisons. Differences are calculated from
unrounded values. This statement concerns numerical differences; it does not
assert statistical significance in every comparison or treat repeated
evaluations of the same tasks as independent tasks.

The 15,000 selected-seed-303 results match the original block-01 PPO results
in every shared field (numeric comparison tolerance `1e-12` in the original
check). Evaluation of the other four checkpoints on this benchmark is a
post-hoc robustness analysis. This is not the separate 1,000-task stability
screen in block `03` (89.54 ± 2.23%), the step-cost screen in block `15`, or
the exploratory policy families in block `18`.

## Files

- `results/ppo_<training_seed>_benchmark_<benchmark_seed>.csv`: 25 files,
  each with 3,000 task-level endpoint outcomes and aggregate trajectory metrics.
- Matching `.json` files and `per_cell_summary.csv`: 25 cell summaries.
- `per_training_seed_summary.csv`: means and sample SDs across the five sets,
  including success, additions, overshoots, final error and delivered volume.
- `paired_set_success_vs_imitation.csv`: all 25 unrounded differences.
- `RUN_CONFIG.json`, `COMPLETE.json`, `PROGRESS.json`, `INDEPENDENT_AUDIT.json`:
  original run configuration, completion and independent arithmetic checks.
- `INPUT_PROVENANCE.json`: paths and hashes of the existing five checkpoint
  files, five task manifests and six evaluator/controller source files.
- `MANIFEST_SHA256.csv`: integrity manifest for this block. Text hashes use
  LF-normalized bytes, permitting only Windows/Git CRLF conversion.
- `source_original/`: unchanged one-off scripts that produced the archived
  results. Their paths assume the original local directory arrangement; use
  the portable entry points below for reproduction.

The original result files have not been rewritten. Their legacy diagnostic
column `strict_success` means error ≤0.05 pH; the reported success rate uses
`true_success`, defined by the final unrounded pH error ≤0.10.
These CSVs record endpoint outcomes, not every intermediate pH observation.

## Protocol and existing inputs

Checkpoint files are reused from
[`02_TEACHER_AND_IMITATION/checkpoints`](../02_TEACHER_AND_IMITATION/checkpoints/).
The five manifests are reused from
[`01_PRIMARY_5x3000_BENCHMARK/formal_matched_evaluation/tasks`](../01_PRIMARY_5x3000_BENCHMARK/formal_matched_evaluation/tasks/).
Both training seeds and benchmark labels are 101, 202, 303, 404 and 555;
they refer to separate sources of variation, not paired train/test datasets.

The portable runner calls the unchanged `rollout_network` function in
[`primary_locked_benchmark_source/run_matched_evaluation.py`](../13_SOURCE_ARCHIVES/primary_locked_benchmark_source/run_matched_evaluation.py).
Actions are selected by argmax. The persistent post-overshoot cap is enabled.
The formal evaluator uses observed pH rounded to 0.01, stopping at observed
error ≤0.10, with limits of 50 additions and 50 mL cumulative titrant. Success
is assessed from the final unrounded equilibrium pH, using error ≤0.10.
Requested additions are constrained to 0.01–10 mL. Successful-task additions
exclude unsuccessful tasks; overshoots are averaged per task, not per step.

The original re-evaluation used Python 3.11.15, NumPy 2.4.6 and PyTorch
2.13.0+cpu, four worker processes and one numerical thread per worker.
Use the repository's pinned environment and retrieve Git LFS model files
before re-evaluation. No network downloads occur during the evaluation itself.

## Verify or reproduce

Run from the repository root (commands also work in Windows PowerShell).
The arithmetic/integrity audit uses only the Python standard library and is
read-only; it neither evaluates neural networks nor trains models:

```powershell
python scripts/audit_primary_ppo_five_seeds.py
```

The audit checks all input hashes and task identities, recalculates success
from final pH, checks addition/dose limits and stop reasons, recalculates every
per-cell and per-training-seed summary, and compares the selected model with
the original formal results. It is also called by `scripts/verify_source.py`.

To reproduce all 75,000 evaluations using the existing weights:

```powershell
python scripts/reevaluate_primary_ppo_five_seeds.py --workers 4 --output runs/primary_ppo_reevaluation
```

The output directory must not already exist and cannot be inside `evidence/`.
After evaluation, the runner audits the new results against the released
task-level rows and writes `REPRODUCTION_AUDIT.json`. Numerical discrepancies
cause a nonzero exit status; the archived evidence is never overwritten.

For a short execution check only (50 tasks per set, 1,250 evaluations):

```powershell
python scripts/reevaluate_primary_ppo_five_seeds.py --workers 4 --tasks-per-set 50 --output runs/primary_ppo_smoke
```

Smoke results are explicitly labelled and compare imitation on exactly the
same task subset. They must not be reported as full benchmark results.
