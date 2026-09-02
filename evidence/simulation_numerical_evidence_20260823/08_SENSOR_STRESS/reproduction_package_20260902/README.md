# Sensor-stress source and raw results

This directory contains the source and raw task-level outputs for the published
sensor-stress comparison of `new_pf` and the NumPy implementation of the
selected PPO checkpoint.

## Contents

- `runner/fixed_controller_stress_benchmark.py`: original benchmark runner.
- `runner/RUN_FIXED_CONTROLLER_STRESS.cmd`: Windows entry point.
- `runner/study_source/`: task generator and chemistry source imported by the
  runner.
- `runner/controllers_release/`: exact controller source and checkpoints used
  by the runner. This mirrors the runner's expected relative paths.
- `controller_source/controllers_release/`: preserved controller-source copy
  with its source-package SHA-256 manifest.
- `results/fixed_pf_ppo_stress/all_task_results.csv`: 150,000 task-method
  result rows (15 regimes x 5 seeds x 1,000 tasks x 2 methods).
- `results/fixed_pf_ppo_stress/*_tasks.jsonl`: 75 task manifests, one for each
  regime and benchmark seed.
- `results/fixed_pf_ppo_stress/completed_shards/*.csv`: 75 resumable result
  shards, one for each regime and seed.
- `results/fixed_pf_ppo_stress/aggregate_summary.csv`: regime-level summary.
- `results/fixed_pf_ppo_stress/per_seed_summary.csv`: seed-level summary.
- `results/fixed_pf_ppo_stress/paired_success_tests.csv`: paired exact tests.
- `results/fixed_pf_ppo_stress/paired_continuous_tests.csv`: paired continuous
  metric tests.
- `results/fixed_pf_ppo_stress/RUN_CONFIG.json`: complete protocol settings.
- `results/fixed_pf_ppo_stress/BENCHMARK_COMPLETE.json`: completion record.
- `source_archive_SHA256SUMS.txt`: SHA-256 manifest from the source archive.

## Protocol

The benchmark uses seeds `101, 202, 303, 404, 555`, 1,000 tasks per seed,
15 stress regimes, CPU execution, NumPy PPO inference, and 24 worker
processes in the completed run. The methods are `new_pf` and `new_ppo`.

The regimes are `nominal`, `close_pka`, `wide_concentration`, observation
noise SD `0.01`, `0.03`, `0.05`, and `0.10` pH, episode bias SD `0.10` pH,
random-walk drift increment SD `0.01` pH, response fractions `0.60` and
`0.70`, actuator log-SD `0.10`, titrant scales `0.90` and `1.10`, and the
combined unseen condition.

The runner records final task-level outcomes including true and measured pH,
success flags, false stops, severe failures, additions, crossings, total dose,
final error, total decision/update time, and the derived controller time per
step. The task JSONL files record the generated chemistry and initial task
state. They are not per-step posterior-state logs; the original runner does
not serialize particle arrays or posterior snapshots at every observation.

## Source hashes

The controller-source manifest is
`controller_source/controllers_release/SHA256SUMS.txt`. Key hashes are:

```text
fixed_controller_stress_benchmark.py
871f2a1547220812621177ed9f85fb1681fb63695ce7e3882498d5d343d9aca4

controllers_release/chemistry_model.py
b1714a78a29006b29883d5aa73bde905fdf627ac4468ac7b71bd6168485516aa

controllers_release/particle_inference.py
0ff9c745aca3779d92fe77de8d4f488adca0c51b1404c8aba13008dfdacc23a3

controllers_release/new_pf_controller.py
8c1577cfdde3f1da92c4a17db6bd69291c15db39d7647d72b7a89d2a5745da0a

controllers_release/models.py
484ecb3b332602b20c698a2f0596ddf43a3244a1ff445d7fdc1bee5cdcf31864

controllers_release/models/ppo_seed_303.pth
496797be6be22dacd9f7360e7980a983dca816b7bded19597c4be6beb58abd23

results/fixed_pf_ppo_stress/all_task_results.csv
b3878a8e8905a1d816c52d98a1a6d4072d5c0b5a5b6c5f7b819a48207136aef3
```

The same source and result hashes are also retained in
`source_archive_SHA256SUMS.txt`.

## Re-running

From `runner/`, install the dependencies used by the source package and run:

```powershell
py -3.11 fixed_controller_stress_benchmark.py `
  --output-dir results/fixed_pf_ppo_stress `
  --seeds 101 202 303 404 555 `
  --tasks-per-seed 1000 `
  --device cpu `
  --ppo-backend numpy `
  --workers 16 `
  --resume
```

The checked-in results were generated on CPU. The output directory must be
empty for a fresh run, or `--resume` must be used with the same scientific
configuration.
