# Formal PF 95.36% reproduction source

This directory is the self-contained historical runner for the PF result reported
as `95.36 +/- 0.59%` on the five locked benchmark sets (3,000 tasks per set).
It uses 1,000 particles, the `pf_pka_conc_variable_k` inference variant, and the
`hybrid_full` controller.

The matching historical teacher generation, imitation-learning, PPO-training,
and evaluation scripts are retained in the same directory. This makes the
recovered PF implementation the common source for future PF-to-policy analyses;
the primary frozen neural checkpoints and their locked outputs remain in evidence
blocks `01`--`03` and are not duplicated here.
See `ANALYSIS_ALIGNMENT.md` for the controls that must remain fixed in matched
posterior-to-control and representation studies.

The source was recovered from
`pf_95_36_reproduction_package_20260901.zip` (archive SHA-256
`75af655fae1f33a41f793d692e6894c4d637d2b78b9ede8c55b601b4a4e3a3d1`).
That archive's `reference/` directory was empty even though
`particle_controllers.py` imports `reference/original_bayesian_controller.py`.
The missing dependency here is the matching preserved historical file already
stored with the repository's PF teacher source. A fresh run using this completed
source reproduces the locked non-timing results.

## Windows reproduction

Create a Python 3.11 environment and install the pinned numerical packages:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-pf.txt
```

Run the formal benchmark:

```powershell
.\.venv\Scripts\python.exe -u bayesian_external_rule_ablation.py `
  --output-dir runs\formal_5x3000 `
  --seeds 101 202 303 404 555 `
  --tasks-per-seed 3000 `
  --particles 1000 `
  --workers 8 `
  --resume
```

Then verify it against the repository's locked manifests and PF reference rows:

```powershell
.\.venv\Scripts\python.exe verify_reproduction.py runs\formal_5x3000
```

The verifier requires byte-identical task manifests and exact agreement in every
non-timing task-result field. Timing columns are intentionally excluded because
they vary with hardware and process scheduling; they remain in the generated
files for separate reporting.

`PF_REPRODUCTION_AUDIT_20260901.json` records the completed independent rerun:
all 45,000 controller trajectories matched the preserved non-timing results, and
the formal PF result was reproduced as `95.36 +/- 0.59%`.

The locked inputs and reference outputs are not duplicated here. They are in
`01_PRIMARY_5x3000_BENCHMARK/formal_matched_evaluation/`.

For historical imitation/PPO retraining, install PyTorch in addition to the PF
requirements. Retraining is a separate seeded experiment and is not required to
verify the frozen primary PF, imitation, or PPO benchmark rows.
