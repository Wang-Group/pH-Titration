# Simulation Numerical Evidence Package (2026-08-23)

This package contains simulation/computational data only. It excludes physical experiments, laboratory records, instrument logs, operator data, Cu-SSA/Job analysis, UV-Vis files, casein data, mixed-acid experimental runs, and wastewater experiments.

The directory is organized by simulation protocol rather than by manuscript section. Results from different protocols are retained separately because they use different task generators, sample sizes, controller interfaces, training budgets, or timing definitions.

Start with `00_INDEX_AND_PROTOCOLS/SIMULATION_STUDY_INDEX.csv` and `00_INDEX_AND_PROTOCOLS/CURRENT_SIMULATION_CLAIMS.csv`.

The local pH-response RMSE values 0.0399, 0.1280 and 0.2452 are supported by
[`06_POSTERIOR_RECOVERY/local_response_reproduction_20260906`](06_POSTERIOR_RECOVERY/local_response_reproduction_20260906/README.md).
This block contains the supplied source, five 300-task manifests and 12,000
snapshots, with a recovered missing dependency and independent verification.
Its terminal diagnostic includes 96 last-state fallbacks among 1,500 tasks;
the README explains the anchoring and checkpoint definitions.

The independent re-evaluation of all five primary PPO checkpoints is in
[`19_PRIMARY_PPO_FIVE_SEED_REEVALUATION`](19_PRIMARY_PPO_FIVE_SEED_REEVALUATION/README.md).
It contains 75,000 model-task outcomes on the existing 15,000 locked tasks,
with an across-training-seed success mean and sample SD of 91.79 ± 1.53%.
No models were retrained. Read-only verification and portable re-evaluation
commands are documented in that block; these results are separate from the
1,000-task stability screen in block `03`.

The analysis source is in `13_SOURCE_ARCHIVES`. The source folders
are grouped by protocol. The primary locked benchmark source is in
`primary_locked_benchmark_source`, and the original rule/PID runner is in
`historical_baseline_runner_20260817`. The locked five formal task manifests
and task-level outcomes are authoritative for the reported primary result, and
their hashes match the original matched-evaluation snapshot.

The five PPO stability checkpoints and their SHA-256 values are recorded in
`03_PPO_TRAINING_STABILITY/CHECKPOINT_PROVENANCE.json`; they are separate from
the deployment checkpoint file in the controller package. The block also
contains the common 1,000-task manifest, five per-seed 500-task validation
manifests, five task-level locked-test outcome files, combined evaluation
tables, and run completion metadata. Seed 303 has the same actor tensors as
the validation-selected deployment checkpoint, although the serialized files
are stored separately.

The local PPO step-cost sensitivity materials are in
`15_PPO_STEP_COST_TUNING`. The reported range is `0` to `0.01` (`0`,
`0.0025`, `0.005`, and `0.01`). The block deliberately retains both the
original validation-selected `0.005` checkpoint and an independently
retrained `0.005` checkpoint so that training stochasticity is visible.

The current timing evidence is split by protocol. Block
`16_MATCHED_TIMING_RECOVERY_100TASKS` retains the matched single-step calls and
one-observation PF/PyMC recovery comparison. Block
`17_PF_CLOSED_LOOP_TIMING_100TASKS` contains PF success and timing measured
together during complete trajectories on the same 100-task cohort. The PF
full-trajectory timing pools all recorded decision cycles and did not use the
CPU-affinity control of the block-16 single-step calls, so cross-block values
are practical matched-cohort comparisons rather than identical-call timing.
Blocks `10`, `11`, and `12` retain earlier protocols for provenance.

The expanded controller and representation analyses are in
`18_CONTROLLER_REPRESENTATION_FACTORIAL`. This block contains the three
posterior-to-control strategies, the effective-*K*-protic-acid and
independent-component PF representations, two model-mismatch benchmarks, and
the F2-F6 exploratory imitation/PPO factorial reported in SI Section 5.6. It
includes all 15 locked 3,000-task manifests, the exact additional source,
machine-readable Tables S14-S16, unrounded policy summaries, completion
metadata, and checkpoint fingerprints. Run
`python scripts/audit_controller_representation_factorial.py` from the
repository root to verify the block.
