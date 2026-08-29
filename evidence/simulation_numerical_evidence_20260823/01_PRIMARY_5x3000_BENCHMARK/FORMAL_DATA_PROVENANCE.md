# Formal Primary Benchmark Provenance

This directory contains the locked task-level package for the principal
controller comparison. It is the same five-set, 3,000-task-per-set benchmark
used for the reported PF, PF-distilled imitation, and validation-selected PPO
results.

- Task-set seeds: `101`, `202`, `303`, `404`, and `555`.
- Tasks per seed: `3,000`.
- Total matched task rows: `15,000`.
- PF population: `1,000` particles.
- Titrant concentration: `0.1 M`.
- pH observation resolution: `0.01`.
- Neural action classes: `0.01`--`10.00 mL`.
- Acid/base direction: common external rule; base below target and acid above.
- Persistent post-overshoot cap: enabled for the neural protocol.

`tasks/` contains the locked task definitions. `seed_*_task_results.csv` and
`all_task_results.csv` contain task-level outcomes. `pf_reference/` contains
the archived PF outcomes evaluated on the same rows. Aggregate summaries and
paired tests are included in this directory.

The exact checkpoint hashes are recorded in `RUN_CONFIG.json`. This directory
is an evaluation manifest and outcome package. The task generator, environment,
and evaluation source used with these manifests are archived in
`../13_SOURCE_ARCHIVES/primary_locked_benchmark_source/`; the JSONL task rows
remain the authoritative inputs for exact audit and replay. The training
generator under the repository `code/` tree is a separate near/medium/far
generator and must not be presented as the formal five-set generator.
