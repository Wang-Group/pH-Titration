# pH-control code and data release

This repository exposes the controller source, training workflows, numerical
evidence, and physical-experiment data used for the revised manuscript. The
source tree is directly inspectable and installable; reviewers do not need to
unpack an archive before running the verification commands.

## Repository layout

| Path | Contents |
|---|---|
| `controllers/` | Deployable PF and PPO controller APIs and selected model weights |
| `training/` | Teacher generation, imitation learning, and PPO training source |
| `scripts/` | Verification, audit, table-generation, and training entry points |
| `tests/` | Release-contract unit tests |
| `evidence/` | Locked simulation tasks, task-level outcomes, analyses, and source snapshots |
| `physical_experiments/` | Mixed-acid, wastewater, casein, and Cu–SSA experimental data |
| `release_archives/` | Downloadable snapshots of the simulation and physical-data releases |

The controllers are importable as packages:

```powershell
python -m pip install -e .
python scripts/verify_source.py
python scripts/audit_primary_benchmark.py
python scripts/audit_controller_representation_factorial.py
python scripts/audit_pf_internal_rule_ablation.py
python scripts/audit_primary_ppo_five_seeds.py
python scripts/audit_pf_local_curves.py
python scripts/generate_publication_tables.py
python -m controllers.controller_package_self_test
python -m unittest discover -s tests -v
```

The downloadable snapshots can be rebuilt from the visible repository tree:

```powershell
python scripts/build_reproducibility_release.py
python scripts/build_physical_data_archive.py
```

Downloadable snapshots are provided as
`release_archives/ph_control_reproducibility_release_20260825.zip` and
`release_archives/physical_experiments_20260825.zip`. The unpacked repository
tree is the preferred review interface.

## Training entry points

The training-only pipeline has three explicitly separated profiles:

```powershell
# Short technical execution check; not a publication reproduction
python scripts/train_pipeline.py --profile smoke --output-dir runs/smoke

# Release-scale workflow used for publication-oriented checkpoint selection
python scripts/train_pipeline.py --profile standard --output-dir runs/standard
```

`training/train_ppo.py` keeps the publication reward as its default
(`--step-cost 0.005`) and also exposes `--step-cost` for the archived local
sensitivity screen.

The `smoke` profile uses intentionally small task counts to test that the
pipeline executes. It must not be used to reproduce or support manuscript
results. The `standard` profile uses 500 closed-loop validation tasks for
imitation checkpoint selection and 500 validation tasks for PPO checkpoint
selection. There is no intermediate 300-task publication profile.

The public controller API is `controllers.RobustPFController`,
`controllers.PPOVolumeController`, and
`controllers.NumpyPPOVolumeController`. Every controller validates its pH
tolerance, positive step limit, nonnegative total-dose limit, and titrant
concentration. PPO action logits and PF candidate volumes are masked by the
remaining total-dose allowance; an observed delivered volume that exceeds the
allowance is rejected. Both PPO deployment backends also apply the shared
persistent post-overshoot cap: after a target crossing or an increase in
absolute observed-pH error, later doses are limited to half of the triggering
delivered dose, and the cap can only decrease until `reset()`. The independent
hardware layer must enforce the same total-dose limit.

## Protocol profiles

The release uses the `pH-control` protocol family, version `2026.08`. The
profiles are explicit because the formal evaluator, training environment, and
deployment API do not use identical endpoint operators:

| Profile | Stop rule | Reported success | Dose limits | Neural post-overshoot cap |
|---|---|---|---|---|
| `formal_evaluation` | rounded observed error `<= 0.10` pH | final unrounded equilibrium error `<= 0.10` pH | 50 additions and 50 mL total | enabled by the formal evaluator |
| `training_environment_strict` | rounded observed error `< 0.10` pH | final unrounded equilibrium error `< 0.10` pH | 50 additions and 50 mL total | disabled |
| `deployment_api_strict` | rounded observed error `< 0.10` pH | no independent true-pH success label | 50 additions and 50 mL total by default | enabled in the released PPO wrappers |

Controller `status()` dictionaries and newly written training checkpoints
include the applicable protocol family, version, profile, stopping rule, cap
setting, and dose limits. The `standard` training profile uses 500 closed-loop
validation tasks for imitation checkpoint selection and 500 PPO validation
tasks, matching the released publication workflow.

## Checkpoint provenance

`controllers/models/ppo_seed_303.pth` has SHA-256
`4004d7a09768fc5ac3f448523f53cb22210ed919ca7e713f13d9aa693cc19de5`.
Its actor-tensor SHA-256 is
`8c2ccdbc7879d1d54b151eeccb76ed3d354e58ace23e8052f6619d57369214fb`,
which is unchanged by the protocol-metadata annotation. The checkpoint file
before that metadata annotation had SHA-256
`bafd85f896945245f4a2275764ee74cfb458aae78cbe91f5c17396c24fd22f1c`;
that value remains in historical task-level result records. The embedded
metadata records the 500-task validation run used during checkpoint selection
(`seed=303`, `100023` environment interactions, `92.2%` validation success).
The final locked evaluation is a separate evaluation of this frozen checkpoint
on five independently generated sets of 3,000 matched tasks; it must not be
described as the embedded 500-task validation result.

The current `training/task_distribution.py` is the teacher/PPO training
generator with near/medium/far tasks. It is intentionally separate from the
locked five-set benchmark generator. The complete simulation evidence,
including the primary benchmark summary, paired tests, task manifests, source
scripts, checkpoints, posterior/stress/timing analyses, and RL algorithm
screen, is included under:

`evidence/simulation_numerical_evidence_20260823/`

Analysis source files are archived under
`evidence/simulation_numerical_evidence_20260823/13_SOURCE_ARCHIVES/`. This
includes the PID-tuning and baseline source, the PF/PyMC analysis source, and
an auxiliary fixed-3000 analysis runner. The auxiliary runner is not the
generator for the current locked primary summary.

Start with its `README_CN.md`, then read the study index and statistical-unit
definitions under `00_INDEX_AND_PROTOCOLS`.

The principal benchmark is fully task-level and locked under
`01_PRIMARY_5x3000_BENCHMARK/formal_matched_evaluation/`: five JSONL task
manifests contain 3,000 tasks each, and `all_task_results.csv` contains the
matched PF, imitation, and PPO outcomes for the 15,000 unique tasks. The
corresponding run configuration, PF references, paired tests, and completion
records are in the same directory. The locked benchmark source is archived in
`13_SOURCE_ARCHIVES/primary_locked_benchmark_source/`, while the baseline
controller source is archived in
`13_SOURCE_ARCHIVES/historical_baseline_runner_20260817/`. The five manifests
are byte-identical to the original matched-evaluation snapshot. The archived
generator reproduces all five manifests byte-for-byte; this is recorded in
`01_PRIMARY_5x3000_BENCHMARK/formal_matched_evaluation/PRIMARY_MANIFEST_REGENERATION_AUDIT.json`.

The historical PF runner that reproduces the reported `95.36 +/- 0.59%`
result is under
`13_SOURCE_ARCHIVES/formal_pf_95_36_reproduction_20260901/`. Its verifier
checks all non-timing task-result fields against the locked PF reference and
reports machine-dependent timing separately.
The public `controllers.RobustPFController` is a later deployable API; it must
not be substituted for this historical runner when reproducing the manuscript's
primary PF benchmark or when constructing a matched posterior-to-control
ablation.

The original simple-rule and PID task-level outputs are under
`01_PRIMARY_5x3000_BENCHMARK/formal_matched_evaluation/rule_baseline_replay/`.
They reproduce the reported five-set means: simple rule, `77.28 +/- 0.41%`
success and `16.11 +/- 0.33` successful steps; prespecified PID,
`84.59 +/- 0.68%` and `17.14 +/- 0.28`; and tuned PID,
`92.44 +/- 0.58%` and `14.75 +/- 0.20`. The PID label
`selected_pid` is normalized to the manuscript label `tuned_pid` in the
release CSV files.

The training generator in `training/task_distribution.py` is intentionally a
separate near/medium/far teacher/PPO generator. It is not the locked primary
5 x 3,000 benchmark generator and must not be used to claim reproduction of
the principal table. Teacher data, the imitation checkpoint, all five
principal-protocol PPO checkpoints, and their SHA-256 provenance are under
`02_TEACHER_AND_IMITATION/`. The five stability-screen PPO checkpoints remain
separate under `03_PPO_TRAINING_STABILITY/`.

## Evidence and source index

The complete simulation evidence is included as a separate simulation-only
release block:

| Block | Contents |
|---|---|
| `01_PRIMARY_5x3000_BENCHMARK` | Main PF/imitation/PPO evidence and paired tests |
| `02_TEACHER_AND_IMITATION` | PF teacher generation, imitation training, PPO source |
| `03_PPO_TRAINING_STABILITY` | Independent PPO training-seed stability |
| `04_RANDOM_INITIALIZATION` | Random-vs-imitation PPO initialization |
| `05_RL_ALGORITHM_SCREEN` | Matched PPO/A2C/REINFORCE screen, manifests and tests |
| `06_POSTERIOR_RECOVERY` | Posterior recovery and natural-endpoint analysis |
| `07_PRIOR_SENSITIVITY` | Prior and likelihood sensitivity |
| `08_SENSOR_STRESS` | Sensor nonideality stress tests, including source and raw task-level results |
| `09_PF_RULE_ABLATIONS` | Current S6 dose-rule source, manifests and 9,000 task outcomes, plus separate historical rule/reward ablations |
| `10_PARTICLE_SCALING` | Particle-count timing and scaling |
| `11_PYMC_COMPARISON` | Earlier 15-task PyMC/PF comparison retained for provenance |
| `12_ONLINE_TIMING` | Earlier separate PyMC and neural timing protocols retained for provenance |
| `13_SOURCE_ARCHIVES` | Analysis source grouped by protocol |
| `15_PPO_STEP_COST_TUNING` | Local PPO step-cost sensitivity from 0 to 0.01, including the independent 0.005 retraining |
| `16_MATCHED_TIMING_RECOVERY_100TASKS` | Matched single-step PyMC/PF/neural timing and one-observation recovery benchmark |
| `17_PF_CLOSED_LOOP_TIMING_100TASKS` | Current PF complete-trajectory timing and outcome benchmark on the same 100-task cohort |
| `18_CONTROLLER_REPRESENTATION_FACTORIAL` | Posterior-to-control, PF-representation, model-mismatch, and exploratory imitation/PPO factorials reported in SI Section 5.6 |
| `19_PRIMARY_PPO_FIVE_SEED_REEVALUATION` | Independent evaluation of all five primary PPO checkpoints: 75,000 task outcomes, unrounded summaries, audit and re-evaluation entry points |

The evidence directory also contains the study index and statistical-unit
definitions under `00_INDEX_AND_PROTOCOLS`. Results from different blocks are
kept separate because their task generators, sample sizes, action interfaces,
training budgets, and timing definitions differ.

The direction-only random-initialization ablation in block `04` uses the same
frozen imitation checkpoint as the primary benchmark but a separate evaluation
protocol and separate locked task sets (evaluation seeds `901`--`905`). The
checkpoint achieved `87.82%` pooled success in block `04`, compared with
`89.17 +/- 0.51%` on the primary benchmark. These values describe different
evaluation protocols and should not be merged.

Timing values must be read with their protocol labels. Block
[`16_MATCHED_TIMING_RECOVERY_100TASKS`](evidence/simulation_numerical_evidence_20260823/16_MATCHED_TIMING_RECOVERY_100TASKS/PROTOCOL_AND_RESULTS.md)
contains the CPU-affinity-controlled single-step calls: median
observation-to-action times were 14,407.376 ms for variable-*K* PyMC, 0.15495
ms for imitation, and 0.15390 ms for PPO. Block
[`17_PF_CLOSED_LOOP_TIMING_100TASKS`](evidence/simulation_numerical_evidence_20260823/17_PF_CLOSED_LOOP_TIMING_100TASKS/PROTOCOL_AND_RESULTS.md)
contains PF outcomes and timing recorded together over complete trajectories.
The pooled median PF times were 40.131, 93.046, and 594.127 ms per recorded
decision cycle for 1,000, 10,000, and 100,000 particles, respectively; all
three configurations achieved 97.0% success on the same 100-task cohort. The
PF run used single-thread numerical settings but did not impose the CPU
affinity used for the block-16 single-step calls. These values support a
matched-cohort practical wall-time comparison, not an identical-call,
fully hardware-controlled head-to-head benchmark. The older timing protocols
in blocks 10-12 remain available for provenance and are not interchangeable
with either current scope.

## Reproducibility status

The [local-response reconstruction package](evidence/simulation_numerical_evidence_20260823/06_POSTERIOR_RECOVERY/local_response_reproduction_20260906/README.md)
now supplies the code, five locked 300-task manifests and 12,000 snapshots
supporting RMSE values `0.0399`, `0.1280` and `0.2452` pH for the ±0.10,
±0.50 and ±1.00 mL windows. These compare pH changes after anchoring both
curves at zero additional dose. The archived terminal label includes 96
12-iteration fallbacks among 1,500 tasks, not only successful stops.
`python scripts/audit_pf_local_curves.py` recalculates all summaries;
`python scripts/reproduce_pf_local_curves.py --workers 8 --output runs/pf_local_curves`
replays the exact tasks with the recovered historical dependency.

The release now contains the latest simulation-only evidence archive. It
includes the current primary summary
(`01_PRIMARY_5x3000_BENCHMARK/PRIMARY_BENCHMARK_SUMMARY.csv`), paired
statistical tables, training and evaluation manifests, RL algorithm-screen
source/checkpoints, posterior and sensor-stress analyses, particle-scaling
timing, PyMC timing, and neural-policy timing. The teacher, imitation, and PPO
training source is retained in block `02` separately from the deployable
controller API.

Block [`08_SENSOR_STRESS`](evidence/simulation_numerical_evidence_20260823/08_SENSOR_STRESS/)
retains the compact publication summaries under `current_pf_noise_stress` and
the complete reproducibility package under
[`reproduction_package_20260902`](evidence/simulation_numerical_evidence_20260823/08_SENSOR_STRESS/reproduction_package_20260902/README.md).
The latter contains the benchmark runner, exact controller snapshots and model
weights, 75 locked task manifests, 75 resumable result shards, and all 150,000
PF/PPO task-level outcomes across the 15 stress regimes.

The current PyMC timing and one-observation posterior comparison is archived in
block `16` with task-level raw outputs, run configurations, paired tests, and
the exact worker, launcher, and finalizer scripts. The timed interval is the
computational path from a new rounded pH observation to the returned next
action. It excludes liquid delivery, mixing, electrode stabilization, and pH
acquisition and therefore must not be described as timing collected during a
physical experiment.

The current SI Table S6 and Response Table R7 are supported by
[`09_PF_RULE_ABLATIONS/internal_rule_reproduction_20260905`](evidence/simulation_numerical_evidence_20260823/09_PF_RULE_ABLATIONS/internal_rule_reproduction_20260905/README.md).
This block contains the exact runner and controller snapshot, five 300-task
manifests, all 9,000 outcomes, unrounded summaries, and paired tests.
`python scripts/audit_pf_internal_rule_ablation.py` independently recalculates
the reported statistics. The older `eq_s5_*` files elsewhere in block `09`
belong to a different study and are not the source for the current S6 table.

The current complete-trajectory PF timing and outcome records are archived in
block `17`. Task-level outcomes, every recorded decision cycle, and complete
trajectories are retained, and the derived tables can be regenerated with
`python scripts/finalize_pf_closed_loop_timing_100tasks.py`.

The expanded controller and representation analyses are archived in block
[`18_CONTROLLER_REPRESENTATION_FACTORIAL`](evidence/simulation_numerical_evidence_20260823/18_CONTROLLER_REPRESENTATION_FACTORIAL/README.md).
This block contains all 15 locked 3,000-task manifests, the exact additional
source, publication Tables S14-S16 in machine-readable form, unrounded
policy-factorial summaries, completion metadata, and checkpoint fingerprints.
Run `python scripts/audit_controller_representation_factorial.py` to verify the
manifests, source, table values, and the reported 12-of-15 and 45-of-75 PPO
comparisons.

The embedded `controllers/models/ppo_seed_303.pth` is the validation-selected
deployment checkpoint. Its metadata describes a 500-task validation run; the
reported principal result is the separate frozen-checkpoint evaluation on five
sets of 3,000 matched tasks. These are both archived but are not
interchangeable.

The five checkpoints under
`evidence/simulation_numerical_evidence_20260823/03_PPO_TRAINING_STABILITY/checkpoints/`
are the independent stability-screen models used for the `89.54 +/- 2.23%`
held-out result. Seed 303 has the same actor tensors as the validation-selected
deployment checkpoint; the stability block records its role as one of five
independently seeded training runs.
The [secondary stability-screen summary](evidence/simulation_numerical_evidence_20260823/03_PPO_TRAINING_STABILITY/evaluation/RESULT_SUMMARY.md)
documents the common 1,000-task comparison of the PF-distilled imitation policy
(`89.10%`), the five PPO checkpoints (`89.54 +/- 2.23%`), and the PF teacher
(`95.10%`). This archived comparison is not the current primary
five-training-seed x five-benchmark analysis (`91.79 +/- 1.53%`).
The independently reproduced primary analysis is now archived in
[`19_PRIMARY_PPO_FIVE_SEED_REEVALUATION`](evidence/simulation_numerical_evidence_20260823/19_PRIMARY_PPO_FIVE_SEED_REEVALUATION/README.md):
all 75,000 evaluations, cell/seed summaries, and comparisons with imitation.
The five checkpoint files in block `02` and locked manifests in block `01`
are reused without retraining or checkpoint selection. The selected model's
15,000 results match the original formal evaluation. Run
`python scripts/audit_primary_ppo_five_seeds.py` for a read-only arithmetic
audit, or reproduce all five models with
`python scripts/reevaluate_primary_ppo_five_seeds.py --workers 4 --output runs/primary_ppo_reevaluation`.

Block `03` also includes the common 1,000-task manifest, each seed's training
and validation manifests, task-level locked-test outcomes, per-run completion
metadata, and the combined task-level evaluation table. Run
`python scripts/audit_ppo_stability.py` to validate those materials and
recompute the five-seed mean and sample SD.

The [PPO step-cost sensitivity block](evidence/simulation_numerical_evidence_20260823/15_PPO_STEP_COST_TUNING/README.md)
reports coefficients `0`, `0.0025`, `0.005`, and `0.01`. The independently
retrained `0.005` checkpoint achieved `91.86 +/- 0.56%` across the five locked
benchmark sets, whereas the original independently validation-selected PPO,
also trained with `0.005`, achieved `93.95 +/- 0.63%`. Both are retained to
make the stochastic training dependence explicit. This is a one-retraining-run
per coefficient sensitivity screen, not a causal hyperparameter comparison.

The archive also includes the PID tuning source and selected parameters,
paired statistical tests, posterior-recovery source/results,
prior-sensitivity results, sensor-stress source/results, historical timing
protocols, the matched single-step PyMC/PF/neural benchmark, and the current PF
complete-trajectory timing benchmark. Run
`python scripts/verify_source.py` to compile the package and verify these
required files, the 5 x 3,000 manifests, the 15,000 matched task keys, and all
teacher/imitation/principal-PPO hashes.

The released rule/PID outputs can be checked with
`python scripts/replay_primary_rule_baselines.py`. This command validates the
45,000 task-level rows, regenerates `comparison_to_reported.csv`, and confirms
the reported aggregate values without modifying the released output.

`python scripts/audit_primary_benchmark.py` validates the five locked manifests,
checks the 15,000 matched task keys and three primary method rows per task, and
regenerates the released summary plus `PRIMARY_REPRODUCTION_AUDIT.json`.

`python scripts/generate_publication_tables.py` is the official table-generation
entry point. It reads the locked task-level CSVs and writes CSV and Markdown
versions of the primary controller comparison under
`formal_matched_evaluation/publication_tables/`.

Released PPO metadata excludes the internal strict-success diagnostic field;
strict-success values remain available in the task-level result tables.

## Numerical regularization

After Liu--West rejuvenation, pKa proposals are sorted and clipped to `1--10`
before the next equilibrium solve. This broad envelope prevents extreme
proposals from producing unstable charge-balance calculations while retaining
the nominal `1.5--9.0` initialization range. It is a numerical stability
safeguard, not a species-specific prior or a universal thermodynamic pKa
claim.
