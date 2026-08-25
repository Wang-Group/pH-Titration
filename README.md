# pH-control code release

This directory is the installable deployment and training-source subset. The
controllers are importable as packages:

```powershell
python -m pip install -e .
python scripts/verify_source.py
python scripts/audit_primary_benchmark.py
python scripts/generate_publication_tables.py
python -m controllers.controller_package_self_test
python -m unittest discover -s tests -v
```

`ph_control_reproducibility_release_20260825_completed.zip` is the current simulation
release. Historical controller-only archives are not part of this release;
use the source package and evidence included here.

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
setting, and dose limits. The `standard` training profile uses 500 PPO
validation tasks, matching the released checkpoint-selection run.

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
| `08_SENSOR_STRESS` | Sensor nonideality stress tests |
| `09_PF_RULE_ABLATIONS` | PF rule/reward/dose ablations |
| `10_PARTICLE_SCALING` | Particle-count timing and scaling |
| `11_PYMC_COMPARISON` | PyMC/PF comparison |
| `12_ONLINE_TIMING` | Single-step PyMC and neural timing |
| `13_SOURCE_ARCHIVES` | Analysis source grouped by protocol |

The evidence directory also contains the study index and statistical-unit
definitions under `00_INDEX_AND_PROTOCOLS`. Results from different blocks are
kept separate because their task generators, sample sizes, action interfaces,
training budgets, and timing definitions differ.

Timing values must be read with their protocol labels. The 65.31 ms PF value is
the median per-step result from the 100-task particle-scaling benchmark, whereas
the 59.33 ms PF and 0.141 ms PPO values in the sensor-stress block are means
from a separate five-seed stress-timing protocol. The 0.1299 ms imitation and
0.1279 ms PPO values are means over 30,000 direct neural timing trials. These
values are not interchangeable measurements of one experiment.

## Reproducibility status

The release now contains the latest simulation-only evidence archive. It
includes the current primary summary
(`01_PRIMARY_5x3000_BENCHMARK/PRIMARY_BENCHMARK_SUMMARY.csv`), paired
statistical tables, training and evaluation manifests, RL algorithm-screen
source/checkpoints, posterior and sensor-stress analyses, particle-scaling
timing, PyMC timing, and neural-policy timing. The teacher, imitation, and PPO
training source is retained in block `02` separately from the deployable
controller API.

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
The same block includes the common 1,000-task manifest, each seed's training
and validation manifests, task-level locked-test outcomes, per-run completion
metadata, and the combined task-level evaluation table. Run
`python scripts/audit_ppo_stability.py` to validate those materials and
recompute the five-seed mean and sample SD.

The archive also includes the PID tuning source and selected parameters,
paired statistical tests, posterior-recovery and prior-sensitivity source,
sensor-stress source/results, particle-scaling timing, PyMC timing, and neural
single-step timing. Run `python scripts/verify_source.py` to compile the
package and verify these required files, the 5 x 3,000 manifests, the 15,000
matched task keys, and all teacher/imitation/principal-PPO hashes.

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
