# Controller and representation factorial

This block contains the expanded analyses reported in Supporting Information
Section 5.6 and Tables S14-S16. It separates three questions:

1. how the particle-filter (PF) posterior is converted into a required-volume
   estimate;
2. how the assumed PF representation affects endpoint control and full-curve
   reconstruction; and
3. how PF teacher construction, representation, and training domain affect
   fixed imitation and PPO policies.

## Names used in the code and paper

The internal identifier `sequential_k123` denotes the manuscript's effective
*K*-protic-acid representation, in which one effective response model contains
one, two, or three successive dissociation transitions. The identifier
`independent_j123` denotes samples containing one, two, or three separate
monoprotic components. These identifiers are retained in the source and raw
summaries so the archived runs remain auditable; the publication tables use
the more readable manuscript terminology.

## Publication tables

- `results/table_s14_posterior_to_control.csv` contains the three
  posterior-to-control strategies on the primary five-set benchmark.
- `results/table_s15_pf_representation.csv` contains the two PF
  representations on all three truth domains.
- `results/table_s16_policy_families.csv` contains the rounded values shown in
  SI Table S16.
- `results/family_method_summary.csv`, `per_policy_summary.csv`, and
  `per_evaluation_cell_summary.csv` retain the unrounded policy-factorial
  aggregates used to construct Table S16.

The five exploratory policy families are:

| Family | PF teacher | Bayesian representation | Policy-training domain |
|---|---|---|---|
| F2 | MAP-*K* posterior predictive | effective *K*-protic acid | effective *K*-protic acid |
| F3 | full-posterior predictive | effective *K*-protic acid | effective *K*-protic acid |
| F4 | weighted parameters | effective *K*-protic acid | independent components |
| F5 | weighted parameters | independent components | effective *K*-protic acid |
| F6 | weighted parameters | independent components | independent components |

The original weighted-parameter/effective-*K*-protic-acid family (F1) was not
retrained in this exploratory factorial. Its primary imitation and PPO results
remain in blocks 01-03 and must not be treated as another cell from this run.

## Locked datasets

Each domain contains the same five benchmark seeds (101, 202, 303, 404, and
555), with 3,000 tasks per seed. The three domains are:

- `datasets/sequential_k123/`: the primary effective-*K*-protic-acid
  benchmark;
- `datasets/fixed_two_independent/`: samples containing exactly two
  separate monoprotic components; and
- `datasets/independent_j123/`: samples containing one, two, or
  three separate monoprotic components.

`MANIFEST_SHA256.csv` records newline-normalized SHA-256 values so verification
is stable on both Windows (CRLF checkout) and Linux (LF checkout).

## Source and reproduction scope

The exact recovered historical PF source is already archived at
`../13_SOURCE_ARCHIVES/formal_pf_95_36_reproduction_20260901/`. The additional
factorial source is in `source/`:

- `run_full_pf_factorial.py` runs the two-representation x three-control x
  three-domain online-PF factorial;
- `independent_mixture_pf.py` implements the independent-component PF extension;
- `generate_independent_j123_benchmark.py` generates the broader independent-
  component benchmark;
- `generate_authoritative_family_teacher.py`,
  `run_authoritative_family_imitation.py`, and
  `run_authoritative_family_ppo_seed.py` generate and train F2-F6; and
- `run_locked_policy_factorial.py` evaluates the frozen policies on the 15
  locked benchmark sets.

The archived full runs used 1,000 PF particles. Each exploratory PPO run began
from its family-specific imitation checkpoint, used a 5,000-task training pool,
a 500-task validation set, and 100,000 environment interactions. Evaluation
used deterministic argmax actions. `CHECKPOINT_SHA256.csv` fingerprints the
exact checkpoints used in the archived evaluation; the exploratory checkpoints
are not deployment models and are therefore not duplicated in the public
controller package.

Example PF command from the repository root (choose a worker count suitable for
the machine):

```powershell
python evidence/simulation_numerical_evidence_20260823/18_CONTROLLER_REPRESENTATION_FACTORIAL/source/run_full_pf_factorial.py `
  --formal-source evidence/simulation_numerical_evidence_20260823/13_SOURCE_ARCHIVES/formal_pf_95_36_reproduction_20260901 `
  --datasets-root evidence/simulation_numerical_evidence_20260823/18_CONTROLLER_REPRESENTATION_FACTORIAL/datasets `
  --staging-root evidence/simulation_numerical_evidence_20260823/18_CONTROLLER_REPRESENTATION_FACTORIAL/source `
  --output runs/controller_representation_factorial `
  --workers 8
```

The source runner uses the internal dataset directory names
`sequential_k123`, `fixed_two_independent`, and `independent_j123`, matching the
released directory names. The locked JSONL contents must not be edited.

Run the lightweight integrity and aggregation audit with:

```powershell
python scripts/audit_controller_representation_factorial.py
```

That audit checks all 15 manifest hashes, the archived source hashes, the
published-table values, the 12-of-15 PPO-imitation comparison count, and the
45-of-75 individual training-seed comparison count.
