# Baseline runner

This source archive contains the original baseline implementation and runner
used to generate the locked five-set simple-rule and PID results. The runner
uses the five authoritative task manifests under
`01_PRIMARY_5x3000_BENCHMARK/formal_matched_evaluation/tasks/` and writes
task-level output for the prespecified PID, selected/tuned PID, and simple
rule controllers.

The released output is stored in
`01_PRIMARY_5x3000_BENCHMARK/formal_matched_evaluation/rule_baseline_replay/`.
The runner calls the selected PID `selected_pid`; the release uses the
manuscript label `tuned_pid` in the CSV files.

Reported five-set means reproduced by the archived output are:

| Method | Success (%) | Successful steps | Final absolute error (pH) |
|---|---:|---:|---:|
| Simple rule | 77.28 +/- 0.41 | 16.11 +/- 0.33 | 0.1106 +/- 0.0047 |
| Prespecified PID | 84.59 +/- 0.68 | 17.14 +/- 0.28 | 0.2214 +/- 0.0108 |
| Tuned PID | 92.44 +/- 0.58 | 14.75 +/- 0.20 | 0.1504 +/- 0.0141 |

The five task manifests are byte-identical to the manifests in the original
matched-evaluation snapshot and the current evidence package. The archived
baseline output is retained as the source of the reported baseline numbers;
the current deployment controllers remain in `controllers/`.
