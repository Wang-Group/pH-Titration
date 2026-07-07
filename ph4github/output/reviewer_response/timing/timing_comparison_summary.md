# Timing Comparison Benchmark

- Dataset: `E:\GitHub\ph4git\pH-Titration\ph4github\experiment_summary.csv`
- Physical reference delay: `20 s` per dosing step
- Metric definition:
  For Bayesian, each timed controller cycle includes posterior updating after observing the current pH and then selecting the next action.
  For IL/RL, each timed controller cycle includes state-vector assembly, candidate filtering, and neural-network inference.
  For PID and expert-rule baselines, each timed controller cycle includes only the rule/controller computation, not the pH-solver step.

| Controller | Median decision (ms) | Mean decision (ms) | P95 decision (ms) | Median episode controller time (ms) | 20 s / median decision |
|---|---:|---:|---:|---:|---:|
| Expert rule | 0.004700 | 0.007450 | 0.010945 | 0.160250 | 4255319.1 |
| Adaptive PID | 0.004900 | 0.008003 | 0.013530 | 0.181600 | 4081632.7 |
| Imitation learning | 8.314750 | 14.885236 | 47.294115 | 143.872250 | 2405.4 |
| Reinforcement learning | 13.444300 | 13.408905 | 32.602365 | 129.380650 | 1487.6 |
| Bayesian (1000 particles) | 100.493800 | 150.397935 | 425.888650 | 889.726550 | 199.0 |
