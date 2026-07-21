# Timing Comparison Benchmark

- Dataset: `E:\GitHub\ph4git\pH-Titration\ph4github\experiment_summary.csv`
- Physical reference delay: `20 s` per dosing step
- Metric definition:
  For Bayesian, each timed controller cycle includes posterior updating after observing the current pH and then selecting the next action.
  For IL/RL, each timed controller cycle includes state-vector assembly, candidate filtering, and neural-network inference.
  For PID and expert-rule baselines, each timed controller cycle includes only the rule/controller computation, not the pH-solver step.

| Controller | Median decision (ms) | Mean decision (ms) | P95 decision (ms) | Median episode controller time (ms) | 20 s / median decision |
|---|---:|---:|---:|---:|---:|
| Adaptive PID | 0.003300 | 0.009151 | 0.007800 | 0.115450 | 6060606.1 |
| Expert rule | 0.003300 | 0.006443 | 0.006900 | 0.087150 | 6060606.1 |
| Reinforcement learning | 1.197650 | 7.927879 | 44.202680 | 93.984350 | 16699.4 |
| Imitation learning | 1.234400 | 7.047621 | 30.544795 | 89.475650 | 16202.2 |
| Bayesian (1000 particles) | 45.078400 | 63.306510 | 167.708750 | 402.965150 | 443.7 |
