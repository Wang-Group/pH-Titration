# Primary controller comparison

Five locked benchmark sets, 3,000 matched tasks per set; values are mean +/- sample SD across the five set-level summaries.

| Controller | Success (%) | Successful steps | Overshoots/task | Final absolute error (pH) |
|---|---:|---:|---:|---:|
| PF controller | 95.36 +/- 0.59 | 4.84 +/- 0.09 | 3.29 +/- 0.21 | 0.0729 +/- 0.0039 |
| Imitation | 89.17 +/- 0.51 | 7.83 +/- 0.16 | 2.93 +/- 0.19 | 0.0953 +/- 0.0046 |
| PPO | 93.95 +/- 0.63 | 5.20 +/- 0.10 | 3.05 +/- 0.21 | 0.0785 +/- 0.0045 |
| Simple rule | 77.28 +/- 0.41 | 16.11 +/- 0.33 | 1.96 +/- 0.18 | 0.1106 +/- 0.0047 |
| Prespecified PID | 84.59 +/- 0.68 | 17.14 +/- 0.28 | 4.22 +/- 0.07 | 0.2214 +/- 0.0108 |
| Tuned PID | 92.44 +/- 0.58 | 14.75 +/- 0.20 | 2.64 +/- 0.15 | 0.1504 +/- 0.0141 |
