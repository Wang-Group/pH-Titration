# Performance highlights

All values are mean +/- sample SD across five independent training seeds.

| Metric | Best condition | Value |
|---|---|---:|
| **Highest mean success** | **PPO / imitation** | **88.12 +/- 0.47** |
| **Highest strict success** | **PPO / imitation** | **35.48 +/- 2.49** |
| **Lowest severe-failure rate** | **PPO / imitation** | **6.40 +/- 0.55** |
| **Fewest mean steps** | **PPO / imitation** | **12.63 +/- 0.23** |
| **Lowest final absolute error** | **PPO / imitation** | **0.13 +/- 0.01** |
| **Lowest mean total volume** | **REINFORCE / imitation** | **16.06 +/- 2.25** |

The lowest-volume condition is not the best overall controller: REINFORCE/imitation uses less liquid on average but has materially lower success and higher severe-failure rates. PPO/imitation is the strongest balanced result.

For randomly initialized actors, A2C achieved the highest final mean success (53.00 +/- 6.91%) and the largest mean gain over the untrained random actor (+21.62 percentage points).
