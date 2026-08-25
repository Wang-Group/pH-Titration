# Protocol and statistical units

| Evidence block | Replication unit | Tasks per unit | Statistical summary |
|---|---|---:|---|
| Primary manuscript benchmark | Five independent task-set seeds | 3,000 | Sample SD across five task-set means |
| Direction-only random-initialization ablation | Five model/PF seeds and five evaluation seeds | 3,000 per evaluation seed | Sample SD across model/PF-seed aggregates |
| Posterior recovery | Five independent task seeds | 300 | Sample SD across seed-level means |
| PF sensor stress | Five independent task seeds | 1,000 per condition and method | Sample SD across seed-level means |
| PyMC comparison | Five seeds, three tasks per seed | 15 total | Pooled medians and percentages |
| New-PF particle scaling | One common task set per particle count | 100 | Task medians for timing |
| Teacher and imitation | Teacher-generation and policy-training seeds | See run configuration | Teacher-fidelity and closed-loop summaries |

Task-set seeds, training seeds, PF seeds, and evaluation seeds are distinct sources of variation. They are kept separate in the study index.
