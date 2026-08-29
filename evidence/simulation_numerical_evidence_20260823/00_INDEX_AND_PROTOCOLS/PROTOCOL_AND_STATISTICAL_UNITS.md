# Protocol and statistical units

| Evidence block | Replication unit | Tasks per unit | Statistical summary |
|---|---|---:|---|
| Primary manuscript benchmark | Five independent task-set seeds | 3,000 | Sample SD across five task-set means |
| Direction-only random-initialization ablation | Five model/PF seeds and five evaluation seeds | 3,000 per evaluation seed | Sample SD across model/PF-seed aggregates |
| Posterior recovery | Five independent task seeds | 300 | Sample SD across seed-level means |
| PF sensor stress | Five independent task seeds | 1,000 per condition and method | Sample SD across seed-level means |
| Earlier PyMC comparison (block 11) | Five seeds, three tasks per seed | 15 total | Pooled medians and percentages |
| Earlier new-PF particle scaling (block 10) | One common task set per particle count | 100 | Task medians for timing |
| Matched single-step timing and one-observation recovery (block 16) | The same 20 task IDs from each of five locked benchmark seeds | 100 per method | Task-level medians and paired PF/PyMC tests |
| PF complete-trajectory timing and outcomes (block 17) | The same 20 task IDs from each of five locked benchmark seeds | 100 tasks per particle count; 592, 589, and 577 recorded cycles | Task-level outcome percentages and pooled step-level timing distributions |
| Teacher and imitation | Teacher-generation and policy-training seeds | See run configuration | Teacher-fidelity and closed-loop summaries |

Task-set seeds, training seeds, PF seeds, and evaluation seeds are distinct sources of variation. They are kept separate in the study index.
