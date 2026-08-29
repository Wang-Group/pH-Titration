# Direction-Only Initialization Raw Data

`direction_only_initialization_raw/` is the task-level archive for the
separate random-initialization ablation. It uses five fixed evaluation task
sets (`901`--`905`) and five model/PF seeds. These rows are not the principal
five-set benchmark and must not be merged with the `01_PRIMARY_5x3000_BENCHMARK`
summary.

The archive contains the locked task JSONL files, per-model task-level result
files, run summaries, model provenance, and paired tests used for the reported
random-actor, PPO-from-random, imitation, and PPO-from-imitation comparisons.
