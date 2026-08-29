# Primary locked benchmark source

This archive contains the task model, task generator, environment, checkpoint
evaluation script, and run settings used with the five locked manifests. The
manifests in the primary evidence directory are the authoritative task inputs;
their SHA-256 values match the original matched-evaluation snapshot.

From this directory, `RUN_FORMAL.cmd` runs the archived evaluator against those
locked manifests and the packaged imitation/PPO checkpoints. It writes generated
results to `results_formal/` without modifying the locked evidence files.
