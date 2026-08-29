# Imitation Dataset Audit and Selection Standard

## Dataset composition

The standard profile requires at least 60,000 unique quality-screened training states and 12,000 validation states. Each direction must independently contribute at least half of the requested total before generation can stop. The full profile raises these thresholds to 120,000 and 24,000 states.

## Automatic teacher selection

The new `pf_pka_conc_variable_k` controller generates all labels. A trajectory is eligible only if it reaches an absolute endpoint error no greater than 0.10 pH, uses no more than 30 steps, crosses the target no more than 8 times, adds no more than 20 mL beyond the oracle volume, contains only finite values, and has a quality score of at least 0.30.

The quality score is fixed before training and combines endpoint accuracy (weight 0.40), step efficiency (0.25), target-crossing efficiency (0.15), and dose overhead (0.20). Near-duplicate state-action records are identified after rounding the five state features to 0.001; only the higher-quality record is kept. Quality also enters the supervised sample weight.

Generation continues in deterministic independent batches until the total state target and both direction targets are satisfied. Coverage is checked for current and target pH bins, pH-error bin, dose bin, concentration, initial volume, acid type, pair count, pKa family, and task difficulty. Joint coverage is also checked for direction by error bin, direction by pKa family, and direction by acid type by difficulty. Each round is written to `train_quality_generation_audit.csv` or `validation_quality_generation_audit.csv`, and all final marginal and joint counts are exported to `train_diversity_audit.csv` or `validation_diversity_audit.csv`.

## Training-time balance

Even after equal task counts, one direction can yield longer trajectories. The imitation data loader therefore constructs each epoch with exactly equal acid- and base-direction state counts. The larger direction is retained once, and the smaller direction is reproducibly resampled. Because generation already enforces a large unique-state minimum for both directions, this balancing does not rely on a small subset.
