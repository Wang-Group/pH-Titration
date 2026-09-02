# Imitation Dataset Audit and Selection Standard

## Archived dataset audit

The archived files in `E:\GitHub\ph4git\pH-Titration` were evaluated using the same action filter as the submitted training script, which retained action categories 0 and 2.

| Split | Raw records | Effective records | Low-to-high pH | High-to-low pH | Low-to-high share |
|---|---:|---:|---:|---:|---:|
| Training | 48,107 | 47,810 | 34,249 | 13,558 | 71.64% |
| Validation | 10,308 | 10,247 | 7,250 | 2,997 | 70.75% |
| Test | 10,310 | 10,240 | 7,379 | 2,861 | 72.06% |

Three training records had current pH exactly equal to target pH. The archived training procedure shuffled these records but did not correct the direction imbalance or screen trajectories by closed-loop quality.

## New minimum size

The standard profile requires at least 60,000 unique quality-screened training states and 12,000 validation states. Each direction must independently contribute at least half of the requested total before generation can stop. The full profile raises these thresholds to 120,000 and 24,000 states.

## Automatic teacher selection

The new `pf_pka_conc_variable_k` controller generates all labels. Teacher states and all downstream neural-policy environments share a 0.01 pH sensor resolution so that supervised states and closed-loop states use the same observation protocol. A trajectory is eligible only if it reaches an absolute endpoint error no greater than 0.10 pH, uses no more than 30 steps, crosses the target no more than 8 times, adds no more than 20 mL beyond the oracle volume, contains only finite values, and has a quality score of at least 0.30.

The quality score is fixed before training and combines endpoint accuracy (weight 0.40), step efficiency (0.25), target-crossing efficiency (0.15), and dose overhead (0.20). Near-duplicate state-action records are identified after rounding the five state features to 0.001; only the higher-quality record is kept. Quality also enters the supervised sample weight.

Generation continues in deterministic independent batches until the total state target and both direction targets are satisfied. Marginal coverage is required for current and target pH bins, pH-error bin, dose bin, concentration, initial volume, acid type, pair count, pKa family, and task difficulty. Joint coverage is also required for direction by error bin, direction by pKa family, and direction by acid type by difficulty. This prevents a high overall sample count from hiding missing combinations. Each round is written to `train_quality_generation_audit.csv` or `validation_quality_generation_audit.csv`, and all final marginal and joint counts are exported to `train_diversity_audit.csv` or `validation_diversity_audit.csv`.

## Training-time balance

Even after equal task counts, one direction can yield longer trajectories. The imitation data loader therefore constructs each epoch with exactly equal acid- and base-direction state counts. The larger direction is retained once, and the smaller direction is reproducibly sampled with replacement. Because generation already enforces a large unique-state minimum for both directions, this balancing does not rely on a small minority subset.
