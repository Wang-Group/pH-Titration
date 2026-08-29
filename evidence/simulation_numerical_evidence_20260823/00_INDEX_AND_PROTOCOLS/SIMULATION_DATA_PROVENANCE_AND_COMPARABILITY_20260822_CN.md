# Simulation data organization

The package contains independent simulation protocols generated for different control and inference questions. The primary controller comparison uses the matched five-set benchmark. Training-seed stability, initialization, RL algorithm screening, posterior recovery, prior sensitivity, sensor stress, dose-rule ablations, particle scaling, PyMC comparison, and online timing are retained as separate blocks with their own task counts and statistical units.

Results should be compared within the protocol whose task generator, controller interface, training budget, and endpoint definition are shared. Timing values are reported with the timing protocol that produced them.

The current publication timing values come from block 16, where PyMC, PF at
three particle counts, imitation, and PPO used the same 100 locked cases and
the same first post-dose pH input. Earlier values in blocks 10-12 are retained
for provenance and are not pooled with or substituted for block 16.
