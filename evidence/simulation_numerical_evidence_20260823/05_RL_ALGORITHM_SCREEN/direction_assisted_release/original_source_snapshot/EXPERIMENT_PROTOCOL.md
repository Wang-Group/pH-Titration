# Direction-assisted volume-policy comparison protocol

## Control allocation

- State: current measured pH, target pH, measured pH change, current-minus-target pH error, and last requested volume.
- Neural action: 1,000 volume classes from 0.01 to 10.00 mL in 0.01 mL increments.
- External direction rule: add base when `measured_pH < target_pH`; otherwise add acid.
- Titrant concentration: fixed at 0.1 M.
- No overshoot-based action masking or maximum-volume reduction.
- No automatic switch to a dilute titrant.
- Overshoot remains an outcome and reward signal, but does not change the available actions.
- Stop at true absolute pH error no greater than 0.10 or after 50 additions.

This protocol matches the functional division of the deployed 1,000-output neural policy: the neural network selects volume, while a common deterministic rule selects reagent direction.

## Comparisons

- Algorithms: PPO, A2C, and REINFORCE.
- Initializations: archived 1,000-class imitation policy or explicit independent random actor weights.
- Training seeds: 101, 202, 303, 404, and 555.
- Random actor seeds: 9,000,101; 9,000,202; 9,000,303; 9,000,404; and 9,000,555.
- For a given seed, the same initial actor is used across algorithms within each initialization condition.
- A2C and PPO critics are paired by constructing the complete model from the same training seed before replacing or explicitly resetting the actor.

## Training and evaluation

- Approximately 25,000 environment interactions per training run.
- A 5,000-task training pool per seed.
- A separate 1,000-task held-out evaluation set per seed.
- Evaluation at zero training and approximately every 5,000 interactions.
- Adam learning rate 1e-4 and discount factor 0.99.
- PPO uses 2,048-interaction batches, four epochs, and a 0.2 clip ratio.
- Frozen evaluation uses deterministic argmax over all 1,000 volume actions.

## Statistical reporting

- Mean and sample SD across five independent training seeds.
- Paired random-minus-imitation differences for each algorithm.
- Pairwise algorithm comparisons within each initialization condition.
- Seed-level bootstrap intervals and exact sign-flip tests.
- Task-level exact McNemar tests with Holm correction, interpreted as conditional on the frozen models rather than as independent training replicates.

