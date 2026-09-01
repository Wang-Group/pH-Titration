# Prioritized Task List

## High priority

- Replace all manuscript, SI, and plotting-notebook simulation metrics with the trusted raw-log values from `data/bayesian.txt`, `data/m_network.txt`, `data/reinforced_network.txt`, and `data/PIDexperiment.txt`.
- Align the methods text with the local code: REINFORCE rather than PPO, 5->256->256->1000 rather than 64-unit hidden layers, and the documented action-volume ranges.
- Add the timing benchmark and explicitly compare decision latency against the 20 s mixing/sensing cycle.
- Weaken or qualify the strongest interpretability claim: the local code supports posterior updating and uncertainty shrinkage, but not robust identification of the true number of species.

## Medium priority

- Use the revised Figure 2 candidate and add the effective mixed-acid titration curves as SI support.
- Decide whether to include the adaptive PID baseline directly in the rebuttal/SI now that the local code reruns cleanly.
- Clarify the controller used in each physical experiment and resolve the mixed-acid Mixture 3 reinforcement-learning step-count mismatch.

## Low priority

- Prepare a direct supervised-on-simulation baseline if the reviewer response needs one beyond the current Bayesian-teacher imitation setup.
- Add a real README, environment file, and script/module entry points before any public code release.
- Manually verify the SI equation/symbol rendering in Word/PDF export, because the extracted text still shows several encoding-sensitive passages.
