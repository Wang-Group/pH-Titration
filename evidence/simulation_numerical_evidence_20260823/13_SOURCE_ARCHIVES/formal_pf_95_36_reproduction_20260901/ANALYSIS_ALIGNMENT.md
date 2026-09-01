# Alignment for PF and learned-policy analyses

Use this recovered source as the reference implementation for the manuscript's
primary PF method.

- Primary PF reproduction: run `bayesian_external_rule_ablation.py` and report
  the `hybrid_full` row on the five locked 3,000-task manifests.
- Posterior-to-control ablation: derive every arm from
  `particle_controllers.JointInferenceController`, keep its task initialization,
  posterior update, action range, external dose shaping, stopping rules, and
  overshoot cap fixed, and replace only the posterior-to-required-volume
  calculation being tested.
- Frozen imitation/PPO comparison: use the existing locked evaluator and
  checkpoints. These policies do not run PF during evaluation, but they must be
  evaluated on the same five manifests and paired with the preserved formal PF
  rows.
- New teacher or representation study: regenerate teacher trajectories from the
  explicitly named PF representation and retain that source identity in the run
  metadata. Do not describe a policy trained from the later deployable
  `controllers.RobustPFController` as distilled from the formal historical PF.

This separation preserves the reported primary results while allowing new
posterior-use and representation experiments to be compared without changing
unrelated controller behavior.
