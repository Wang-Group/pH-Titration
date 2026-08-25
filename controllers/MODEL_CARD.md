# Deployment Model Card

## Models

| Artifact | Purpose | Backend |
|---|---|---|
| `models/ppo_seed_303.pth` | Frozen PPO actor checkpoint | PyTorch |
| `models/ppo_seed_303_numpy.npz` | Exported copy of the same actor | NumPy |

The actor maps a five-feature state to 1,000 discrete volume classes. Class `0` is 0.01 mL and class `999` is 10.00 mL. Deployment uses deterministic `argmax` inference.

## Training scope

The checkpoint belongs to the `pH-control` protocol family, version `2026.08`.
It was trained under `training_environment_strict`, which uses rounded 0.01-pH
observations and a strict `< 0.10` stopping and true-success threshold. The
actor selects volume only under an external acid/base direction rule. It was
initialized from a PF-distilled imitation policy and refined with PPO.
Deployment uses the `deployment_api_strict` profile and applies the shared
persistent post-overshoot cap after a target crossing or an increase in
absolute observed-pH error. Formal published evaluation is separately labeled
`formal_evaluation` and uses the inclusive `<= 0.10` endpoint definition.

## Operating scope

The checkpoint is intended for the documented titrant concentration, sensor definition, action range, chemistry model, and state representation. Changes to these conditions require new training and independent evaluation. The model is research software and should be integrated with an independent hardware safety layer.

## Numerical regularization

After Liu--West rejuvenation, proposed pKa values are sorted and clipped to
1--10 before the next equilibrium calculation. This broad envelope prevents
extreme proposals from causing unstable charge-balance predictions while
retaining the nominal 1.5--9.0 initialization range. It is a numerical
stability safeguard, not a species-specific prior or a claim about universal
thermodynamic pKa limits.

## Integrity

The verification script prints SHA-256 values for both packaged artifacts. The PyTorch deployment wrapper also verifies the selected checkpoint and actor tensor hashes before use.
