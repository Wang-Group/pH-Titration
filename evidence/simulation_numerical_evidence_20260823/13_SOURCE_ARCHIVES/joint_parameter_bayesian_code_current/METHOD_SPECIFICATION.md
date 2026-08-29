# Frozen comparison specification

The comparison was defined before the external full run.

## Particle-filter variants

- `pf_pka_only_k3`: fixed concentration 0.1 M, fixed K=3, ordered pKa particles.
- `pf_pka_conc_k3`: log-concentration and three ordered pKa values are updated jointly.
- `pf_pka_conc_variable_k`: three independent particle banks for K=1,2,3 jointly infer log-concentration and ordered pKa values. Model probabilities are updated from sequential predictive evidence.

All variants use a total budget of 1,000 particles in the standard and full profiles. The variable-K model splits that total budget across the three model banks.

## Priors and update

- pKa order-statistic prior: independent U(1.5, 9.0) draws followed by sorting.
- concentration prior: log-uniform from 0.02 to 0.30 mol/L.
- likelihood: Student-t on observed minus predicted delta-pH, df=4 and scale=0.20 pH.
- resampling: ESS below 0.5 N within each fixed-K bank.
- regularization: Liu-West h=0.15 on pKa and log-concentration.

## PyMC variants

PyMC uses the same priors and complete fixed-trajectory delta-pH likelihood. `pm.sample_smc` is used because the numerical charge-balance root solver is non-differentiable. Variable K is evaluated by enumerating K=1,2,3 and comparing SMC marginal-likelihood estimates under equal prior model probabilities.

## Primary analyses

1. Five-seed nominal closed-loop control at 0.1 M.
2. Five-seed variable-concentration closed-loop transfer audit.
3. Fixed-trajectory PF parameter and response recovery.
4. Fixed-trajectory PF versus PyMC SMC recovery comparison.

The local +/-0.10 mL response RMSE is the primary control-relevant recovery measure. The 0-33 mL base-titration RMSE is a stricter global-identification measure.
