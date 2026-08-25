# Physical-experiment data

This directory contains the experimental records used for the four physical
studies in the revised manuscript. Volumes are reported in millilitres, pH
values are the recorded electrode readings, dry masses are reported in
milligrams, wavelengths are in nanometres, and absorbance values are
dimensionless.

## Directory index

| Directory | Files and purpose |
|---|---|
| `mixed_acid/` | Sequential pH-adjustment logs, independently measured titration curves, and PF posterior-plot data |
| `wastewater/` | Sequential pH-adjustment logs for the PF and validation-selected PPO experiments |
| `casein/` | Sequential pH-adjustment logs and separated-product dry masses for PF and PID runs |
| `cu_ssa/` | Sequential pH-adjustment logs, Job-method composition data, UV–Vis spectra, and the Job-analysis reproduction script |

The common pH-adjustment tables identify the experiment, chemical system, run
and sample labels, controller or operator, sequential point, experimental
phase, reagent, delivered dose, measured pH, and target pH. A row with
`phase=initial` records the starting measurement and therefore has a zero
delivered dose.

Run the Cu–SSA analysis from the repository root with:

```powershell
python physical_experiments/cu_ssa/reproduce_job_analysis.py
```

The physical-control notebook is not part of this release. The CSV files are
the released experimental records; the Cu–SSA directory additionally contains
the analysis script needed to reproduce the reported continuous-variation fit.
