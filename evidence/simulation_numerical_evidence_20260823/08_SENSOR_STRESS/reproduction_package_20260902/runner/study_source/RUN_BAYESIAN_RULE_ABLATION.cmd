@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" bayesian_external_rule_ablation.py --output-dir results\bayesian_rule_ablation_standard --seeds 101 202 303 404 555 --tasks-per-seed 3000 --particles 1000 --resume
) else (
  py -3.11 bayesian_external_rule_ablation.py --output-dir results\bayesian_rule_ablation_standard --seeds 101 202 303 404 555 --tasks-per-seed 3000 --particles 1000 --resume
)
pause
