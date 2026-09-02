@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" bayesian_external_rule_ablation.py --output-dir results\bayesian_rule_ablation_quick --seeds 101 --tasks-per-seed 50 --particles 100 --resume
) else (
  py -3.11 bayesian_external_rule_ablation.py --output-dir results\bayesian_rule_ablation_quick --seeds 101 --tasks-per-seed 50 --particles 100 --resume
)
pause
