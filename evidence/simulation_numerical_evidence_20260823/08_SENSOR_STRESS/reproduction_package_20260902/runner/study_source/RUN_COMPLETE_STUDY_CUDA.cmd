@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run INSTALL_ENV_CUDA.cmd first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" run_complete_study.py --profile standard --output-dir results\complete_study_standard_v1 --ablation-dir results\bayesian_rule_ablation_standard_v2 --device cuda --workers 8 --resume
pause
