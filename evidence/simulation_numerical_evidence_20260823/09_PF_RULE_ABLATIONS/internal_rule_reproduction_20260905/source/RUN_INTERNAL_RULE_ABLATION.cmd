@echo off
setlocal
cd /d "%~dp0"
set PY=.venv\Scripts\python.exe
if not exist "%PY%" set PY=py -3.11
%PY% pf_internal_rule_ablation.py --output-dir results\new_pf_internal_rule_ablation --seeds 101 202 303 404 555 --tasks-per-seed 300 --particles 1000 --workers 8
pause
