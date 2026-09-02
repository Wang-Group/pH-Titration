@echo off
setlocal
cd /d "%~dp0"
set PY=.venv\Scripts\python.exe
if not exist "%PY%" set PY=py -3.11
%PY% fixed_controller_stress_benchmark.py --output-dir results\fixed_pf_ppo_stress --seeds 101 202 303 404 555 --tasks-per-seed 1000 --device cpu --ppo-backend numpy --workers 16 --resume
pause
