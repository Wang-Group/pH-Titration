@echo off
setlocal
cd /d "%~dp0"
.venv\Scripts\python.exe -u bayesian_external_rule_ablation.py --output-dir runs\formal_5x3000 --seeds 101 202 303 404 555 --tasks-per-seed 3000 --particles 1000 --workers 8 --resume
