@echo off
setlocal
cd /d "%~dp0"
set PY=.venv\Scripts\python.exe
if not exist "%PY%" set PY=py -3.11
%PY% pf_local_response_diagnostics.py --output-dir results\new_pf_local_response --seeds 101 202 303 404 555 --tasks-per-seed 300 --particles 1000 --workers 8
pause
