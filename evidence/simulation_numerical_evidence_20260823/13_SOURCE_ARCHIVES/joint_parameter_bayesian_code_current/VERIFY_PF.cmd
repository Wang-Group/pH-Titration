@echo off
setlocal
cd /d "%~dp0"
if exist .venv\Scripts\python.exe (set PY=.venv\Scripts\python.exe) else (set PY=py -3.11)
%PY% -m compileall -q .
if errorlevel 1 exit /b 1
%PY% run_pf_multiseed_control.py --seeds 101 --tasks-per-seed 1 --particles 60 --workers 1 --output-dir results\verification_control
if errorlevel 1 exit /b 1
%PY% run_pf_curve_recovery.py --seeds 101 --tasks-per-seed 1 --particles 60 --workers 1 --output-dir results\verification_curve
if errorlevel 1 exit /b 1
echo Particle-filter verification complete.
pause
