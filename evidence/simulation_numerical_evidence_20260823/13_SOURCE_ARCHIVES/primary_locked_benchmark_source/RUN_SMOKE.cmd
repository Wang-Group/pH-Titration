@echo off
setlocal
cd /d "%~dp0"
py -3.11 run_matched_evaluation.py --tasks-per-seed 20 --output-dir results_smoke --device cpu
if errorlevel 1 exit /b %errorlevel%
echo Smoke evaluation completed.
endlocal
