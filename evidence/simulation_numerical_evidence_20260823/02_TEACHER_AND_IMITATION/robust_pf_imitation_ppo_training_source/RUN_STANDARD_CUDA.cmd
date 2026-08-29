@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" run_pipeline.py --profile standard --output-dir results\standard_run --device cuda --resume
) else (
  py -3.11 run_pipeline.py --profile standard --output-dir results\standard_run --device cuda --resume
)
pause
