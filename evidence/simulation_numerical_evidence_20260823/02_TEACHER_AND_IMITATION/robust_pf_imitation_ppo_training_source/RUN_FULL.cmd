@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" run_pipeline.py --profile full --output-dir results\full_run --device auto --resume
) else (
  py -3.11 run_pipeline.py --profile full --output-dir results\full_run --device auto --resume
)
pause
