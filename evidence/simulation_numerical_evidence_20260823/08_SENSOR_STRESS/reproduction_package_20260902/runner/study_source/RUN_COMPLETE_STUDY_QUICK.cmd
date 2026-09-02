@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" run_complete_study.py --profile quick --output-dir results\complete_study_quick_v1 --device auto --workers 2 --resume
) else (
  py -3.11 run_complete_study.py --profile quick --output-dir results\complete_study_quick_v1 --device cpu --workers 2 --resume
)
pause
