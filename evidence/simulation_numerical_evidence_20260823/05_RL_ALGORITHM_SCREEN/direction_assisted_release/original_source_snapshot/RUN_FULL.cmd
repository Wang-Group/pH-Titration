@echo off
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_experiment.ps1" -Mode Full -Device cpu
if errorlevel 1 (
  echo.
  echo The full experiment did not complete. Run this file again to resume completed conditions.
  pause
  exit /b 1
)
echo.
echo Full experiment completed successfully.
pause

