@echo off
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_experiment.ps1" -Mode Quick -Device cpu
if errorlevel 1 (
  echo.
  echo The quick test did not complete. Review the error above.
  pause
  exit /b 1
)
echo.
echo Quick test completed successfully.
pause

