@echo off
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_experiment.ps1" -Mode Full -Device cuda
if errorlevel 1 (
  echo.
  echo The CUDA run did not complete. Check the CUDA-enabled PyTorch installation or use RUN_FULL.cmd.
  pause
  exit /b 1
)
echo.
echo Full CUDA experiment completed successfully.
pause

