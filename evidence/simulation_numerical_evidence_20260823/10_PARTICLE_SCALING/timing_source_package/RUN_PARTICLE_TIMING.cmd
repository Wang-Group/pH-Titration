@echo off
setlocal
cd /d "%~dp0"
echo Running particle-count timing benchmark from:
echo %CD%
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_particle_timing.ps1"
if errorlevel 1 (
  echo.
  echo Benchmark failed. Check results/particle_count_timing_run.log and RUN_FAILED.txt.
  pause
  exit /b 1
)
echo.
echo Finished. Results are in particle_count_timing_results.
pause
