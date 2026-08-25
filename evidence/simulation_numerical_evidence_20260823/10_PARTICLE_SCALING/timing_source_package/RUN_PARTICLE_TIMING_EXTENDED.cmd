@echo off
setlocal
cd /d "%~dp0"
echo Running extended particle-count timing benchmark from:
echo %CD%
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_particle_timing.ps1" -Notebook "%~dp0particle_count_timing_benchmark_extended_20260806.ipynb" -ExecutedNotebook "particle_count_timing_benchmark_extended_executed.ipynb" -CompletionMarker "RUN_COMPLETE_EXTENDED.txt"
if errorlevel 1 (
  echo.
  echo Extended benchmark failed. Check particle_count_timing_run.log and RUN_FAILED.txt.
  pause
  exit /b 1
)
echo.
echo Extended benchmark finished. Results are in particle_count_timing_results_extended_20260806.
pause
