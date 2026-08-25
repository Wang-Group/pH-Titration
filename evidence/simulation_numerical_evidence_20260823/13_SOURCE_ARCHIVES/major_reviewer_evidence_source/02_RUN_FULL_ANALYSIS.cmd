@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_all.ps1" -Mode Full
set "RUN_EXIT=%ERRORLEVEL%"
echo.
if not "%RUN_EXIT%"=="0" (
  echo Full analysis FAILED. Review results_full\RUN_FAILED.txt and results_full\run_full.log.
  pause
  exit /b %RUN_EXIT%
)
echo Full analysis completed successfully. Review results_full\RESULT_SUMMARY.md.
pause
exit /b 0
