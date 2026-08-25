@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_all.ps1" -Mode Quick
set "RUN_EXIT=%ERRORLEVEL%"
echo.
if not "%RUN_EXIT%"=="0" (
  echo Quick check FAILED. Review results_quick\RUN_FAILED.txt and results_quick\run_quick.log.
  pause
  exit /b %RUN_EXIT%
)
echo Quick check completed successfully. Review results_quick\RESULT_SUMMARY.md.
pause
exit /b 0
