@echo off
setlocal
cd /d "%~dp0"
call INSTALL_ENV.cmd --no-pause
if errorlevel 1 exit /b 1
.venv\Scripts\python.exe run_all.py --profile standard
if errorlevel 1 pause
