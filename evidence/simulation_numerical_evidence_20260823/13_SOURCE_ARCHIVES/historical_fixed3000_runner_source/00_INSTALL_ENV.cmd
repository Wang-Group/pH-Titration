@echo off
cd /d "%~dp0"
py -3.11 -m venv .venv
if errorlevel 1 py -3.12 -m venv .venv
if errorlevel 1 py -3.10 -m venv .venv
if errorlevel 1 py -3 -m venv .venv
if errorlevel 1 goto :error
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto :error
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto :error
echo Environment installation completed.
pause
exit /b 0
:error
echo Environment installation failed. Install Python 3.10-3.12 and run this file again.
pause
exit /b 1
