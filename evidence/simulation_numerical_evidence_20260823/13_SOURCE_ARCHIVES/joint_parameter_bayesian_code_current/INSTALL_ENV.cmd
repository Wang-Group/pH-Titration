@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  py -3.11 -m venv .venv
  if errorlevel 1 exit /b 1
)
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 exit /b 1
.venv\Scripts\python.exe -m pip install --only-binary=:all: -r requirements_tested.txt
if errorlevel 1 exit /b 1
echo Environment installation complete.
if /I "%~1"=="--no-pause" exit /b 0
pause
