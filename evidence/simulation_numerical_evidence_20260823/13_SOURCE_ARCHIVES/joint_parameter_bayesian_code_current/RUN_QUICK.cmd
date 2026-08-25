@echo off
setlocal
cd /d "%~dp0"
if exist .venv\Scripts\python.exe (set PY=.venv\Scripts\python.exe) else (set PY=py -3.11)
%PY% run_all.py --profile quick
if errorlevel 1 pause
