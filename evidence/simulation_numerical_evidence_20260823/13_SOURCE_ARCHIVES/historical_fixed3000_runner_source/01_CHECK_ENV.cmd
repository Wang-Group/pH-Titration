@echo off
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File run_with_python.ps1 validate_fixed_package.py
pause
