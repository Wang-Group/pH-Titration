@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" controller_package_self_test.py
) else (
  python controller_package_self_test.py
)
endlocal
