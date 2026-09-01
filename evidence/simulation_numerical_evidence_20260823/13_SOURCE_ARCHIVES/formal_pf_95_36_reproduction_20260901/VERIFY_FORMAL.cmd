@echo off
setlocal
cd /d "%~dp0"
.venv\Scripts\python.exe verify_reproduction.py runs\formal_5x3000
