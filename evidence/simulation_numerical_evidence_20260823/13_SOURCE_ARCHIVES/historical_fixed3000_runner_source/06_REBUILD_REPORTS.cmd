@echo off
cd /d "%~dp0"
if exist results_fixed3000_primary\settings.json powershell -NoProfile -ExecutionPolicy Bypass -File run_with_python.ps1 analyze_fixed3000.py results_fixed3000_primary
if exist results_fixed3000_crossed_winner\settings.json powershell -NoProfile -ExecutionPolicy Bypass -File run_with_python.ps1 analyze_fixed3000.py results_fixed3000_crossed_winner
if exist results_fixed3000_extended\settings.json powershell -NoProfile -ExecutionPolicy Bypass -File run_with_python.ps1 analyze_fixed3000.py results_fixed3000_extended
pause
