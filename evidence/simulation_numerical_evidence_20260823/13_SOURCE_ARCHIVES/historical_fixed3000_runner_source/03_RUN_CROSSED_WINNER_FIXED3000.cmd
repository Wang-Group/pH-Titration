@echo off
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File run_with_python.ps1 run_fixed3000.py --output-dir results_fixed3000_crossed_winner --design cross --scenarios close_random_actuator --task-count 3000 --task-seed 20260724 --method-set winner --particles 500 --workers 5 --bootstrap-iterations 20000
pause
