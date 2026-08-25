@echo off
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File run_with_python.ps1 run_fixed3000.py --output-dir results_smoke_test --design paired --scenarios nominal close_random_actuator --task-count 10 --task-seed 20260724 --train-seeds 101 --eval-seeds 7101 --method-set winner --include-native-bayesian --particles 20 --workers 1 --bootstrap-iterations 200
pause
