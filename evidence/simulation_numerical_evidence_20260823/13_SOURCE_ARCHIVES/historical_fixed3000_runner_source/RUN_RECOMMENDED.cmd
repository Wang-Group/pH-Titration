@echo off
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File run_with_python.ps1 validate_fixed_package.py
if errorlevel 1 goto :error
powershell -NoProfile -ExecutionPolicy Bypass -File run_with_python.ps1 run_fixed3000.py --output-dir results_fixed3000_primary --design paired --scenarios nominal close_random_actuator --task-count 3000 --task-seed 20260724 --method-set all --include-native-bayesian --particles 500 --workers 5 --bootstrap-iterations 10000
if errorlevel 1 goto :error
powershell -NoProfile -ExecutionPolicy Bypass -File run_with_python.ps1 run_fixed3000.py --output-dir results_fixed3000_crossed_winner --design cross --scenarios close_random_actuator --task-count 3000 --task-seed 20260724 --method-set winner --particles 500 --workers 5 --bootstrap-iterations 20000
if errorlevel 1 goto :error
echo Recommended confirmatory runs completed.
pause
exit /b 0
:error
echo A step failed. Run the same file again after fixing the reported issue; completed shards will be reused.
pause
exit /b 1
