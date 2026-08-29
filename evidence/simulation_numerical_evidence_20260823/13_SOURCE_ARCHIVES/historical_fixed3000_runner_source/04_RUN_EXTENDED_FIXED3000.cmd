@echo off
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File run_with_python.ps1 run_fixed3000.py --output-dir results_fixed3000_extended --design paired --scenarios high_conc_under large_volume_drift close_pka out_of_range tetra_noise noise_010 partial_response partial_bias --task-count 3000 --task-seed 20260724 --method-set core --particles 500 --workers 5 --bootstrap-iterations 10000
pause
