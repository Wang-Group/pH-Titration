@echo off
setlocal
cd /d "%~dp0"
set "EVIDENCE_ROOT=%~dp0..\.."
set "FORMAL_DIR=%EVIDENCE_ROOT%\01_PRIMARY_5x3000_BENCHMARK\formal_matched_evaluation"
set "CHECKPOINT_DIR=%EVIDENCE_ROOT%\02_TEACHER_AND_IMITATION\checkpoints"
py -3.11 run_matched_evaluation.py --package-dir "%FORMAL_DIR%" --output-dir "%~dp0results_formal" --device cpu --imitation-checkpoint "%CHECKPOINT_DIR%\imitation_best.pth" --ppo-checkpoint "%CHECKPOINT_DIR%\principal_ppo_seed_303.pth"
if errorlevel 1 exit /b %errorlevel%
echo Formal matched evaluation completed.
endlocal
