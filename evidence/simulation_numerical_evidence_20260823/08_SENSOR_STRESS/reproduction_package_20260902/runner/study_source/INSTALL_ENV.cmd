@echo off
setlocal
cd /d "%~dp0"

py -3.11 -m venv .venv
if errorlevel 1 goto :python_error

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
if errorlevel 1 goto :install_error
python -m pip install -r requirements.txt
if errorlevel 1 goto :install_error

echo.
echo Environment installation completed.
echo Next run RUN_BAYESIAN_RULE_ABLATION_QUICK.cmd.
pause
exit /b 0

:python_error
echo.
echo Python 3.11 was not found. Install 64-bit Python 3.11 and enable the Python launcher.
pause
exit /b 1

:install_error
echo.
echo Dependency installation failed. Review the messages above.
pause
exit /b 1
