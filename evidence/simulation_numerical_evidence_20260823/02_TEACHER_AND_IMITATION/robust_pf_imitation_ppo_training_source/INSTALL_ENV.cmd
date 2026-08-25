@echo off
setlocal
cd /d "%~dp0"
py -3.11 -m venv .venv
if errorlevel 1 goto :error
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
if errorlevel 1 goto :error
python -m pip install -r requirements.txt
if errorlevel 1 goto :error
echo.
echo Environment installation completed.
pause
exit /b 0
:error
echo.
echo Environment installation failed. Review the messages above.
pause
exit /b 1
