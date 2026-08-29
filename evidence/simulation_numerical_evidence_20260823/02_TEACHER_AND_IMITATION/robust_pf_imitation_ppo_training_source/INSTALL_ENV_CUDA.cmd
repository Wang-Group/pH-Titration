@echo off
setlocal
cd /d "%~dp0"
py -3.11 -m venv .venv
if errorlevel 1 goto :error
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
if errorlevel 1 goto :error
python -m pip install -r requirements-core.txt
if errorlevel 1 goto :error
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto :error
python -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available to PyTorch'; print(torch.cuda.get_device_name(0))"
if errorlevel 1 goto :error
echo.
echo CUDA environment installation completed.
pause
exit /b 0
:error
echo.
echo CUDA environment installation or validation failed.
pause
exit /b 1
