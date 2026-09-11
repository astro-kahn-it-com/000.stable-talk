@echo off
setlocal
cd /d "%~dp0"

set PYTHON=%~dp0python_embeded\python.exe

echo ========================================================
echo Launching Stable Audio 3 Standalone Inference Engine...
echo ========================================================

%PYTHON% generate.py

echo.
pause