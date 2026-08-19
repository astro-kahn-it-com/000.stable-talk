@echo off
set PYTHON=%~dp0python_embeded\python.exe
%PYTHON% -m pip install git+https://github.com/Stability-AI/stable-audio-3.git torchaudio soundfile --prefer-binary
%PYTHON% generate.py
pause
