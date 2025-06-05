@echo off
setlocal
cd ..
cd ..
py -3.10 -m venv .venv
SET VenvPythonPath=%CD%\.venv\Scripts\python.exe
call %VenvPythonPath% -m pip install -r requirements-windows.txt
call %VenvPythonPath% -m pip install torch==2.3.1+cu118 torchvision==0.18.1+cu118 torchaudio==2.3.1+cu118 --index-url https://download.pytorch.org/whl/cu118
endlocal