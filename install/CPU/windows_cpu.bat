@echo off
setlocal
cd ..
cd ..
py -m venv .venv
SET VenvPythonPath=%CD%\.venv\Scripts\python.exe
call %VenvPythonPath% -m pip install -r requirements.txt
call %VenvPythonPath% -m pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1
endlocal