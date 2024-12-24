@echo off
cd ..
cd ..
py -3.10 -m venv .venv
SET VenvPythonPath=%CD%\.venv\Scripts\python.exe
call %VenvPythonPath% -m pip install -r requirements.txt
call %VenvPythonPath% -m pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
