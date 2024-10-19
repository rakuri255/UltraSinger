@echo off
cd ..
cd ..
py -3.10 -m venv .venv2
call .venv\Scripts\activate
pip install -r requirements.txt
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2