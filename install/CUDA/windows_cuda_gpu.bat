@echo off
cd ..
cd ..
py -3.10 -m venv .venv
call .venv\Scripts\activate
pip install -r requirements.txt
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2+cu117 --index-url https://download.pytorch.org/whl/cu117