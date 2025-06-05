#!/bin/bash
cd ..
cd ..
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements-linux.txt
pip install torch==2.3.1+cu118 torchvision==0.18.1+cu118 torchaudio==2.3.1+cu118 --index-url https://download.pytorch.org/whl/cu118
