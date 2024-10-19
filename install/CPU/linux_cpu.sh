#!/bin/bash
cd ..
cd ..
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2