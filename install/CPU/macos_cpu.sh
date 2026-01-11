#!/bin/bash
cd ..
cd ..
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements-macos.txt
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1
