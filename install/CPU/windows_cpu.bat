@echo off
cd ..
cd ..
py -3.10 -m venv .venv
call .venv\Scripts\activate
pip install -r requirements.txt