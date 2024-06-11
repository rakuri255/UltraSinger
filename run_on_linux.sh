#!/bin/bash
gnome-terminal -- bash -c ".venv\Scripts\activate; cd src; py UltraSinger.py -h; exec bash"