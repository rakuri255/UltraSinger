#!/bin/bash
# Activate virtual environment and run UltraSinger on macOS

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Check if .venv exists
if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment not found at .venv"
    echo "Please run the installation script first:"
    echo "  - install/CPU/macos_cpu.sh"
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Navigate to src and start UltraSinger
cd src
echo "Starting UltraSinger..."
python UltraSinger.py
