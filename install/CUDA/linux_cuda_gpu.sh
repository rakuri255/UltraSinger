#!/bin/bash
set -e

cd "$(dirname "$0")"
cd ../..

# Set link mode to copy to avoid hardlink warnings on different filesystems
export UV_LINK_MODE=copy

# Install uv if not already installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Update PATH for current session
    export PATH="$HOME/.local/bin:$PATH"
fi

# Verify uv is available
if ! command -v uv &> /dev/null; then
    echo "Error: uv could not be found or installed"
    echo "Please ensure your shell PATH includes ~/.local/bin"
    exit 1
fi

echo "uv version:"
uv --version

# Sync dependencies with uv (GPU version with CUDA)
echo "Installing PyTorch with CUDA support..."
uv pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128 --force-reinstall

echo "Installation completed successfully!"
echo "To run UltraSinger:"
echo "  source .venv/bin/activate"
echo "  cd src"
echo "  python UltraSinger.py"
