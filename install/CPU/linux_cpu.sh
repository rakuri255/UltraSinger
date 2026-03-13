#!/bin/bash
set -e

cd "$(dirname "$0")"
cd ../..

# Set link mode to copy to avoid hardlink warnings
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

# Remove old virtual environment to ensure clean state (e.g., switching between CPU/CUDA)
if [ -d ".venv" ]; then
    echo "Removing old virtual environment..."
    rm -rf .venv
fi

# Clear skip-worktree flags if previously set by CUDA install script
# (allows git to manage these files normally again since CPU is the default)
if command -v git &> /dev/null && git rev-parse --is-inside-work-tree &> /dev/null; then
    git update-index --no-skip-worktree pyproject.toml 2>/dev/null || true
    git update-index --no-skip-worktree uv.lock 2>/dev/null || true
fi

# Ensure PyTorch index is set to CPU in pyproject.toml
# (restores default if previously changed by CUDA install script)
sed -i 's|whl/cu[0-9]*|whl/cpu|' pyproject.toml

# Regenerate lockfile with CPU PyTorch index and sync
echo "Resolving dependencies..."
uv lock
echo "Syncing dependencies..."
uv sync --extra linux

echo "Installation completed successfully!"
echo "To run UltraSinger:"
echo "  source .venv/bin/activate"
echo "  cd src"
echo "  python UltraSinger.py"
