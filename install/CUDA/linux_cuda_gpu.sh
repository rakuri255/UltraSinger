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

# Remove old virtual environment to ensure clean state (e.g., switching between CPU/CUDA)
if [ -d ".venv" ]; then
    echo "Removing old virtual environment..."
    rm -rf .venv
fi

# Set PyTorch index to CUDA in pyproject.toml
# (uv.toml cannot override named indexes used by [tool.uv.sources])
echo "Configuring PyTorch index for CUDA..."
sed -i 's|whl/cpu|whl/cu128|' pyproject.toml

# Regenerate lockfile with CUDA PyTorch index and sync
echo "Resolving dependencies..."
uv lock
echo "Syncing dependencies..."
uv sync --extra linux

# Protect local CUDA config from being reverted by git operations
# (branch switches, pulls, etc. would otherwise reset to CPU default)
if command -v git &> /dev/null && git rev-parse --is-inside-work-tree &> /dev/null; then
    echo "Protecting CUDA configuration from git resets..."
    git update-index --skip-worktree pyproject.toml
    git update-index --skip-worktree uv.lock
fi

echo "Installation completed successfully!"
echo "To run UltraSinger:"
echo "  source .venv/bin/activate"
echo "  cd src"
echo "  python UltraSinger.py"
