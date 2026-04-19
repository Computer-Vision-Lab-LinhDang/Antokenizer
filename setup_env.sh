#!/usr/bin/env bash
# Install MAVT dependencies using uv
set -euo pipefail

if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "Creating virtual environment and installing dependencies..."
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"

echo ""
echo "Environment ready. Activate with: source .venv/bin/activate"
