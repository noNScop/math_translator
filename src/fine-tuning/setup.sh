#!/bin/bash

module load python/3.11.9-gcc-11.5.0-5l7rvgy
PROJECT="${PROJECT_PATH:-/mnt/storage_6/project_data/pl0925-02}"
SFT_DIR="$PROJECT/gemma-fine-tuning"

export TMPDIR="$PROJECT/pip_tmp"
export PIP_CACHE_DIR="$PROJECT/pip_cache"

mkdir -p "$TMPDIR"
mkdir -p "$PIP_CACHE_DIR"
mkdir -p "$SFT_DIR/logs"

VENV_DIR="$PROJECT/venv"
VENV_ACTIVATE="$VENV_DIR/bin/activate"

echo "Starting setup in directory: $PWD"
echo "Using project directory: $PROJECT"

if [ ! -f "$VENV_ACTIVATE" ]; then
    echo "No venv environment found. Creating a new one..."
    python3 -m venv "$VENV_DIR"
else
    echo "venv environment already exists. Proceeding to update libraries..."
fi

source "$VENV_ACTIVATE"

echo "Installing/Updating libraries (this may take a few minutes)..."
python -m pip install --upgrade pip
python -m pip install -r "$SFT_DIR/requirements.txt"

echo "Setup complete."