#!/bin/bash
# STT Fine-Tune Evaluation GUI Launcher
# Activates the virtual environment and launches the GUI

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$SCRIPT_DIR/.venv"

# Check if venv exists
if [ ! -d "$VENV_PATH" ]; then
    echo "Error: Virtual environment not found at $VENV_PATH"
    echo "Create it with: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate venv and run GUI
source "$VENV_PATH/bin/activate"

echo "=========================================="
echo "STT Fine-Tune Evaluation GUI"
echo "=========================================="
echo "Virtual environment: $VENV_PATH"
echo "Python: $(which python)"
echo "=========================================="

# Run the GUI
python "$SCRIPT_DIR/gui_evaluate.py" "$@"
