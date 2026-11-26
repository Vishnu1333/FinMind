#!/usr/bin/env bash
set -e
PYTHON_VERSION=$(python3 -c 'import sys; print(sys.version_info[0])')
if [ "$PYTHON_VERSION" -ne 3 ]; then
  echo "Python 3 is required"
  exit 1
fi
# Create virtual environment
python3 -m venv backend/venv
source backend/venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "Setup complete. Activate with: source backend/venv/bin/activate"