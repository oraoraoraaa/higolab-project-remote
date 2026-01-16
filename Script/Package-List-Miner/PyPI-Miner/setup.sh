#!/bin/bash

# PyPI Miner Setup Script

echo "Setting up PyPI Miner environment..."

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Setup complete!"
echo ""
echo "To run the miner:"
echo "  source venv/bin/activate"
echo "  python mine_pypi.py"
echo ""
