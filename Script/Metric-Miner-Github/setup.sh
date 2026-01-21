#!/bin/bash

echo "Setting up GitHub Metrics Miner..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Setup complete!"
echo ""
echo "To use the script:"
echo "  1. Activate the virtual environment: source venv/bin/activate"
echo "  2. Run the script: python mine_metrics_github.py"
echo "  3. Deactivate when done: deactivate"
