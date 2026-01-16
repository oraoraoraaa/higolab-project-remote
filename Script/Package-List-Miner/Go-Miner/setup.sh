#!/bin/bash

echo "Setting up Go Package Miner..."

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Setup complete!"
echo "To run the miner:"
echo "  source venv/bin/activate"
echo "  python mine_go.py"
echo "For more options, run: python mine_go.py --help"