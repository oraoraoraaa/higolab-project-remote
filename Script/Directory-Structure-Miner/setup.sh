#!/bin/bash

# Setup script for Directory Structure Miner

echo "Setting up Directory Structure Miner..."

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create results directory if it doesn't exist
if [ ! -d "results" ]; then
    echo "Creating results directory..."
    mkdir results
fi

echo ""
echo "Setup complete!"
echo ""
echo "To use the miner:"
echo "  1. Activate the virtual environment: source venv/bin/activate"
echo "  2. (Optional) Set GitHub token: export GITHUB_TOKEN='your_token_here'"
echo "  3. Run the miner: python mine_directory_structure.py"
echo ""
echo "For more options, run: python mine_directory_structure.py --help"
