#!/bin/bash

# Setup script for Multi-Repo Detector

echo "Setting up Multi-Repo Detector..."
echo "=================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

if [ $? -ne 0 ]; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "Setup complete!"
echo ""
echo "To run the script:"
echo "  python3 detect_multirepo.py"
echo ""
echo "For higher GitHub API rate limits (recommended):"
echo "  export GITHUB_TOKEN='your_github_token'"
echo "  python3 detect_multirepo.py"
echo ""
echo "Get a GitHub token at: https://github.com/settings/tokens"
