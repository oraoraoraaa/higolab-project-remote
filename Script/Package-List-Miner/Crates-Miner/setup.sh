#!/bin/bash

# Deactivate Conda environment if active
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    conda deactivate
fi

# Create a virtual environment named 'venv' if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate the virtual environment
source venv/bin/activate

# Install the required packages
pip install -r requirements.txt

echo "Setup complete. The virtual environment 'venv' is ready and packages are installed."
