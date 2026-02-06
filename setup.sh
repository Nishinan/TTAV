#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Function to display error messages and exit
error_exit() {
  echo "$1" >&2
  exit 1
}

# Set ENV_NAME with default value "phishpedia" if not already set
ENV_NAME="${ENV_NAME:-visualizer}"

# Ensure Conda is installed
if ! command -v conda &> /dev/null; then
  error_exit "Conda is not installed. Please install Conda and try again."
fi

# Create and activate the Conda environment
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

if conda info --envs | grep -w "^$ENV_NAME" > /dev/null 2>&1; then
  echo "Activating existing Conda environment: $ENV_NAME"
else
  echo "Creating new Conda environment: $ENV_NAME with Python 3.10"
  conda create -y -n "$ENV_NAME" python=3.10
fi
conda init
conda activate "$ENV_NAME"


# Install additional Python dependencies from requirements.txt
if [ -f "requirements.txt" ]; then
  echo "Installing additional Python dependencies from requirements.txt..."
  conda run -n "$ENV_NAME" pip install -r requirements.txt
else
  error_exit "requirements.txt not found in the current directory."
fi


echo "All packages installed and models downloaded successfully!"