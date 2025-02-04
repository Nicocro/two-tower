
#!/bin/bash

# Exit on error
set -e

echo "Starting setup script..."

# Install Python 3.12
echo "Installing Python 3.12..."
sudo apt update
sudo apt install -y python3.12 python3.12-venv

# Install Poetry
echo "Installing Poetry..."
curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to PATH
export PATH="/root/.local/bin:$PATH"

# Install dependencies using Poetry
echo "Installing dependencies with Poetry..."
poetry env use python3.12
poetry install

# Create a directory for model weights and vocab
mkdir -p models

# Download model weights and vocab using wget
echo "Downloading model weights and vocabulary..."
wget -O models/cbow_text8_weights.pt https://huggingface.co/nico-x/text8-cbow-w2v/resolve/main/cbow_text8_weights.pt
wget -O models/text8_vocab.json https://huggingface.co/nico-x/text8-cbow-w2v/resolve/main/text8_vocab.json

echo "Setup complete!