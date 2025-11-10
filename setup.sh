#!/bin/bash

# Smart Contract Vulnerability Detector - Setup Script
# This script helps set up the development environment

set -e  # Exit on any error

echo "🚀 Setting up Smart Contract Vulnerability Detector..."

# Check Python version
python_version=$(python3 --version 2>&1 | grep -o '[0-9]\+\.[0-9]\+')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.8 or higher is required. Found: $python_version"
    exit 1
fi

echo "✅ Python version check passed: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📈 Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing requirements..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directory structure..."
mkdir -p data/processed data/raw data/external
mkdir -p models
mkdir -p results/checkpoints results/metrics results/visualizations results/predictions
mkdir -p logs
mkdir -p outputs

echo "🎉 Setup complete!"
echo ""
echo "To activate the environment in future sessions, run:"
echo "source venv/bin/activate"
echo ""
echo "To start Jupyter Lab, run:"
echo "jupyter lab"
echo ""
echo "Happy coding! 🚀"