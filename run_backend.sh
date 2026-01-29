#!/bin/bash

# Smart Contract Vulnerability Detector - Backend Runner
# This script runs the FastAPI backend server

set -e  # Exit on any error

echo "🚀 Starting Smart Contract Vulnerability Detector Backend..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup.sh first."
    echo "   Execute: ./setup.sh"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Run the FastAPI server
echo "📡 Starting FastAPI server on http://0.0.0.0:8000"
echo "📖 API Documentation available at http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
