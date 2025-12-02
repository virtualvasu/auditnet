#!/bin/bash

# Smart Contract Vulnerability Analyzer - Streamlit App

echo "🔐 Smart Contract Vulnerability Analyzer - Streamlit Interface"
echo "============================================================="

# Check if we're in the right directory
if [ ! -f "streamlit_app.py" ]; then
    echo "❌ Error: streamlit_app.py not found. Please run this script from the project root directory"
    exit 1
fi

# Check if the models directory exists
if [ ! -d "models" ] && [ ! -d "results/checkpoints" ]; then
    echo "⚠️  Warning: No trained models found!"
    echo "   Please train a model first using notebook 04_train_codebert_baseline.ipynb"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "🚀 Starting Streamlit app..."
echo "   🌐 Open your browser to the URL shown below"
echo "   🛑 Use Ctrl+C to stop the app"
echo ""

# Use the project's virtual environment if available
if [ -f "venv/bin/python" ]; then
    echo "   Using project virtual environment"
    ./venv/bin/streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501
else
    echo "   Using system Python"
    streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501
fi