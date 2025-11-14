#!/bin/bash

# Setup script for Rust Visual Memory Debugger

echo "🦀 Setting up Rust Visual Memory Debugger..."

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
echo "✓ Python version: $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "⚡ Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Run migrations
echo "🔄 Running database migrations..."
python manage.py migrate

# Create static directory if it doesn't exist
mkdir -p static/css static/js

echo "✅ Setup complete!"
echo ""
echo "To start the server:"
echo "  source venv/bin/activate"
echo "  python manage.py runserver"
echo ""
echo "Then visit: http://localhost:8000"
