#!/bin/bash

# Start script for Rust Visual Memory Debugger
# This script activates the virtual environment and starts the Django server

echo "🦀 Starting Rust Visual Memory Debugger..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run ./setup.sh first to set up the project."
    exit 1
fi

# Activate virtual environment
echo "⚡ Activating virtual environment..."
source venv/bin/activate

# Check if database exists
if [ ! -f "db.sqlite3" ]; then
    echo "⚠️  Database not found. Running migrations..."
    python manage.py migrate
fi

# Start the Django development server
echo "🚀 Starting Django development server..."
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Rust Visual Memory Debugger"
echo "  Server: http://127.0.0.1:8000"
echo "  Press CTRL+C to stop the server"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python manage.py runserver
