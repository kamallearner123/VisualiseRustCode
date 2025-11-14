#!/bin/bash

# Stop script for Rust Visual Memory Debugger
# This script stops any running Django development server

echo "🛑 Stopping Rust Visual Memory Debugger..."

# Find and kill Django runserver processes
PIDS=$(ps aux | grep "[p]ython manage.py runserver" | awk '{print $2}')

if [ -z "$PIDS" ]; then
    echo "ℹ️  No running Django server found."
else
    echo "🔄 Stopping Django server (PIDs: $PIDS)..."
    echo "$PIDS" | xargs kill -9 2>/dev/null
    echo "✅ Server stopped successfully."
fi
