#!/bin/bash
# PoseSense AI - Startup Script

echo "🚀 Starting PoseSense AI..."
echo ""

# Go to project directory
cd "$(dirname "$0")"

echo "🌐 Starting server..."

# Start the Flask app using Gunicorn (production server)
gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 0
