#!/bin/bash
# PoseSense AI - Startup Script

echo "🚀 Starting PoseSense AI..."
echo ""

# Install dependencies
echo "📦 Installing dependencies..."
pip install flask opencv-python mediapipe reportlab --break-system-packages -q

echo ""
echo "✅ Dependencies installed"
echo ""
echo "🌐 Starting server at http://localhost:5000"
echo ""
echo "📌 Routes:"
echo "   Landing Page : http://localhost:5000/"
echo "   Dashboard    : http://localhost:5000/dashboard"
echo ""

cd "$(dirname "$0")"
python app.py