#!/bin/bash
# Start Cybersecurity World Model Dashboard with venv activation

cd "$(dirname "$0")"

# Check if we're already in a venv (VIRTUAL_ENV is set)
if [ -n "$VIRTUAL_ENV" ]; then
    echo "✓ Using active virtual environment: $VIRTUAL_ENV"
# Try to find and activate venv
elif [ -d "venv" ]; then
    source venv/bin/activate
    echo "✓ Activated local venv"
elif [ -d "../venv" ]; then
    source ../venv/bin/activate
    echo "✓ Activated parent venv"
elif [ -d "../../venv" ]; then
    source ../../venv/bin/activate
    echo "✓ Activated grandparent venv"
else
    echo "⚠️  No venv directory found, but continuing..."
    echo "   If you're already in a venv (you should see (venv) in prompt), that's fine!"
fi

# Check if streamlit is installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "📦 Installing Streamlit and Plotly..."
    pip install streamlit plotly
fi

# Get port from argument or use default
PORT=${1:-8502}

echo "🚀 Starting Cybersecurity World Model Dashboard..."
echo "📍 Port: $PORT"
echo "🌐 URL: http://localhost:$PORT"
echo ""
echo "Press Ctrl+C to stop the dashboard"
echo ""

# Start dashboard
streamlit run dashboard.py --server.port $PORT --server.address localhost

