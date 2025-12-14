#!/bin/bash
# Run the Cybersecurity World Model Dashboard

# Default port
PORT=${1:-8502}

echo "Starting Cybersecurity World Model Dashboard..."
echo "Port: $PORT"
echo "URL: http://localhost:$PORT"
echo ""
echo "Press Ctrl+C to stop the dashboard"
echo ""

streamlit run dashboard.py --server.port $PORT --server.address localhost

