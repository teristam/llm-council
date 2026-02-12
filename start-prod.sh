#!/bin/bash

# LLM Council - Production Start Script
# Use this when accessing remotely to avoid HMR reload issues

echo "Building frontend for production..."
cd frontend
npm run build
cd ..

echo ""
echo "Starting LLM Council in production mode..."
echo ""

# Start backend (which now serves frontend static files)
uv run python -m backend.main &
BACKEND_PID=$!

echo ""
echo "LLM Council is running in production mode!"
echo "  URL: http://localhost:8001"
echo ""
echo "Press Ctrl+C to stop"

# Wait for Ctrl+C
trap "kill $BACKEND_PID 2>/dev/null; exit" SIGINT SIGTERM
wait
