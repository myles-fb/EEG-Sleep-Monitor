#!/usr/bin/env bash
#
# Launch both the FastAPI WebSocket server and the Streamlit physician dashboard.
# Usage: ./scripts/start_server.sh
#
# Stops both processes on Ctrl+C.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$PROJECT_DIR/src"

FASTAPI_PORT="${FASTAPI_PORT:-8765}"
STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"

cleanup() {
    echo ""
    echo "Shutting down..."
    kill "$FASTAPI_PID" "$STREAMLIT_PID" 2>/dev/null || true
    wait "$FASTAPI_PID" "$STREAMLIT_PID" 2>/dev/null || true
    echo "Done."
}

trap cleanup INT TERM

echo "Starting FastAPI WebSocket server on port $FASTAPI_PORT..."
cd "$SRC_DIR"
uvicorn server.ws_server:app --host 0.0.0.0 --port "$FASTAPI_PORT" &
FASTAPI_PID=$!

echo "Starting Streamlit dashboard on port $STREAMLIT_PORT..."
streamlit run "$SRC_DIR/app/physician_app.py" --server.port "$STREAMLIT_PORT" &
STREAMLIT_PID=$!

echo ""
echo "Services running:"
echo "  WebSocket server: http://localhost:$FASTAPI_PORT"
echo "  Physician dashboard: http://localhost:$STREAMLIT_PORT"
echo ""
echo "Press Ctrl+C to stop both."

wait
