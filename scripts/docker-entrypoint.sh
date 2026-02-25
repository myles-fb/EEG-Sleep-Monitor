#!/usr/bin/env bash
set -e

echo "Starting EEG Sleep Monitor..."
echo "  DATABASE_URL: ${DATABASE_URL}"
echo "  FASTAPI_PORT: ${FASTAPI_PORT}"
echo "  STREAMLIT_PORT: ${STREAMLIT_PORT}"

# Start FastAPI WebSocket server
cd /app/src
uvicorn server.ws_server:app --host 0.0.0.0 --port "${FASTAPI_PORT}" &
FASTAPI_PID=$!

# Start Streamlit dashboard
streamlit run app/physician_app.py \
    --server.port "${STREAMLIT_PORT}" \
    --server.address 0.0.0.0 \
    --server.headless true \
    &
STREAMLIT_PID=$!

# Trap signals for clean shutdown
cleanup() {
    echo ""
    echo "Shutting down..."
    kill "$FASTAPI_PID" "$STREAMLIT_PID" 2>/dev/null || true
    wait "$FASTAPI_PID" "$STREAMLIT_PID" 2>/dev/null || true
    echo "Done."
}
trap cleanup INT TERM

echo ""
echo "Services running:"
echo "  WebSocket server: http://0.0.0.0:${FASTAPI_PORT}"
echo "  Dashboard:        http://0.0.0.0:${STREAMLIT_PORT}"
echo ""

wait
