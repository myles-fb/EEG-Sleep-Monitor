"""Shared configuration for the Streamlit dashboard."""

import os

FASTAPI_URL = os.environ.get("FASTAPI_URL", "http://localhost:8765")
