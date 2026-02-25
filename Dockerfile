FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for brainflow and scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY src/ src/
COPY scripts/ scripts/
COPY configs/ configs/
COPY .streamlit/ .streamlit/

# Create data directory for SQLite and spectrograms
RUN mkdir -p data/spectrograms

# Expose ports: FastAPI (8765) and Streamlit (8501)
EXPOSE 8765 8501

# Environment variables with defaults
ENV DATABASE_URL=sqlite:////app/data/physician.db
ENV FASTAPI_URL=http://localhost:8765
ENV FASTAPI_PORT=8765
ENV STREAMLIT_PORT=8501
ENV SPECTROGRAM_DATA_DIR=/app/data/spectrograms

# Copy and use entrypoint script
COPY scripts/docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

ENTRYPOINT ["/docker-entrypoint.sh"]
