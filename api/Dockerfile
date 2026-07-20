FROM python:3.11-slim

WORKDIR /app

# System dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# Build context is repo root — requirements live inside api/
COPY api/requirements-api.txt ./requirements-api.txt
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy api/ as a Python package — preserves `from api.xxx import …` import paths
COPY api/ ./api/

# Copy SQL schema — used by _ensure_schema() on startup for cloud DBs
COPY sql/ ./sql/

# Copy committed model metadata (production_model.txt, metrics JSON)
# .h5 weights are gitignored; create_stub_model.py generates them at build time
COPY models/ ./models/

# Generate stub .h5 if no real trained model exists in the image
RUN python api/create_stub_model.py

EXPOSE 8000

# Use shell form so $PORT is expanded — Render injects PORT at runtime
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
