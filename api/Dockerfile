FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY . .

EXPOSE 8000

# Challenge 9 fix: model loads on startup event, not per request
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
