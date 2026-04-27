FROM python:3.12-slim

WORKDIR /app

# System deps + SSL certs
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# numpy first (must be before torch)
RUN pip install --no-cache-dir "numpy>=1.26,<2.0"

# CPU-only torch
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.2.2+cpu

# Remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=300 -r requirements.txt

# Copy source
COPY api/ ./api/
COPY db/ ./db/
COPY vectorizer/ ./vectorizer/
COPY scrapers/ ./scrapers/

EXPOSE 8001

CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8001}
