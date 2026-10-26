FROM python:3.12-slim

WORKDIR /app

# System deps + SSL certs
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# CPU-only torch first so requirements.txt's torch pin resolves to this wheel
# instead of pulling the multi-GB CUDA build.
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.14.1 torchvision==0.29.1

# Remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=300 -r requirements.txt

# Copy source
COPY api/ ./api/
COPY db/ ./db/
COPY vectorizer/ ./vectorizer/
COPY scrapers/ ./scrapers/

# Models default to the benchmarked set (see vectorizer/*.py). They MUST match
# the models the Qdrant collection was indexed with — the API refuses to start
# on a dimension mismatch. For a small CPU box, index AND serve with the light
# image/rerank models:
#   ENV CLIP_MODEL=clip-ViT-B-32
#   ENV RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

EXPOSE 8001

CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8001}
