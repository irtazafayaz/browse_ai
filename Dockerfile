# Use official Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements first (so Docker caches this layer)
COPY requirements.txt .

# Install system SSL certificates
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*

# Upgrade pip then install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --timeout=300 -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose port
EXPOSE 8001

# Start the app
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8001"]
