FROM python:3.11-slim

# Install system dependencies for git-lfs (needed for large HuggingFace files)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    curl \
    && git lfs install \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy worker script
COPY worker.py .

# Create temp directory for downloads
RUN mkdir -p /tmp/worker && chmod 777 /tmp/worker

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TMPDIR=/tmp/worker

# Health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Run worker
CMD ["python", "-u", "worker.py"]
