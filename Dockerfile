# 3.5-bit Quantization for LLM Inference
# Docker image for reproducible builds and deployment

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY 2025-3.5bit-groq-mvp/ /app/2025-3.5bit-groq-mvp/
COPY papers/ /app/papers/

# Set Python path
ENV PYTHONPATH=/app/2025-3.5bit-groq-mvp:$PYTHONPATH

# Create output directories
RUN mkdir -p /app/output /app/figures /app/results

# Default command
CMD ["python", "2025-3.5bit-groq-mvp/benchmark_3p5bit.py"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import numpy; import matplotlib"

# Labels
LABEL maintainer="jimxzai@github.com"
LABEL version="1.0"
LABEL description="3.5-bit Dynamic Asymmetric Quantization for LLMs"
