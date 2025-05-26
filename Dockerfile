FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    ffmpeg \
    libsndfile1 \
    sox \
    libsox-dev \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Upgrade pip and install basic dependencies first, then requirements
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir numpy>=1.21.0 typing_extensions && \
    pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY test_client.py .

# Create directory for temporary files
RUN mkdir -p /tmp/transcription_cache

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python3", "app.py"] 