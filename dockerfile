FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgoogle-perftools4 \
    libtcmalloc-minimal4 \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Create models and credentials directories
RUN mkdir -p /app/models/weights/CodeFormer \
    && mkdir -p /app/models/weights/facelib \
    && mkdir -p /app/models/weights/realesrgan \
    && mkdir -p /app/credentials

# Copy requirements and install Python dependencies
COPY requirements.txt codeformer_requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt \
    && pip3 install --no-cache-dir -r codeformer_requirements.txt

# Copy the application code
COPY app/ ./app/
COPY models/ ./models/

# Expose the port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/docs || exit 1

# Run the FastAPI application
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]