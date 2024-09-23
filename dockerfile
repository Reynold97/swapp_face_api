# Start from the Anyscale CPU image
FROM anyscale/ray:2.35.0-py310-cpu

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Install CUDA 11.8
RUN wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb \
    && dpkg -i cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb \
    && cp /var/cuda-repo-ubuntu2204-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/ \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-11-8 \
    && rm -rf /var/lib/apt/lists/* \
    && rm cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb

# Set CUDA environment variables
ENV PATH /usr/local/cuda-11.8/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

# Install cuDNN 8.2 for CUDA 11.8
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/libcudnn8_8.2.4.15-1+cuda11.8_amd64.deb \
    && dpkg -i libcudnn8_8.2.4.15-1+cuda11.8_amd64.deb \
    && rm libcudnn8_8.2.4.15-1+cuda11.8_amd64.deb

# Install Python packages
RUN pip install --no-cache-dir \
    opencv-python-headless \
    cupy-cuda11x \
    onnxruntime-gpu==1.19.0 \
    python-multipart

# Set the working directory
WORKDIR /app

# Command to run your application
