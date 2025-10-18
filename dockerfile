# Start with CUDA 12.9.1 devel image that includes cuDNN 8
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

# Set environment variable to avoid tzdata prompt
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.12 and other necessary tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3-pip \
    python3-venv \
    python3-dev \
    wget \
    git \
    sudo \
    tzdata \
    supervisor \
    openssh-client \
    openssh-server \
    rsync \
    zip \
    unzip \
    nfs-common \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create the `ray` user
RUN (userdel -r $(getent passwd 1000 | cut -d: -f1) 2>/dev/null || true) \
    && useradd -ms /bin/bash -d /home/ray ray --uid 1000 --gid 100 \
    && usermod -aG root ray \
    && echo 'ray ALL=NOPASSWD: ALL' >> /etc/sudoers

# Switch to the `ray` user.
USER ray
ENV HOME=/home/ray

# Create a virtual environment
RUN python3.12 -m venv --system-site-packages $HOME/virtualenv
ENV PATH=$HOME/virtualenv/bin:$PATH
ENV VIRTUAL_ENV=$HOME/virtualenv

# Upgrade pip and ensure no old cupy versions exist
RUN pip install --upgrade pip setuptools wheel && \
    pip uninstall -y cupy-cuda11x cupy-cuda10x cupy-cuda110 cupy-cuda111 cupy-cuda112 cupy || true

# Install Google Cloud SDK without using deprecated apt-key
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | \
    sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
    sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
RUN sudo apt-get update && sudo apt-get install -y google-cloud-cli

# Create necessary directories
RUN mkdir -p /tmp/ray && mkdir -p /tmp/supervisord

# Copy requirements.txt and install dependencies
COPY --chown=ray:100 requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy CodeFormer requirements and install them
COPY --chown=ray:100 codeformer_requirements.txt .
RUN pip install --no-cache-dir -r codeformer_requirements.txt

# Install CUDA and TensorRT packages
RUN pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com tensorrt tensorrt-cu12 tensorrt-cu12-bindings tensorrt-cu12-libs

# Install CUDA runtime packages
RUN pip install --no-cache-dir \
    nvidia-cuda-runtime-cu12 \
    nvidia-cuda-nvrtc-cu12 \
    nvidia-cudnn-cu12 \
    nvidia-cublas-cu12

# Install cupy for CUDA 12.x
RUN pip install --no-cache-dir cupy-cuda12x

# Install cuda-python 12.6.0 (compatible with old import style)
RUN pip install --no-cache-dir cuda-python==12.6.0

# Install ONNX Runtime with CUDA 12.x support (CRITICAL FIX)
RUN pip uninstall -y onnxruntime onnxruntime-gpu || true
RUN pip install --no-cache-dir onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

# Install additional Python packages
RUN pip install --no-cache-dir \
    ray[serve]==2.46.0 \
    anyscale \
    'urllib3<1.27' \
    Pillow \
    awscli \
    google-cloud-storage

# Set CUDA environment variables
ENV CUDA_HOME="/usr/local/cuda"
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH}"
ENV CUDA_MODULE_LOADING="LAZY"

# Give ray user permissions to write to /tmp/workspace
RUN sudo mkdir -p /tmp/workspace && sudo chmod a+rwx -R /tmp/workspace

# Set working directory
WORKDIR /app

# No CMD or ENTRYPOINT to keep it flexible for Anyscale