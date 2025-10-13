# Start with CUDA 12.9.1 devel image that includes cuDNN 8
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

# Set environment variable to avoid tzdata prompt
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.12 and other necessary tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3-pip \
    python3-venv \
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

# Copy BR requirements and install them
COPY --chown=ray:100 br_requirements.txt .
RUN pip install --no-cache-dir -r br_requirements.txt

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
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Give ray user permissions to write to /tmp/workspace
RUN sudo mkdir -p /tmp/workspace && sudo chmod a+rwx -R /tmp/workspace

# Set working directory
WORKDIR /app

# No CMD or ENTRYPOINT to keep it flexible for Anyscale