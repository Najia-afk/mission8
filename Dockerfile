# Mission 8 - Technical Watch: PanCAN POC
# CUDA-enabled image for GPU training (RTX 5070 - Blackwell sm_120)
# Using CUDA 12.8 for better Blackwell architecture support

FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

LABEL maintainer="OpenClassrooms Data Scientist"
LABEL project="Mission 8 - PanCAN POC"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install Python 3.12 and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && ln -sf /usr/local/bin/pip /usr/bin/pip

# Install PyTorch 2.9.1 with CUDA 12.8 for RTX 5070 (Blackwell sm_120 support)
RUN pip install --upgrade pip && \
    pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128

# Copy requirements after torch install to preserve cache
COPY requirements.txt .

# Install requirements (excluding torch/torchvision/torchaudio and comments)
# Using --ignore-installed to avoid conflicts with distutils-installed packages
RUN sed 's/#.*//g' requirements.txt | grep -v -E '^\s*torch|^\s*torchvision|^\s*torchaudio|^\s*$' | xargs pip install --ignore-installed

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p /app/models /app/reports/figures /app/mlruns

# Expose Jupyter port
EXPOSE 8888

# Default command - Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
