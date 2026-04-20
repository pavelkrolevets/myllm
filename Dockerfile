# Dockerfile for GPT Pretraining with ROCm Support
FROM rocm/pytorch:latest

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.1;8.6;8.9;9.0;9.1;9.2"
ENV ROCM_PATH=/opt/rocm

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    vim \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install tiktoken (Python package)
RUN pip install tiktoken

# Copy project files
COPY . /workspace/gpt-training

# Set working directory
WORKDIR /workspace/gpt-training

# Create necessary directories
RUN mkdir -p /workspace/gpt-training/checkpoints /workspace/gpt-training/logs

# Set entrypoint
ENTRYPOINT ["python", "gpt_pretrain_from_checkpoint.py"]