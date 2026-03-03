# Wan 2.2 I2V-A14B — RunPod Serverless Worker
# Model is stored on a RunPod network volume (not baked in).
#
# Build:
#   docker build -t jaycmedia8761/wan22-i2v:latest .
#   docker push jaycmedia8761/wan22-i2v:latest
#
# Runtime environment variables:
#   MODEL_PATH  (default: /runpod-volume/models/Wan2.2-I2V-A14B)
#   WAN_REPO    (default: /runpod-volume/Wan2.2)

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies (without flash_attn first — it's slow/optional)
RUN pip install --no-cache-dir \
    torch>=2.4.0 \
    torchvision>=0.19.0 \
    torchaudio && \
    pip install --no-cache-dir \
    opencv-python>=4.9.0.80 \
    diffusers>=0.31.0 \
    "transformers>=4.49.0,<=4.51.3" \
    tokenizers>=0.20.3 \
    "accelerate>=1.1.1" \
    tqdm \
    "imageio[ffmpeg]" \
    easydict \
    ftfy \
    imageio-ffmpeg \
    "numpy>=1.23.5,<2" \
    requests \
    Pillow \
    "runpod>=1.7.9"

# Install flash_attn (may fail on some build envs — optional for inference)
RUN pip install flash_attn --no-build-isolation || echo "flash_attn not installed — will use fallback attention"

# Copy handler
COPY handler.py .

# Default environment
ENV MODEL_PATH=/runpod-volume/models/Wan2.2-I2V-A14B
ENV WAN_REPO=/runpod-volume/Wan2.2
ENV PYTHONUNBUFFERED=1

# RunPod expects handler.py to be the entrypoint
CMD ["python", "-u", "handler.py"]
