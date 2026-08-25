FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FORCE_CUDA=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip \
    git build-essential wget curl ca-certificates \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /workspace

COPY requirements.txt pyproject.toml ./

RUN pip install --upgrade pip \
 && pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch torchvision torchaudio \
 && pip install --no-cache-dir -r requirements.txt \
 || true

COPY . .

RUN pip install -e .

ENV HF_HOME=/workspace/.cache/huggingface \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

CMD ["bash"]
