FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

# Set timezone and configure non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install Python 3.12 from deadsnakes PPA, audio libraries, and build tools
RUN apt-get update \
    && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && apt-get install -y git python3.12 python3.12-venv python3.12-dev ffmpeg curl tzdata \
    libsndfile1 build-essential gcc g++ make \
    && apt-get clean  \
    && rm -rf /var/lib/apt/lists/*

# Symlink python to python3.12 so `python` is available in container
RUN ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3

# Configure dynamic library path for NVIDIA CUDA / cuDNN libraries used by CTranslate2 (WhisperX)
ENV LD_LIBRARY_PATH="/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH}"
ENV PYTHONWARNINGS="ignore::SyntaxWarning"

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
ENV UV_LINK_MODE=copy

# copy pyproject.toml first to leverage container image build cache
COPY ./pyproject.toml /app/UltraSinger/pyproject.toml
# Need to copy some minimal source structure for editable install
RUN mkdir -p /app/UltraSinger/src
WORKDIR /app/UltraSinger

# Install dependencies from pyproject.toml directly without venv (container is already isolated)
# Using build isolation (without --no-build-isolation) so uv handles all build dependencies automatically
RUN uv pip install --system --python 3.12 -e .

# Ensure Levenshtein, SwiftF0, torchcrepe, audio-separator, onnxruntime-gpu (CUDA 12), rmvpe-onnx, and Spotify Basic Pitch are installed
RUN uv pip install --system --python 3.12 Levenshtein "swift-f0==0.1.2" torchcrepe "audio-separator[gpu]" onnxruntime-gpu==1.20.0 rmvpe-onnx mir-eval \
    && uv pip install --system --python 3.12 --no-deps basic-pitch==0.4.0

# Install PyTorch with CUDA support (override the CPU version from pyproject.toml)
RUN uv pip install --system --python 3.12 torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128 --reinstall

# Fix onnxruntime conflict: sub-dependencies (swift-f0, rmvpe-onnx) install CPU onnxruntime,
# which collides with onnxruntime-gpu in dist-packages causing AttributeError on import.
RUN uv pip uninstall --system --python 3.12 onnxruntime onnxruntime-gpu \
    && uv pip install --system --python 3.12 --no-deps onnxruntime-gpu==1.20.0

# copy sources late to allow for caching of layers which contain all the dependencies
COPY . /app/UltraSinger

# Create ultrasinger user (UID 1000) with a valid HOME directory for AI model caches (Torch, HuggingFace)
RUN useradd -m -u 1000 -s /bin/bash ultrasinger \
    && mkdir -p /home/ultrasinger/.cache \
    && chown -R ultrasinger:ultrasinger /home/ultrasinger /app/UltraSinger

ENV HOME=/home/ultrasinger
USER ultrasinger

WORKDIR /app/UltraSinger/src
CMD ["bash"]