FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

# Set timezone and configure non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install Python 3.12 from deadsnakes PPA and build tools
RUN apt-get update \
    && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && apt-get install -y git python3.12 python3.12-venv python3.12-dev ffmpeg curl tzdata \
       build-essential gcc g++ make \
    && apt-get clean  \
    && rm -rf /var/lib/apt/lists/*

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

# Install PyTorch with CUDA support (override the CPU version from pyproject.toml)
RUN uv pip install --system --python 3.12 torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128 --reinstall

# copy sources late to allow for caching of layers which contain all the dependencies
COPY . /app/UltraSinger


# no need to run as root
RUN chown -R 1000:1000 /app/UltraSinger
USER 1000:1000

WORKDIR /app/UltraSinger/src
CMD ["bash"]
