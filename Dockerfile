FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04

# note: the python3-pip package contains Python 3.12 on Ubuntu 22.04
RUN apt-get update \
    && apt-get install git python3 python3-venv ffmpeg curl -y  \
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
RUN uv pip install --system --python 3.12 -e . --no-build-isolation

# Install PyTorch with CUDA support (override the CPU version from pyproject.toml)
RUN uv pip install --system --python 3.12 torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128 --reinstall

# copy sources late to allow for caching of layers which contain all the dependencies
COPY . /app/UltraSinger


# no need to run as root
RUN chown -R 1000:1000 /app/UltraSinger
USER 1000:1000

WORKDIR /app/UltraSinger/src
CMD ["bash"]
