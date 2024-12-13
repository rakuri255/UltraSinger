FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04 AS base

# note: the python3-pip package contains Python 3.10 on Ubuntu 22.04
RUN apt-get update \
    && apt-get install git python3-pip python3.10-venv ffmpeg -y  \
    && apt-get clean  \
    && rm -rf /var/lib/apt/lists/*

# copy only the requirements file to leverage container image build cache
COPY ./requirements.txt /app/UltraSinger/requirements.txt
WORKDIR /app/UltraSinger

# no need to run as root
RUN chown -R 1000:1000 /app/UltraSinger
USER 1000:1000

# setup venv
ENV VIRTUAL_ENV=/app/UltraSinger/.venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir tensorflow[and-cuda]==2.16.1

# create runtime image with minimal dependencies
FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04 AS runtime

# note: the python3 package contains Python 3.10 on Ubuntu 22.04
RUN apt-get update \
    && apt-get install python3 ffmpeg -y  \
    && apt-get clean  \
    && rm -rf /var/lib/apt/lists/*

USER 1000:1000

COPY --from=base --chown=1000:1000 /app/UltraSinger/ /app/UltraSinger/

# setup venv
ENV VIRTUAL_ENV=/app/UltraSinger/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# copy sources late to allow for caching of layers which contain all the dependencies
COPY . /app/UltraSinger
WORKDIR /app/UltraSinger/src
CMD ["bash" ]
