FROM nvidia/cuda:12.5.0-runtime-ubuntu22.04

WORKDIR /app
RUN apt-get update
RUN apt-get install git python3-pip -y

RUN git clone https://github.com/rakuri255/UltraSinger.git
WORKDIR /app/UltraSinger
RUN apt-get install ffmpeg curl -y
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir  torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir  tensorflow[and-cuda]==2.16.1
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
ENV UID=0
ENV GID=0
ENV UMASK=022

EXPOSE 8088 
WORKDIR /app/UltraSinger/src
CMD ["bash" ]
