FROM nvidia/cuda:12.5.0-runtime-ubuntu22.04

WORKDIR /app
RUN apt update
RUN apt install git python3-pip -y

RUN git clone https://github.com/rakuri255/UltraSinger.git
WORKDIR /app/UltraSinger
RUN cp -Rv * ..
WORKDIR /app
RUN apt install ffmpeg curl -y
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --upgrade pip
RUN pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install tensorflow[and-cuda]==2.16.1
ENV UID=0
ENV GID=0
ENV UMASK=022

EXPOSE 8088 
WORKDIR /app/src
CMD ["bash" ]