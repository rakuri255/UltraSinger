FROM nvidia/cuda:12.5.0-runtime-ubuntu22.04

WORKDIR /app
RUN apt update
RUN apt install git python3-pip -y
RUN pip3 install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2+cu117 --index-url https://download.pytorch.org/whl/cu117


WORKDIR /app

COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt
RUN git clone https://github.com/rakuri255/UltraSinger.git
WORKDIR /app/UltraSinger
RUN cp -Rv * ..
RUN apt install ffmpeg curl -y
RUN curl -L https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp -o /app/yt-dlp
RUN chmod a+rx /app/yt-dlp 
ENV UID=0
ENV GID=0
ENV UMASK=022

EXPOSE 8088 
WORKDIR /app/src
CMD ["bash" ]