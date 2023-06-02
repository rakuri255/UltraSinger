import io
import os

import yt_dlp
from PIL import Image

from modules.Log import PRINT_ULTRASTAR


def get_youtube_title(url):
    ydl_opts = {}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(
            url, download=False  # We just want to extract the info
        )

    return result["title"]


def download_youtube_audio(url, clear_filename, output_path):
    print(f"{PRINT_ULTRASTAR} Downloading Audio")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path + "/" + clear_filename,
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}
        ],
    }

    start_download(ydl_opts, url)


def download_youtube_thumbnail(url, clear_filename, output_path):
    print(f"{PRINT_ULTRASTAR} Downloading thumbnail")
    ydl_opts = {
        "skip_download": True,
        "writethumbnail": True,
    }

    download_and_convert_thumbnail(ydl_opts, url, clear_filename, output_path)


def download_and_convert_thumbnail(ydl_opts, url, clear_filename, output_path):
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=False)
        thumbnail_url = info_dict.get("thumbnail")
        if thumbnail_url:
            response = ydl.urlopen(thumbnail_url)
            image_data = response.read()
            image = Image.open(io.BytesIO(image_data))
            image.save(
                os.path.join(output_path, clear_filename + " [CO].jpg"), "JPEG"
            )


def download_youtube_video(url, clear_filename, output_path):
    print(f"{PRINT_ULTRASTAR} Downloading Video")
    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4",
        "outtmpl": output_path + "/" + clear_filename + ".mp4",
    }
    start_download(ydl_opts, url)


def start_download(ydl_opts, url):
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        errors = ydl.download(url)
        if errors:
            raise Exception("Download failed with error: " + str(errors))
