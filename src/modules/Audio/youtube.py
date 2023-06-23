"""YouTube Downloader"""

import io
import os

import yt_dlp
from PIL import Image

from modules.console_colors import ULTRASINGER_HEAD
from modules.Image.image_helper import crop_image_to_square

def get_youtube_title(url: str) -> tuple[str, str]:
    """Get the title of the YouTube video"""

    ydl_opts = {}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(
            url, download=False  # We just want to extract the info
        )

    if "artist" in result:
        return result["artist"], result["track"]
    if "-" in result["title"]:
        return result["title"].split("-")[0], result["title"].split("-")[1]
    return result["channel"], result["title"]


def download_youtube_audio(url: str, clear_filename: str, output_path: str):
    """Download audio from YouTube"""

    print(f"{ULTRASINGER_HEAD} Downloading Audio")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path + "/" + clear_filename,
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}
        ],
    }

    start_download(ydl_opts, url)


def download_youtube_thumbnail(url: str, clear_filename: str, output_path: str):
    """Download thumbnail from YouTube"""

    print(f"{ULTRASINGER_HEAD} Downloading thumbnail")
    ydl_opts = {
        "skip_download": True,
        "writethumbnail": True,
    }

    download_and_convert_thumbnail(ydl_opts, url, clear_filename, output_path)


def download_and_convert_thumbnail(ydl_opts, url: str, clear_filename: str, output_path: str) -> None:
    """Download and convert thumbnail from YouTube"""

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=False)
        thumbnail_url = info_dict.get("thumbnail")
        if thumbnail_url:
            response = ydl.urlopen(thumbnail_url)
            image_data = response.read()
            image = Image.open(io.BytesIO(image_data))
            image_path = os.path.join(output_path, clear_filename + " [CO].jpg")
            image.save(image_path, "JPEG")
            crop_image_to_square(image_path)


def download_youtube_video(url: str, clear_filename: str, output_path: str) -> None:
    """Download video from YouTube"""

    print(f"{ULTRASINGER_HEAD} Downloading Video")
    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4",
        "outtmpl": output_path + "/" + clear_filename + ".mp4",
    }
    start_download(ydl_opts, url)


def start_download(ydl_opts, url: str) -> None:
    """Start the download the ydl_opts"""

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        errors = ydl.download(url)
        if errors:
            raise Exception("Download failed with error: " + str(errors))
