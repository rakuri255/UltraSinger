"""YouTube Downloader"""

import io
import os

import yt_dlp
from PIL import Image

from modules.os_helper import sanitize_filename, get_unused_song_output_dir
from modules import os_helper
from modules.ProcessData import MediaInfo
from modules.Audio.bpm import get_bpm_from_file
from modules.console_colors import ULTRASINGER_HEAD
from modules.Image.image_helper import crop_image_to_square
from modules.musicbrainz_client import get_music_infos


def get_youtube_title(url: str) -> tuple[str, str]:
    """Get the title of the YouTube video"""

    ydl_opts = {}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(
            url, download=False  # We just want to extract the info
        )

    if "artist" in result:
        return result["artist"].strip(), result["track"].strip()
    if "-" in result["title"]:
        return result["title"].split("-")[0].strip(), result["title"].split("-")[1].strip()
    return result["channel"].strip(), result["title"].strip()


def __download_youtube_audio(url: str, clear_filename: str, output_path: str):
    """Download audio from YouTube"""

    print(f"{ULTRASINGER_HEAD} Downloading Audio")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path + "/" + clear_filename,
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}
        ],
    }

    __start_download(ydl_opts, url)


def __download_youtube_thumbnail(url: str, clear_filename: str, output_path: str) -> str:
    """Download thumbnail from YouTube"""

    print(f"{ULTRASINGER_HEAD} Downloading thumbnail")
    ydl_opts = {
        "skip_download": True,
        "writethumbnail": True,
    }

    thumbnail_url = download_and_convert_thumbnail(ydl_opts, url, clear_filename, output_path)
    return thumbnail_url


def download_and_convert_thumbnail(ydl_opts, url: str, clear_filename: str, output_path: str) -> str:
    """Download and convert thumbnail from YouTube"""

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=False)
        thumbnail_url = info_dict.get("thumbnail")
        if thumbnail_url:
            response = ydl.urlopen(thumbnail_url)
            image_data = response.read()
            image = Image.open(io.BytesIO(image_data))
            image = image.convert('RGB') # Convert to RGB to avoid transparency or RGBA issues
            image_path = os.path.join(output_path, clear_filename + " [CO].jpg")
            image.save(image_path, "JPEG")
            crop_image_to_square(image_path)
            return thumbnail_url
        else:
            return ""


def __download_youtube_video(url: str, clear_filename: str, output_path: str) -> None:
    """Download video from YouTube"""

    print(f"{ULTRASINGER_HEAD} Downloading Video")
    ydl_opts = {
        "format": "bestvideo[ext=mp4]/mp4",
        "outtmpl": output_path + "/" + clear_filename + ".mp4",
    }
    __start_download(ydl_opts, url)


def __start_download(ydl_opts, url: str) -> None:
    """Start the download the ydl_opts"""

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        errors = ydl.download(url)
        if errors:
            raise Exception("Download failed with error: " + str(errors))


def download_from_youtube(input_file_path: str, output_folder_path: str) -> tuple[str, str, str, MediaInfo]:
    """Download from YouTube"""
    (artist, title) = get_youtube_title(input_file_path)

    # Get additional data for song
    (title_info, artist_info, year_info, genre_info) = get_music_infos(
        f"{artist} - {title}"
    )

    if title_info is not None:
        title = title_info
        artist = artist_info

    basename_without_ext = sanitize_filename(f"{artist} - {title}")
    basename = basename_without_ext + ".mp3"
    song_output = os.path.join(output_folder_path, basename_without_ext)
    song_output = get_unused_song_output_dir(song_output)
    os_helper.create_folder(song_output)
    __download_youtube_audio(input_file_path, basename_without_ext, song_output)
    __download_youtube_video(input_file_path, basename_without_ext, song_output)
    thumbnail_url = __download_youtube_thumbnail(
        input_file_path, basename_without_ext, song_output
    )
    audio_file_path = os.path.join(song_output, basename)
    real_bpm = get_bpm_from_file(audio_file_path)
    return (
        basename_without_ext,
        song_output,
        audio_file_path,
        MediaInfo(artist=artist, title=title, year=year_info, genre=genre_info, bpm=real_bpm,
                  youtube_thumbnail_url=thumbnail_url),
    )
