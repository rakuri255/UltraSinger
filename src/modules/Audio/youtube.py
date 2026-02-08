"""YouTube Downloader"""

import os
import yt_dlp

from modules.os_helper import sanitize_filename, get_unused_song_output_dir
from modules import os_helper
from modules.ProcessData import MediaInfo
from modules.Audio.bpm import get_bpm_from_file
from modules.console_colors import ULTRASINGER_HEAD
from modules.Image.image_helper import save_image
from modules.musicbrainz_client import search_musicbrainz
from modules.ffmpeg_helper import separate_audio_video


def get_youtube_title(url: str, cookiefile: str = None) -> tuple[str, str]:
    """Get the title of the YouTube video"""

    ydl_opts = {
        "cookiefile": cookiefile,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(
            url, download=False  # We just want to extract the info
        )

    if "artist" in result:
        return result["artist"].strip(), result["track"].strip()
    if "-" in result["title"]:
        return result["title"].split("-")[0].strip(), result["title"].split("-")[1].strip()
    return result["channel"].strip(), result["title"].strip()


def __download_youtube_video_with_audio(url: str, clear_filename: str, output_path: str, cookiefile: str = None) -> str:
    """Download video with audio from YouTube and return the file extension"""

    print(f"{ULTRASINGER_HEAD} Downloading Video with Audio")
    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio/best",
        "outtmpl": output_path + "/" + clear_filename + ".%(ext)s",
        "merge_output_format": "mp4",
        "cookiefile": cookiefile,
    }
    __start_download(ydl_opts, url)
    return "mp4"


def __download_youtube_thumbnail(url: str, clear_filename: str, output_path: str, cookiefile: str = None) -> str:
    """Download thumbnail from YouTube"""

    print(f"{ULTRASINGER_HEAD} Downloading thumbnail")
    ydl_opts = {
        "skip_download": True,
        "writethumbnail": True,
        "cookiefile": cookiefile,
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
            save_image(image_data, clear_filename, output_path)
            return thumbnail_url
        else:
            return ""




def __start_download(ydl_opts, url: str) -> None:
    """Start the download the ydl_opts"""

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        errors = ydl.download(url)
        if errors:
            raise Exception("Download failed with error: " + str(errors))


def download_from_youtube(input_url: str, output_folder_path: str, cookiefile: str = None) -> tuple[str, str, str, MediaInfo]:
    """Download from YouTube"""
    (artist, title) = get_youtube_title(input_url, cookiefile)

    # Get additional data for song
    song_info = search_musicbrainz(title, artist)

    basename_without_ext = sanitize_filename(f"{song_info.artist} - {song_info.title}")
    song_output = os.path.join(output_folder_path, basename_without_ext)
    song_output = get_unused_song_output_dir(song_output)
    os_helper.create_folder(song_output)

    print(f"{ULTRASINGER_HEAD} Downloading from YouTube")
    video_ext = __download_youtube_video_with_audio(
        input_url, basename_without_ext, song_output, cookiefile
    )
    video_with_audio_path = os.path.join(song_output, f"{basename_without_ext}.{video_ext}")

    # Separate audio and video
    audio_file_path, final_video_path = separate_audio_video(
        video_with_audio_path, basename_without_ext, song_output
    )

    if song_info.cover_url is not None and song_info.cover_image_data is not None:
        cover_url = song_info.cover_url
        save_image(song_info.cover_image_data, basename_without_ext, song_output)
    else:
        cover_url = __download_youtube_thumbnail(
            input_url, basename_without_ext, song_output, cookiefile
        )

    real_bpm = get_bpm_from_file(audio_file_path)
    return (
        basename_without_ext,
        song_output,
        audio_file_path,
        MediaInfo(
            artist=song_info.artist,
            title=song_info.title,
            year=song_info.year,
            genre=song_info.genres,
            bpm=real_bpm,
            cover_url=cover_url,
            video_url=input_url,
        ),
    )
