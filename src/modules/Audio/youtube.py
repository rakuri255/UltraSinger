"""YouTube Downloader"""

import os
import re

import yt_dlp

from modules.os_helper import sanitize_filename, get_unused_song_output_dir
from modules import os_helper
from modules.ProcessData import MediaInfo
from modules.Audio.bpm import get_bpm_from_file
from modules.console_colors import ULTRASINGER_HEAD, red_highlighted
from modules.Image.image_helper import save_image
from modules.musicbrainz_client import search_musicbrainz
from modules.ffmpeg_helper import separate_audio_video


# Suffixes that YouTube Music commonly adds to track names which may not
# reflect the actual video content.  Only these are considered for removal;
# arbitrary parenthetical metadata like "(feat. …)" or "(Part 2)" is kept.
_STRIPPABLE_SUFFIXES = re.compile(
    r"^("
    r"live(?:\s.+)?|"           # live, live at …, live from …
    r"acoustic(?:\s.+)?|"       # acoustic, acoustic version
    r"remaster(?:ed)?(?:\s.+)?|"  # remastered, remastered 2024
    r"(?:official\s)?(?:video|audio|music\svideo)|"
    r"lyrics?(?:\svideo)?|"     # lyric, lyrics, lyric video
    r"explicit|clean|"
    r"radio\s?edit|"
    r"single\sversion"
    r")$",
    re.IGNORECASE,
)


def strip_unmatched_suffixes(track: str, video_title: str) -> str:
    """Remove trailing parenthetical suffixes from the YT Music track name
    that do not appear in the actual video title.

    YouTube Music metadata sometimes adds qualifiers like '(live)' or
    '(acoustic)' that do not match the uploaded video.  By cross-checking
    against the video title we can strip these misleading suffixes while
    keeping legitimate ones like '(feat. …)'.

    Only *trailing* parenthesised groups whose content matches a known
    set of media qualifiers are considered, so arbitrary metadata like
    '(feat. Guest)' or '(Part 2)' is never touched.
    """
    # Iteratively strip trailing (...) groups that are absent from the title
    while True:
        m = re.search(r'\s*\(([^)]+)\)\s*$', track)
        if not m:
            break
        suffix_content = m.group(1).strip()
        # Only consider known media qualifiers for stripping
        if not _STRIPPABLE_SUFFIXES.match(suffix_content):
            break  # not a known qualifier - leave it alone
        if re.search(
            rf"(?<!\w){re.escape(suffix_content)}(?!\w)",
            video_title,
            re.IGNORECASE,
        ):
            break  # suffix is present in video title - keep it
        print(
            f"{ULTRASINGER_HEAD} {red_highlighted(f'Stripped YT Music suffix \"({suffix_content})\" - not in video title')}"
        )
        track = track[: m.start()].strip()
    return track


def get_youtube_title(url: str, cookiefile: str | None = None) -> tuple[str, str, str]:
    """Get the title of the YouTube video.

    Returns:
        A 3-tuple of (artist, track, video_title).  *video_title* is the
        raw title string from the YouTube video page, useful for
        cross-checking metadata from other sources (e.g. MusicBrainz).
    """

    ydl_opts = {
        "cookiefile": cookiefile,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(
            url, download=False  # We just want to extract the info
        )

    video_title = (result.get("title") or "").strip()
    artist = (result.get("artist") or "").strip()
    track = (result.get("track") or "").strip()
    if artist and track:
        return artist, strip_unmatched_suffixes(track, video_title), video_title
    channel = (result.get("channel") or "").strip()
    if "-" in video_title:
        parts = video_title.split("-", 1)
        return parts[0].strip(), parts[1].strip(), video_title
    return channel, video_title, video_title


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
    (artist, title, video_title) = get_youtube_title(input_url, cookiefile)

    # Get additional data for song
    song_info = search_musicbrainz(title, artist)

    # MusicBrainz may re-introduce suffixes that were already stripped from
    # the yt-dlp metadata (e.g. "(live)" when the video is not actually live).
    # Cross-check the MusicBrainz title against the video title as well.
    song_info.title = strip_unmatched_suffixes(song_info.title, video_title)

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
    audio_file_path, final_video_path, audio_ext, video_ext = separate_audio_video(
        video_with_audio_path, basename_without_ext, song_output
    )

    if song_info.cover_url is not None and song_info.cover_image_data is not None:
        cover_url = song_info.cover_url
        save_image(song_info.cover_image_data, basename_without_ext, song_output)
    else:
        cover_url = __download_youtube_thumbnail(
            input_url, basename_without_ext, song_output, cookiefile
        )

    return (
        basename_without_ext,
        song_output,
        audio_file_path,
        MediaInfo(
            artist=song_info.artist,
            title=song_info.title,
            year=song_info.year,
            genre=song_info.genres,
            cover_url=cover_url,
            video_url=input_url,
            audio_extension=audio_ext,
            video_extension=video_ext
        ),
    )
