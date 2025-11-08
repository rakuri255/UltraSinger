"""Input validation utilities"""

import re
from urllib.parse import urlparse, parse_qs


def is_valid_youtube_url(url: str) -> bool:
    """Validate if URL is a valid YouTube URL"""
    youtube_regex = re.compile(
        r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/'
        r'(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})'
    )
    return bool(youtube_regex.match(url))


def extract_youtube_video_id(url: str) -> str | None:
    """Extract video ID from YouTube URL"""
    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]

    parsed_url = urlparse(url)
    if parsed_url.hostname in ("www.youtube.com", "youtube.com", "m.youtube.com"):
        if parsed_url.path == "/watch":
            query_params = parse_qs(parsed_url.query)
            return query_params.get("v", [None])[0]
        elif parsed_url.path.startswith("/embed/"):
            return parsed_url.path.split("/")[2]
        elif parsed_url.path.startswith("/v/"):
            return parsed_url.path.split("/")[2]

    return None


def is_valid_audio_file(filename: str) -> bool:
    """Check if file extension is a supported audio format"""
    supported_extensions = (".mp3", ".wav", ".ogg", ".m4a", ".flac")
    return filename.lower().endswith(supported_extensions)


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent directory traversal and other issues"""
    # Remove any directory path components
    filename = filename.split("/")[-1].split("\\")[-1]

    # Remove or replace problematic characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)

    # Limit length
    max_length = 200
    if len(filename) > max_length:
        name, ext = filename.rsplit(".", 1) if "." in filename else (filename, "")
        filename = name[:max_length - len(ext) - 1] + "." + ext if ext else name[:max_length]

    return filename
