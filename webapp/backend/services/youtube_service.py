"""YouTube download service using yt-dlp"""

import asyncio
import logging
from pathlib import Path
from typing import Callable
import yt_dlp

logger = logging.getLogger(__name__)


async def download_youtube_audio(
    url: str,
    output_dir: Path,
    progress_callback: Callable[[str, float], None] | None = None
) -> tuple[Path, str]:
    """
    Download audio from YouTube URL

    Args:
        url: YouTube video URL
        output_dir: Directory to save the audio file
        progress_callback: Optional callback for progress updates (message, percentage)

    Returns:
        Tuple of (audio_file_path, video_title)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    def progress_hook(d):
        """Hook to track download progress"""
        if progress_callback:
            if d["status"] == "downloading":
                try:
                    percentage = (d.get("downloaded_bytes", 0) / d.get("total_bytes", 1)) * 100
                    progress_callback(f"Downloading: {percentage:.1f}%", percentage)
                except (KeyError, ZeroDivisionError):
                    pass
            elif d["status"] == "finished":
                progress_callback("Download completed, processing...", 100)

    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
        }],
        "outtmpl": str(output_dir / "%(title)s.%(ext)s"),
        "progress_hooks": [progress_hook] if progress_callback else [],
        "quiet": True,
        "no_warnings": True,
    }

    try:
        # Run yt-dlp in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        video_info, audio_path = await loop.run_in_executor(
            None,
            _download_sync,
            url,
            ydl_opts
        )

        title = video_info.get("title", "Unknown")
        logger.info(f"Downloaded: {title}")

        return Path(audio_path), title

    except Exception as e:
        logger.error(f"Failed to download from YouTube: {e}")
        raise ValueError(f"YouTube download failed: {str(e)}")


def _download_sync(url: str, ydl_opts: dict) -> tuple[dict, str]:
    """Synchronous download helper"""
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        # Get the actual output path
        audio_path = ydl.prepare_filename(info)
        # Change extension to wav
        audio_path = audio_path.rsplit(".", 1)[0] + ".wav"
        return info, audio_path


async def get_video_info(url: str) -> dict:
    """Get video information without downloading"""
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
    }

    try:
        loop = asyncio.get_event_loop()
        info = await loop.run_in_executor(
            None,
            _get_info_sync,
            url,
            ydl_opts
        )
        return {
            "title": info.get("title", "Unknown"),
            "duration": info.get("duration", 0),
            "uploader": info.get("uploader", "Unknown"),
        }
    except Exception as e:
        logger.error(f"Failed to get video info: {e}")
        raise ValueError(f"Failed to get video information: {str(e)}")


def _get_info_sync(url: str, ydl_opts: dict) -> dict:
    """Synchronous info extraction helper"""
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        return ydl.extract_info(url, download=False)
