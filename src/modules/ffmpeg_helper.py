"""FFmpeg helper module."""

import os
import shutil
import subprocess


def __add_ffmpeg_to_path(ffmpeg_path) -> None:
    os.environ["PATH"] = os.pathsep.join([os.environ["PATH"], ffmpeg_path])


def get_ffmpeg_and_ffprobe_paths():
    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")

    if not ffmpeg_path:
        raise FileNotFoundError("FFmpeg executable not found in the system path.")
    if not ffprobe_path:
        raise FileNotFoundError("FFprobe executable not found in the system path.")

    return ffmpeg_path, ffprobe_path


def is_ffmpeg_available(user_path: str = "") -> bool:
    if shutil.which("ffmpeg"):
        return True
    if (path := user_path) and path not in os.environ["PATH"]:
        __add_ffmpeg_to_path(path)
        if shutil.which("ffmpeg"):
            return True
    return False


def extract_audio(input_video_path: str, output_audio_path: str, ffmpeg_path: str) -> None:
    """Extract audio from video file in original format without re-encoding"""
    cmd = [
        ffmpeg_path,
        "-i",
        input_video_path,
        "-vn",  # No video
        "-acodec",
        "copy",  # Copy audio codec without re-encoding
        "-y",  # Overwrite output file if exists
        output_audio_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"FFmpeg audio extraction failed: {result.stderr}")


def remove_audio_from_video(input_video_path: str, output_video_path: str, ffmpeg_path: str) -> None:
    """Remove audio from video file without re-encoding video"""

    cmd = [
        ffmpeg_path,
        "-i",
        input_video_path,
        "-an",  # No audio
        "-vcodec",
        "copy",  # Copy video codec without re-encoding
        "-y",  # Overwrite output file if exists
        output_video_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"FFmpeg audio removal failed: {result.stderr}")
