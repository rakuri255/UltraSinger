"""FFmpeg helper module."""

import os
import shutil


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

