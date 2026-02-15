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


def extract_audio(input_video_path: str, output_audio_path: str) -> None:
    """Extract audio from video file in original format without re-encoding"""
    ffmpeg_path, _ = get_ffmpeg_and_ffprobe_paths()

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


def remove_audio_from_video(input_video_path: str, output_video_path: str) -> None:
    """Remove audio from video file without re-encoding video"""
    ffmpeg_path, _ = get_ffmpeg_and_ffprobe_paths()

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


def is_video_file(file_path: str) -> bool:
    """Check if file contains video streams using ffprobe"""
    try:
        _, ffprobe_path = get_ffmpeg_and_ffprobe_paths()

        cmd = [
            ffprobe_path,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_type",
            "-of", "default=noprint_wrappers=1:nokey=1",
            file_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0 and result.stdout.strip() == "video"
    except Exception:
        return False


def get_audio_codec_and_extension(video_file_path: str) -> str:
    """
    Detect audio codec from video file and return codec name and appropriate file extension.
    """
    try:
        _, ffprobe_path = get_ffmpeg_and_ffprobe_paths()

        cmd = [
            ffprobe_path,
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=codec_name",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_file_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 or not result.stdout.strip():
            # Default to wav if detection fails
            return "wav"

        codec_name = result.stdout.strip()

        codec_to_extension = {
            "aac": "aac",
            "mp3": "mp3",
            "opus": "opus",
            "vorbis": "ogg",
            "flac": "flac",
            "pcm_s16le": "wav",
            "pcm_s24le": "wav",
            "pcm_s32le": "wav",
            "ac3": "ac3",
            "eac3": "eac3",
        }

        extension = codec_to_extension.get(codec_name, "wav")
        return extension

    except Exception:
        # Default to wav if detection fails
        return "wav"


def separate_audio_video(video_with_audio_path: str, basename_without_ext: str, output_folder: str) -> tuple[str, str, str, str]:
    """
    Separate audio and video from a video file.
    Automatically detects the audio codec and uses the appropriate file extension.
    """
    from modules.console_colors import ULTRASINGER_HEAD

    # Get original video file extension without the dot
    _, video_ext_with_dot = os.path.splitext(video_with_audio_path)
    video_ext = video_ext_with_dot.lstrip('.')

    # Detect audio codec and get appropriate extension
    audio_ext = get_audio_codec_and_extension(video_with_audio_path)

    print(f"{ULTRASINGER_HEAD} Extracting audio from video")
    audio_file_path = os.path.join(output_folder, f"{basename_without_ext}.{audio_ext}")
    extract_audio(video_with_audio_path, audio_file_path)

    print(f"{ULTRASINGER_HEAD} Creating video without audio")
    video_only_path = os.path.join(output_folder, f"{basename_without_ext}_video.{video_ext}")
    remove_audio_from_video(video_with_audio_path, video_only_path)

    # Remove original video with audio
    os.remove(video_with_audio_path)

    # Rename video without audio to final name
    final_video_path = os.path.join(output_folder, f"{basename_without_ext}.{video_ext}")
    os.rename(video_only_path, final_video_path)

    return audio_file_path, final_video_path, audio_ext, video_ext

