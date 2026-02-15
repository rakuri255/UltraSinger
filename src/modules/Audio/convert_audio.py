"""Convert audio to other formats"""

import subprocess
import os
import librosa
import soundfile as sf

from modules.console_colors import ULTRASINGER_HEAD


def convert_audio_to_mono_wav(input_file_path: str, output_file_path: str) -> None:
    """Convert audio to mono wav"""
    print(f"{ULTRASINGER_HEAD} Converting audio for AI")
    y, sr = librosa.load(input_file_path, mono=True, sr=None)
    sf.write(output_file_path, y, sr)


def convert_audio_format(input_file_path: str, output_file_path: str) -> None:
    """Convert audio to the format specified by the output file extension using ffmpeg"""
    output_format = os.path.splitext(output_file_path)[1].lstrip('.')

    print(f"{ULTRASINGER_HEAD} Converting audio to {output_format}. -> {output_file_path}")
    # todo: makes it sense to reencode here? Its only used for Instrumental and Vocal
    # Use ffmpeg for audio conversion
    # -i: input file
    # -y: overwrite output file without asking
    # -loglevel error: only show errors
    # -q:a 0: best quality for VBR formats (mp3, ogg)
    # -codec:a copy would be fastest but only works if formats match
    cmd = [
        "ffmpeg",
        "-i", input_file_path,
        "-y",
        "-loglevel", "error",
        "-q:a", "0",
        output_file_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg audio conversion failed: {result.stderr}")
