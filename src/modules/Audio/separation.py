"""Separate vocals from audio"""
import os
import shlex

import demucs.separate

from modules.Audio.convert_audio import convert_audio_to_mono_wav

from modules.console_colors import (
    ULTRASINGER_HEAD,
    blue_highlighted,
    red_highlighted, green_highlighted,
)
from modules.os_helper import check_file_exists


def separate_audio(input_file_path: str, output_folder: str, device="cpu") -> None:
    """Separate vocals from audio with demucs."""

    print(
        f"{ULTRASINGER_HEAD} Separating vocals from audio with {blue_highlighted('demucs')} and {red_highlighted(device)} as worker."
    )

    demucs.separate.main(shlex.split(f'--two-stems vocals -d {device} --float32 --out "{os.path.join(output_folder, "separated")}" "{input_file_path}"'))


def separate_vocal_from_audio(cache_folder_path: str,
                              ultrastar_audio_input_path: str,
                              processing_audio_path: str,
                              use_separated_vocal: bool,
                              create_karaoke: bool,
                              pytorch_device: str,
                              skip_cache: bool = False) -> str:
    """Separate vocal from audio"""
    demucs_output_folder = os.path.splitext(os.path.basename(ultrastar_audio_input_path))[0]
    audio_separation_path = os.path.join(cache_folder_path, "separated", "htdemucs", demucs_output_folder)

    vocals_path = os.path.join(audio_separation_path, "vocals.wav")
    instrumental_path = os.path.join(audio_separation_path, "no_vocals.wav")
    if use_separated_vocal or create_karaoke:
        cache_available = check_file_exists(vocals_path) and check_file_exists(instrumental_path)
        if skip_cache or not cache_available:
            separate_audio(ultrastar_audio_input_path, cache_folder_path, pytorch_device)
        else:
            print(f"{ULTRASINGER_HEAD} {green_highlighted('cache')} reusing cached separated vocals")

    if use_separated_vocal:
        input_path = vocals_path
    else:
        input_path = ultrastar_audio_input_path

    convert_audio_to_mono_wav(input_path, processing_audio_path)

    return audio_separation_path
