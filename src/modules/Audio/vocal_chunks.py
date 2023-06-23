"""Vocal chunks module."""

import os
import re
import wave

from modules.console_colors import ULTRASINGER_HEAD
from modules.os_helper import create_folder
from modules.Ultrastar.ultrastar_converter import (
    get_end_time_from_ultrastar,
    get_start_time_from_ultrastar,
)
from modules.Ultrastar.ultrastar_txt import UltrastarTxtValue


class AudioManipulation:
    """Docstring"""


def export_chunks_from_transcribed_data(
    audio_filename: str, transcribed_data: [], output_folder_name: str
) -> None:
    """Export transcribed_data as vocal chunks wav files"""
    print(
        f"{ULTRASINGER_HEAD} Export transcribed data as vocal chunks wav files"
    )

    wave_file = wave.open(audio_filename, "rb")
    sample_rate, n_channels = wave_file.getparams()[2], wave_file.getparams()[0]

    for i, data in enumerate(transcribed_data):
        start_byte = int(data.start * sample_rate * n_channels)
        end_byte = int(data.end * sample_rate * n_channels)

        chunk = get_chunk(end_byte, start_byte, wave_file)
        export_chunk_to_wav_file(
            chunk, output_folder_name, i, data.word, wave_file
        )

    wave_file.close()


def export_chunks_from_ultrastar_data(
    audio_filename: str, ultrastar_data: UltrastarTxtValue, folder_name: str
) -> None:
    """Export ultrastar data as vocal chunks wav files"""
    print(f"{ULTRASINGER_HEAD} Export Ultrastar data as vocal chunks wav files")

    create_folder(folder_name)

    wave_file = wave.open(audio_filename, "rb")
    sample_rate, n_channels = wave_file.getparams()[2], wave_file.getparams()[0]

    for i, word in enumerate(ultrastar_data.words):
        start_time = get_start_time_from_ultrastar(ultrastar_data, i)
        end_time = get_end_time_from_ultrastar(ultrastar_data, i)

        start_byte = int(start_time * sample_rate * n_channels)
        end_byte = int(end_time * sample_rate * n_channels)

        chunk = get_chunk(end_byte, start_byte, wave_file)
        export_chunk_to_wav_file(
            chunk, folder_name, i, word, wave_file
        )


def export_chunk_to_wav_file(chunk, folder_name: str, i: int, word: str, wave_file) -> None:
    """Export vocal chunks to wav file"""

    clean_word = re.sub("[^A-Za-z0-9]+", "", word)
    # todo: Progress?
    # print(f"{str(i)} {clean_word}")
    with wave.open(
        os.path.join(folder_name, f"chunk_{i}_{clean_word}.wav"), "wb"
    ) as chunk_file:
        chunk_file.setparams(wave_file.getparams())
        chunk_file.writeframes(chunk)


def get_chunk(end_byte: int, start_byte: int, wave_file):
    """
    Gets the chunk from wave file.
    Returns chunk as n frames of audio, as a bytes object.
    """

    # todo: get out of position error message
    wave_file.setpos(start_byte)  # ({:.2f})
    chunk = wave_file.readframes(end_byte - start_byte)
    return chunk
