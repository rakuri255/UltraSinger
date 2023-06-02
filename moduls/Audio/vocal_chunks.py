import csv
import os
import re
import wave

import librosa
from pydub import AudioSegment

from moduls.Log import PRINT_ULTRASTAR
from moduls.os_helper import create_folder
from moduls.Ultrastar.ultrastar_converter import (
    get_end_time_from_ultrastar,
    get_start_time_from_ultrastar,
)


def convert_audio_to_mono_wav(input_file, output_file):
    """Convert audio to mono wav"""
    print(f"{PRINT_ULTRASTAR} Converting audio for AI")

    if ".mp3" in input_file:
        sound = AudioSegment.from_mp3(input_file)
    elif ".wav" in input_file:
        sound = AudioSegment.from_wav(input_file)
    else:
        raise ValueError("data format not supported")

    sound = sound.set_channels(1)
    sound.export(output_file, format="wav")


def convert_wav_to_mp3(input_file, output_file):
    sound = AudioSegment.from_wav(input_file)
    sound.export(output_file, format="mp3")


class AudioManipulation:
    pass


def export_chunks_from_transcribed_data(
    audio_filename, transcribed_data, output_folder_name
):
    """Export transcribed_data as vocal chunks wav files"""
    print(f"{PRINT_ULTRASTAR} Export transcribed data as vocal chunks wav files")

    wf = wave.open(audio_filename, "rb")
    sr, n_channels = wf.getparams()[2], wf.getparams()[0]

    for i in range(len(transcribed_data)):
        start_byte = int(transcribed_data[i].start * sr * n_channels)
        end_byte = int(transcribed_data[i].end * sr * n_channels)

        chunk = get_chunk(end_byte, start_byte, wf)
        export_chunk_to_wav_file(
            chunk, output_folder_name, i, transcribed_data[i].word, wf
        )

    wf.close()


def remove_silence_from_transcribtion_data(audio_path, transcribed_data):
    print(
        f"{PRINT_ULTRASTAR} Removing silent start and ending, from transcription data"
    )

    y, sr = librosa.load(audio_path, sr=None)

    for i in range(len(transcribed_data)):
        start_time = transcribed_data[i].start
        end_time = transcribed_data[i].end
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        chunk = y[start_sample:end_sample]

        # todo: why 5 works good? It should be 40db ?!?
        # max_dB = librosa.amplitude_to_db(chunk, ref=np.max)
        silence_threshold = 5
        onsets = librosa.effects.split(
            chunk, top_db=silence_threshold, frame_length=2048, hop_length=100
        )

        # Get the duration of the first and last silent intervals
        if len(onsets) > 0:
            first_silence = onsets[0][0]
            last_silence = len(chunk) - onsets[-1][1]

            first_silence_duration = librosa.samples_to_time(first_silence, sr=sr)
            last_silence_duration = librosa.samples_to_time(last_silence, sr=sr)
        else:
            first_silence_duration = 0
            last_silence_duration = 0

        transcribed_data[i].start = transcribed_data[i].start + first_silence_duration
        transcribed_data[i].end = transcribed_data[i].end - last_silence_duration

    return transcribed_data


def export_chunks_from_ultrastar_data(audio_filename, ultrastar_data, folder_name):
    """Export ultrastar data as vocal chunks wav files"""
    print(f"{PRINT_ULTRASTAR} Export Ultrastar data as vocal chunks wav files")

    create_folder(folder_name)

    wf = wave.open(audio_filename, "rb")
    sr, n_channels = wf.getparams()[2], wf.getparams()[0]

    for i in range(len(ultrastar_data.words)):
        start_time = get_start_time_from_ultrastar(ultrastar_data, i)
        end_time = get_end_time_from_ultrastar(ultrastar_data, i)

        start_byte = int(start_time * sr * n_channels)
        end_byte = int(end_time * sr * n_channels)

        chunk = get_chunk(end_byte, start_byte, wf)
        export_chunk_to_wav_file(chunk, folder_name, i, ultrastar_data.words[i], wf)


def export_chunk_to_wav_file(chunk, folder_name, i, word, wf):
    """Export vocal chunks to wav file"""

    clean_word = re.sub("[^A-Za-z0-9]+", "", word)
    # todo: Progress?
    # print(str(i) + ' ' + clean_word)
    with wave.open(
        os.path.join(folder_name, f"chunk_{i}_{clean_word}.wav"), "wb"
    ) as chunk_file:
        chunk_file.setparams(wf.getparams())
        chunk_file.writeframes(chunk)


def get_chunk(end_byte, start_byte, wf):
    """
    Gets the chunk from wave file.
    Returns chunk as n frames of audio, as a bytes object.
    """

    # todo: get out of position error message
    wf.setpos(start_byte)  # ({:.2f})
    chunk = wf.readframes(end_byte - start_byte)
    return chunk


def export_transcribed_data_to_csv(transcribed_data, filename):
    """Export transcribed data to csv"""
    print(f"{PRINT_ULTRASTAR} Exporting transcribed data to CSV")

    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        header = ["word", "start", "end", "confidence"]
        writer.writerow(header)
        for i in range(len(transcribed_data)):
            writer.writerow(
                [
                    transcribed_data[i].word,
                    transcribed_data[i].start,
                    transcribed_data[i].end,
                    transcribed_data[i].conf,
                ]
            )
