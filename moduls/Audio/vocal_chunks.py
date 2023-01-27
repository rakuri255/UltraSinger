import os
import wave
import re

from moduls.os_helper import create_folder
from moduls.Ultrastar.ultrastar_converter import get_start_time_from_ultrastar, get_end_time_from_ultrastar
from pydub import AudioSegment


def convert_audio_to_mono_wav(input_file, output_file):
    """Convert audio to mono wav"""

    if '.mp3' in input_file:
        sound = AudioSegment.from_mp3(input_file)
    elif '.wav' in input_file:
        sound = AudioSegment.from_wav(input_file)

    sound = sound.set_channels(1)
    sound.export(output_file, format="wav")


class AudioManipulation:
    pass


def export_chunks_from_vosk_data(wf, vosk_words, output_folder_name):
    """Export vosk data as vocal chunks wav files"""

    sr, n_channels = wf.getparams()[2], wf.getparams()[0]

    for i in range(len(vosk_words)):
        start_byte = int(vosk_words[i].start * sr * n_channels)
        end_byte = int(vosk_words[i].end * sr * n_channels)

        chunk = get_chunk(end_byte, start_byte, wf)
        export_chunk_to_wav_file(chunk, output_folder_name, i, vosk_words[i].word, wf)


def export_chunks_from_ultrastar_data(audio_filename, ultrastar_data, folder_name):
    """Export ultrastar data as vocal chunks wav files"""

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

    clean_word = re.sub('[^A-Za-z0-9]+', '', word)
    # todo: Progress?
    # print(str(i) + ' ' + clean_word)
    with wave.open(os.path.join(folder_name, "chunk_{}_{}.wav".format(i, clean_word)),
                   "wb") as chunk_file:
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
