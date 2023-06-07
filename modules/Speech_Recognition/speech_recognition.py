"""Docstring"""

import os

import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence

from modules.console_colors import ULTRASINGER_HEAD

# todo: Code from here: https://www.thepythoncode.com/article/using-speech-recognition-to-convert-speech-to-text-python


def print_text(wav_file):
    """Docstring"""

    # English speech!
    recognizer = sr.Recognizer()

    # open the file
    with sr.AudioFile(wav_file) as source:
        # listen for the data (load audio to memory)
        audio_data = recognizer.record(source)
        # recognize (convert from speech to text)
        text = recognizer.recognize_google(audio_data)

        print(text)


def get_large_audio_transcription(wav_file):
    """
    Splitting the large audio file into chunks
    and apply speech recognition on each of these chunks
    """
    # open the audio file using pydub
    sound = AudioSegment.from_wav(wav_file)

    # split audio sound where silence is 700 miliseconds or more and get chunks
    chunks = split_on_silence(
        sound,
        # experiment with this value for your target audio file
        min_silence_len=500,
        # adjust this per requirement
        silence_thresh=sound.dBFS - 14,
        # keep the silence for 1 second, adjustable as well
        keep_silence=500,
    )

    folder_name = "audio-chunks"
    # create a directory to store the audio chunks
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    whole_text = ""

    recognizer = sr.Recognizer()

    # process each chunk
    for i, audio_chunk in enumerate(chunks, start=1):
        # export audio chunk and save it in
        # the `folder_name` directory.
        chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
        audio_chunk.export(chunk_filename, format="wav")
        # recognize the chunk
        with sr.AudioFile(chunk_filename) as source:
            audio_listened = recognizer.record(source)
            # try converting it to text
            try:
                text = recognizer.recognize_google(audio_listened)
            except sr.UnknownValueError as error:
                print("Error:", str(error))
            else:
                text = f"{text.capitalize()}. "
                print(chunk_filename, ":", text)
                whole_text += text
    # return the text for all chunks detected
    return whole_text


def transcribe_audio(audio_file):
    """Docstring"""

    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
    try:
        transcript = recognizer.recognize_google(audio, show_all=True)
        start_time = transcript["result"][0]["alternative"][0]["words"][0][
            "startTime"
        ]
        end_time = transcript["result"][0]["alternative"][0]["words"][-1][
            "endTime"
        ]
        return (
            transcript["result"][0]["alternative"][0]["transcript"],
            start_time,
            end_time,
        )
    except sr.UnknownValueError:
        print(f"{ULTRASINGER_HEAD} Could not understand audio")
    except sr.RequestError as error:
        print(f"{ULTRASINGER_HEAD} Error with recognizing service; {error}")


class SpeechToText:
    """Docstring"""
