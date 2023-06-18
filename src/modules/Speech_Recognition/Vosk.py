"""Vosk speech recognition module."""

import json
import wave

from vosk import KaldiRecognizer, Model

from src.modules.console_colors import ULTRASINGER_HEAD
from src.modules.Speech_Recognition.TranscribedData import TranscribedData

# todo: Rename to Transcoder?


def transcribe_with_vosk(audio_filename: str, model_path: str) -> list[TranscribedData]:
    """Transcribe with vosk"""
    # Code from here: https://towardsdatascience.com/speech-recognition-with-timestamps-934ede4234b2
    print(
        f"{ULTRASINGER_HEAD} Transcribing {audio_filename} with vosk and model {model_path}"
    )

    model = Model(model_path)
    wave_file = wave.open(audio_filename, "rb")
    recognizer = KaldiRecognizer(model, wave_file.getframerate())

    recognizer.SetWords(True)

    # get the list of JSON dictionaries
    results = []
    # recognize speech using vosk model
    while True:
        data = wave_file.readframes(4000)
        if len(data) == 0:
            break
        if recognizer.AcceptWaveform(data):
            part_result = json.loads(recognizer.Result())
            results.append(part_result)
    wave_file.close()  # close audiofile
    part_result = json.loads(recognizer.FinalResult())
    results.append(part_result)

    # convert list of JSON dictionaries to list of 'Word' objects
    transcribed_data = []
    for sentence in results:
        if len(sentence) == 1:
            # sometimes there are bugs in recognition
            # and it returns an empty dictionary
            # {'text': ''}
            continue
        for obj in sentence["result"]:
            vtd = TranscribedData(obj)  # create custom Word object
            vtd.word = vtd.word + " "
            transcribed_data.append(vtd)  # and add it to list

    # Todo: remove silent part from each word

    # output to the screen
    # todo: progress?
    # for word in vosk_transcribed_data:
    #    print(word.to_string())

    return transcribed_data


class SpeechToText:
    """Docstring"""
