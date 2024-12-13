"""Whisper Speech Recognition Module"""

import sys

import torch
import whisperx
from enum import Enum
from torch.cuda import OutOfMemoryError

from modules.Speech_Recognition.TranscriptionResult import TranscriptionResult
from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted, red_highlighted
from modules.Speech_Recognition.TranscribedData import TranscribedData, from_whisper

#Addition for numbers to words
import re
import ast
from num2words import num2words

#Addition for numbers to words
re_split_preserve_space = re.compile(r'(\d+|\W+|\w+)')


MEMORY_ERROR_MESSAGE = f"{ULTRASINGER_HEAD} {blue_highlighted('whisper')} ran out of GPU memory; reduce --whisper_batch_size or force usage of cpu with --force_cpu"

class WhisperModel(Enum):
    """Whisper model"""
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE_V1 = "large-v1"
    LARGE_V2 = "large-v2"
    LARGE_V3 = "large-v3"

#Addition for numbers to words (Using previous code from louispan in PR#135)
def number_to_words(line,language='en'):
    # https://github.com/m-bain/whisperX
    # Transcript words which do not contain characters in the alignment models dictionary e.g. "2014." or "£13.60" cannot be aligned and therefore are not given a timing.
    # Therefore, convert numbers to words
    out_tokens = []
    in_tokens = re_split_preserve_space.findall(line)
    for token in in_tokens:
        try:
            num = ast.literal_eval(token)
            try:
                out_tokens.append(num2words(num, lang=language))
            except NotImplementedError:
                print(
                    f"{ULTRASINGER_HEAD} {red_highlighted('Error:')} Unknown language for number transcription. Keeping number as numeric characters for line: {line}, token: {token}"
                )
        except Exception:
            out_tokens.append(token)
    return ''.join(out_tokens) 

def transcribe_with_whisper(
    audio_path: str,
    model: WhisperModel,
    device="cpu",
    alignment_model: str = None,
    batch_size: int = 16,
    compute_type: str = None,
    language: str = None,
    keep_numbers: bool = False,
) -> TranscriptionResult:
    """Transcribe with whisper"""

    # Info: Regardless of the audio sampling rate used in the original audio file, whisper resample the audio signal to 16kHz (via ffmpeg). So the standard input from (44.1 or 48 kHz) should work.

    print(
        f"{ULTRASINGER_HEAD} Loading {blue_highlighted('whisper')} with model {blue_highlighted(model.value)} and {red_highlighted(device)} as worker"
    )
    if alignment_model is not None:
        print(f"{ULTRASINGER_HEAD} using alignment model {blue_highlighted(alignment_model)}")

    if compute_type is None:
        compute_type = "float16" if device == "cuda" else "int8"

    try:
        torch.cuda.empty_cache()
        loaded_whisper_model = whisperx.load_model(
            model.value, language=language, device=device, compute_type=compute_type
        )

        audio = whisperx.load_audio(audio_path)

        print(f"{ULTRASINGER_HEAD} Transcribing {audio_path}")

        result = loaded_whisper_model.transcribe(
            audio, batch_size=batch_size, language=language
        )

        detected_language = result["language"]
        if language is None:
            language = detected_language

        # load alignment model and metadata
        try:
            model_a, metadata = whisperx.load_align_model(
                language_code=language, device=device, model_name=alignment_model
            )
        except ValueError as ve:
            print(
                f"{red_highlighted(f'{ve}')}"
                f"\n"
                f"{ULTRASINGER_HEAD} {red_highlighted('Error:')} Unknown language. "
                f"Try add it with --align_model [huggingface]."
            )
            raise ve

        #Addition for numbers to words (Using previous code from louispan in PR#135)
        if keep_numbers == False: 
            for obj in result["segments"]:
                obj["text"] = number_to_words(obj["text"],language)

        # align whisper output
        result_aligned = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )

        transcribed_data = convert_to_transcribed_data(result_aligned)

        return TranscriptionResult(transcribed_data, detected_language)
    except ValueError as value_error:
        if (
            "Requested float16 compute type, but the target device or backend do not support efficient float16 computation."
            in str(value_error.args[0])
        ):
            print(value_error)
            print(
                f"{ULTRASINGER_HEAD} Your GPU does not support efficient float16 computation; run UltraSinger with '--whisper_compute_type int8'"
            )

        raise value_error
    except OutOfMemoryError as oom_exception:
        print(oom_exception)
        print(MEMORY_ERROR_MESSAGE)
        raise oom_exception
    except Exception as exception:
        if "CUDA failed with error out of memory" in str(exception.args[0]):
            print(exception)
            print(MEMORY_ERROR_MESSAGE)
        raise exception


def convert_to_transcribed_data(result_aligned):
    transcribed_data = []
    for segment in result_aligned["segments"]:
        for obj in segment["words"]:
            vtd = from_whisper(obj)  # create custom Word object
            vtd.word = vtd.word + " "  # add space to end of word
            if len(obj) < 4:
                #Addition for numbers to words (Using previous code from louispan in PR#135)
                if len(transcribed_data) == 0: # if the first word doesn't have any timing data
                    vtd.start = 0.0
                    vtd.end = 0.1
                    msg = f'Error: There is no timestamp for word: "{obj["word"]}". ' \
                        f'Fixing it by placing it at beginning. At start: {vtd.start} end: {vtd.end}. Fix it manually!'
                else:
                    previous = transcribed_data[-1] if len(transcribed_data) != 0 else TranscribedData()
                    vtd.start = previous.end + 0.1
                    vtd.end = previous.end + 0.2
                    msg = f'Error: There is no timestamp for word: "{obj["word"]}". ' \
                          f'Fixing it by placing it after the previous word: "{previous.word}". At start: {vtd.start} end: {vtd.end}. Fix it manually!'
                print(f"{red_highlighted(msg)}")
            transcribed_data.append(vtd)  # and add it to list
    return transcribed_data
