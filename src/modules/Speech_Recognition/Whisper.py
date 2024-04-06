"""Whisper Speech Recognition Module"""

import sys

import whisperx
from torch.cuda import OutOfMemoryError

from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted, red_highlighted
from modules.Speech_Recognition.TranscribedData import TranscribedData

import re
import ast
import inflect

re_split_preserve_space = re.compile(r'(\S+)')
inflect_engine = inflect.engine()

def any_number_to_words(line):
    # https://github.com/m-bain/whisperX
    # Transcript words which do not contain characters in the alignment models dictionary e.g. "2014." or "£13.60" cannot be aligned and therefore are not given a timing.
    # Therefore, convert numbers to words
    out_tokens = []
    in_tokens = re_split_preserve_space.split(line)
    for token in in_tokens:
        try:
            num = ast.literal_eval(token)
            out_tokens.append(inflect_engine.number_to_words(num))
        except Exception:
            out_tokens.append(token)
    return ''.join(out_tokens)

def transcribe_with_whisper(
    audio_path: str,
    model: str,
    device="cpu",
    model_name: str = None,
    batch_size: int = 16,
    compute_type: str = None,
    language: str = None,
) -> (list[TranscribedData], str):
    """Transcribe with whisper"""

    # Info: Regardless of the audio sampling rate used in the original audio file, whisper resample the audio signal to 16kHz (via ffmpeg). So the standard input from (44.1 or 48 kHz) should work.

    print(
        f"{ULTRASINGER_HEAD} Loading {blue_highlighted('whisper')} with model {blue_highlighted(model)} and {red_highlighted(device)} as worker"
    )
    if model_name is not None:
        print(f"{ULTRASINGER_HEAD} using alignment model {blue_highlighted(model_name)}")

    if compute_type is None:
        compute_type = "float16" if device == "cuda" else "int8"

    try:
        loaded_whisper_model = whisperx.load_model(
            model, language=language, device=device, compute_type=compute_type
        )
    except ValueError as value_error:
        if (
            "Requested float16 compute type, but the target device or backend do not support efficient float16 computation."
            in str(value_error.args[0])
        ):
            print(value_error)
            print(
                f"{ULTRASINGER_HEAD} Your GPU does not support efficient float16 computation; run UltraSinger with '--whisper_compute_type int8'"
            )
            sys.exit(1)

        raise value_error
    except OutOfMemoryError as oom_exception:
        print(oom_exception)
        print(
            f"{ULTRASINGER_HEAD} {blue_highlighted('whisper')} ran out of GPU memory; reduce --whisper_batch_size or force usage of cpu with --force_cpu"
        )
        sys.exit(1)

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
            language_code=language, device=device, model_name=model_name
        )
    except ValueError as ve:
        print(
            f"{red_highlighted(f'{ve}')}"
            f"\n"
            f"{ULTRASINGER_HEAD} {red_highlighted('Error:')} Unknown language. "
            f"Try add it with --align_model [huggingface]."
        )
        sys.exit(1)

    # convert any numbers to words so align will have timing
    for obj in result["segments"]:
        obj['text'] = any_number_to_words(obj['text'])

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

    return transcribed_data, detected_language


def convert_to_transcribed_data(result_aligned):
    transcribed_data = []
    for segment in result_aligned["segments"]:
        for obj in segment["words"]:
            vtd = TranscribedData(obj)  # create custom Word object
            vtd.word = vtd.word + " "  # add space to end of word
            if len(obj) < 4:
                if len(transcribed_data) == 0: # if the first word doesn't have any timing data
                    vtd.start = 0.0
                    vtd.end = 0.1
                    msg = f'Error: There is no timestamp for word: "{obj["word"]}". ' \
                        f'Fixing it by placing it at beginning. At start: {vtd.start} end: {vtd.end}. Fix it manually!'
                else:
                    previous = transcribed_data[-1]
                    if not previous:
                        previous.end = 0
                        previous.end = ""
                    vtd.start = previous.end + 0.1
                    vtd.end = previous.end + 0.2
                    msg = f'Error: There is no timestamp for word: "{obj["word"]}". ' \
                        f'Fixing it by placing it after the previous word: "{previous.word}". At start: {vtd.start} end: {vtd.end}. Fix it manually!'
                print(f"{red_highlighted(msg)}")
            transcribed_data.append(vtd)  # and add it to list
    return transcribed_data
