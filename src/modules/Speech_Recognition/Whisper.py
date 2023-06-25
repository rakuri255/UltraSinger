"""Whisper Speech Recognition Module"""

import whisperx
import sys

from modules.console_colors import (
    ULTRASINGER_HEAD,
    blue_highlighted,
    red_highlighted,
)
from modules.Speech_Recognition.TranscribedData import TranscribedData


def transcribe_with_whisper(audio_path: str, model: str, device="cpu", model_name: str = None) -> (
list[TranscribedData], str):
    """Transcribe with whisper"""

    print(
        f"{ULTRASINGER_HEAD} Loading {blue_highlighted('whisper')} with model {blue_highlighted(model)} and {red_highlighted(device)} as worker"
    )
    if model_name is not None:
        print(f"{ULTRASINGER_HEAD} using alignment model {blue_highlighted(model_name)}")

    batch_size = 16  # reduce if low on GPU mem
    compute_type = (
        "float16" if device == "cuda" else "int8"
    )  # change to "int8" if low on GPU mem (may reduce accuracy)

    loaded_whisper_model = whisperx.load_model(
        model, device=device, compute_type=compute_type
    )

    audio = whisperx.load_audio(audio_path)

    print(f"{ULTRASINGER_HEAD} Transcribing {audio_path}")

    result = loaded_whisper_model.transcribe(audio, batch_size=batch_size)
    language = result["language"]

    # load alignment model and metadata
    try:
        model_a, metadata = whisperx.load_align_model(language_code=language, device=device, model_name=model_name)
    except ValueError as ve:
        print(f"{red_highlighted(f'{ve}')}"
              f"\n"
              f"{ULTRASINGER_HEAD} {red_highlighted('Error:')} Unknown language. "
              f"Try add it with --align_model [hugingface].")
        sys.exit(0)

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

    return transcribed_data, language


def convert_to_transcribed_data(result_aligned):
    transcribed_data = []
    for segment in result_aligned["segments"]:
        for obj in segment["words"]:
            if len(obj) < 4:
                print(
                    f"{red_highlighted('Error: Skipping Word {}, because of missing timings'.format(obj['word']))}"
                )
                continue
            vtd = TranscribedData(obj)  # create custom Word object
            vtd.word = vtd.word + " "  # add space to end of word
            transcribed_data.append(vtd)  # and add it to list
    return transcribed_data
