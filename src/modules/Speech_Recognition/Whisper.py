"""Whisper Speech Recognition Module"""

import sys

import whisperx

from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted, red_highlighted
from modules.Speech_Recognition.TranscribedData import TranscribedData


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

    print(
        f"{ULTRASINGER_HEAD} Loading {blue_highlighted('whisper')} with model {blue_highlighted(model)} and {red_highlighted(device)} as worker"
    )
    if model_name is not None:
        print(f"{ULTRASINGER_HEAD} using alignment model {blue_highlighted(model_name)}")

    if compute_type is None:
        compute_type = "float16" if device == "cuda" else "int8"

    loaded_whisper_model = whisperx.load_model(
        model, language=language, device=device, compute_type=compute_type
    )

    audio = whisperx.load_audio(audio_path)

    print(f"{ULTRASINGER_HEAD} Transcribing {audio_path}")

    result = loaded_whisper_model.transcribe(audio, batch_size=batch_size, language=language)

    detected_language = result["language"]
    if language is None:
        language = detected_language

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

    return transcribed_data, detected_language


def convert_to_transcribed_data(result_aligned):
    transcribed_data = []
    for segment in result_aligned["segments"]:
        for obj in segment["words"]:
            vtd = TranscribedData(obj)  # create custom Word object
            vtd.word = vtd.word + " "  # add space to end of word
            if len(obj) < 4:
                previous = transcribed_data[-1]
                if not previous:
                    previous.end = 0
                    previous.end = ""
                vtd.start = previous.end + 0.1
                vtd.end = previous.end + 0.2
                msg = f'Error: There is no timestamp for word:  {obj["word"]}. ' \
                      f'Fixing it by placing it after the previous word: {previous.word}. At start: {vtd.start} end: {vtd.end}. Fix it manually!'
                print(f"{red_highlighted(msg)}")
            transcribed_data.append(vtd)  # and add it to list
    return transcribed_data
