import whisperx

from moduls.Log import (PRINT_ULTRASTAR, print_blue_highlighted_text,
                        print_red_highlighted_text)
from moduls.Speech_Recognition.TranscribedData import TranscribedData


def transcribe_with_whisper(audio_path, model, device="cpu"):
    print(f"{PRINT_ULTRASTAR} Loading {print_blue_highlighted_text('whisper')} with model {print_blue_highlighted_text(model)} and {print_red_highlighted_text(device)} as worker")

    batch_size = 16  # reduce if low on GPU mem
    compute_type = "float16" if device == "cuda" else "int8"  # change to "int8" if low on GPU mem (may reduce accuracy)

    # transcribe with original whisper
    loaded_whisper_model = whisperx.load_model(model, device=device, compute_type=compute_type)
    audio = whisperx.load_audio(audio_path)

    print(f"{PRINT_ULTRASTAR} Transcribing {audio_path}")

    result = loaded_whisper_model.transcribe(audio, batch_size=batch_size)
    language = result["language"]

    # load alignment model and metadata
    model_a, metadata = whisperx.load_align_model(language_code=language, device=device)

    # align whisper output
    result_aligned = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    transcribed_data = []

    for segment in result_aligned["segments"]:
        for obj in segment["words"]:
            if len(obj) < 4:
                print(f"{print_red_highlighted_text('Error: Skipping Word {}, because of missing timings'.format(obj['word']))}")
                continue
            vtd = TranscribedData(obj)  # create custom Word object
            vtd.word = vtd.word + ' '
            transcribed_data.append(vtd)  # and add it to list

    return transcribed_data, language
