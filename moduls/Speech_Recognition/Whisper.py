import whisper_timestamped as whisper
from moduls.Speech_Recognition.TranscribedData import TranscribedData
from moduls.Log import PRINT_ULTRASTAR
from moduls.Log import print_blue_highlighted_text, print_red_highlighted_text


def transcribe_with_whisper(audioPath, model, device="CPU"):
    print("{} Loading {} with model {} and {} as worker".format(PRINT_ULTRASTAR, print_blue_highlighted_text("whisper"),
                                                                print_blue_highlighted_text(model),
                                                                print_red_highlighted_text(device)))

    model = whisper.load_model(model, device=device)

    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audioPath)
    audio_30 = whisper.pad_or_trim(audio)

    print(f"{PRINT_ULTRASTAR} Start detecting language")

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio_30).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    language = max(probs, key=probs.get)

    print(f"{PRINT_ULTRASTAR} Detected language: {print_blue_highlighted_text(language)}")

    print("{} Transcribing {}".format(PRINT_ULTRASTAR, audioPath))
    results = whisper.transcribe(model, audio, language=language)

    transcribed_data = []

    for segment in results["segments"]:
        # todo:
        # if sentence != 'segments':
        #    continue
        # to class
        for obj in segment["words"]:
            vtd = TranscribedData(obj)  # create custom Word object
            vtd.word = vtd.word + ' '
            transcribed_data.append(vtd)  # and add it to list

    return transcribed_data, language
