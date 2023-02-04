import whisper_timestamped as whisper
from moduls.Speech_Recognition.TranscribedData import TranscribedData


def transcribe_with_whisper(audio, model):
    model = whisper.load_model(model, device="cpu")

    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio)
    audio_30 = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio_30).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    language = max(probs, key=probs.get)

    print(f"Detected language: {language}")

    results = whisper.transcribe(model, audio, language=language)

    transcribed_data = []

    for segment in results["segments"]:
        # todo:
        #if sentence != 'segments':
        #    continue
        # to class
        for obj in segment["words"]:
            vtd = TranscribedData(obj)  # create custom Word object
            transcribed_data.append(vtd)  # and add it to list

    return transcribed_data

