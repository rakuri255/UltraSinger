import whisper_timestamped as whisper
import json


def transcribe(audio):
    audio = whisper.load_audio(audio)
    model = whisper.load_model("tiny", device="cpu")
    result = whisper.transcribe(model, audio, language="en")

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return json.dumps(result, indent=2, ensure_ascii=False)

