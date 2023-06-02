class TranscribedData:
    """Transcribed data from json file"""

    def __init__(self, transcribed_json):
        # Vosk = conf, Whisper = confidence
        self.conf = transcribed_json.get(
            "conf", transcribed_json.get("confidence", None)
        )
        # Vosk = word, Whisper = text
        self.word = transcribed_json.get("word", transcribed_json.get("text", None))
        self.end = transcribed_json["end"]
        self.start = transcribed_json["start"]
        self.is_hyphen = None
