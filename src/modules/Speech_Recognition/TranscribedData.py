"""Docstring"""


class TranscribedData:
    """Transcribed data from json file"""

    def __init__(self, transcribed_json = None):
        if transcribed_json is None:
            return
        # Vosk = conf, Whisper = confidence
        self.conf = transcribed_json.get(
            "conf", transcribed_json.get("confidence", None)
        )
        # Vosk = word, Whisper = text
        self.word = transcribed_json.get(
            "word", transcribed_json.get("text", None)
        )
        self.end = transcribed_json.get("end", None)
        self.start = transcribed_json.get("start", None)
        self.is_hyphen = transcribed_json.get("is_hyphen", None)
        self.is_word_end = transcribed_json.get("is_word_end", True)
