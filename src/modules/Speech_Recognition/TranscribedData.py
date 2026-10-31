from dataclasses import dataclass

try:
    from dataclasses_json import dataclass_json
except Exception:
    def dataclass_json(cls):
        return cls


@dataclass_json
@dataclass
class TranscribedData:
    """Transcribed data from json file"""

    confidence: float = 0
    word: str = ""
    start: float = 0
    end: float = 0
    is_hyphen: bool = False
    is_word_end: bool = True
    is_line_end: bool = False
    chars: list[dict] = None


def from_whisper(whisper_dict) -> TranscribedData:
    transcribed_data = TranscribedData()
    if "score" in whisper_dict:
        transcribed_data.confidence = whisper_dict["score"]
    if "word" in whisper_dict:
        transcribed_data.word = whisper_dict["word"]
    if "start" in whisper_dict:
        transcribed_data.start = whisper_dict["start"]
    if "end" in whisper_dict:
        transcribed_data.end = whisper_dict["end"]
    if "chars" in whisper_dict:
        transcribed_data.chars = whisper_dict["chars"]
    return transcribed_data
