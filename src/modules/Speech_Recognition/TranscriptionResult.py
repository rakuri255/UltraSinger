from dataclasses import dataclass

from dataclasses_json import dataclass_json

from modules.Speech_Recognition.TranscribedData import TranscribedData


@dataclass_json
@dataclass
class TranscriptionResult:
    """Transcription result"""

    transcribed_data: list[TranscribedData]
    detected_language: str
