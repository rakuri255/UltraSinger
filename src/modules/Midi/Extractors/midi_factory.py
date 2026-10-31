"""Factory to instantiate MIDI Extractor strategies."""
from modules.Midi.Extractors.base_midi_extractor import BaseMidiExtractor
from modules.Midi.Extractors.basic_pitch_extractor import BasicPitchMidiExtractor
from modules.Midi.Extractors.some_midi_extractor import SomeMidiExtractor
from modules.Midi.Extractors.omnizart_midi_extractor import OmnizartMidiExtractor


class MidiExtractorFactory:
    """Factory for creating MIDI extractor instances based on strategy name."""

    _EXTRACTORS = {
        "basic_pitch": BasicPitchMidiExtractor,
        "basic-pitch": BasicPitchMidiExtractor,
        "basicpitch": BasicPitchMidiExtractor,
        "some": SomeMidiExtractor,
        "singing_oriented": SomeMidiExtractor,
        "omnizart": OmnizartMidiExtractor,
    }

    @classmethod
    def get_extractor(cls, name: str) -> BaseMidiExtractor:
        key = (name or "basic_pitch").lower().strip()
        extractor_class = cls._EXTRACTORS.get(key)
        if not extractor_class:
            valid_keys = ", ".join(cls.available_extractors())
            raise ValueError(
                f"Unknown MIDI extractor: '{name}'. Available extractors: {valid_keys}"
            )
        return extractor_class()

    @classmethod
    def available_extractors(cls) -> list[str]:
        return ["basic_pitch", "some", "omnizart"]

