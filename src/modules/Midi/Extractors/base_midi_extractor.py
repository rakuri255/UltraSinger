"""Abstract Base Class for MIDI Extractor strategies."""
from abc import ABC, abstractmethod
from modules.Midi.MidiSegment import MidiSegment


class BaseMidiExtractor(ABC):
    """Abstract Strategy interface for converting vocal audio to MIDI segments."""

    @abstractmethod
    def extract(
        self,
        audio_path: str,
        **kwargs,
    ) -> list[MidiSegment]:
        """Extracts MIDI segments from audio.

        Args:
            audio_path: Path to the isolated vocal audio file.
            **kwargs: Extra parameters (e.g. bpm, device, transcribed_data).

        Returns:
            list[MidiSegment]: Extracted MIDI segments.
        """
        pass

