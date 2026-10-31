"""Abstract Base Class for Pitch Extraction strategies."""
from abc import ABC, abstractmethod

from modules.Pitcher.pitched_data import PitchedData


class BasePitchExtractor(ABC):
    """Abstract Strategy interface for extracting fundamental pitch (F0) from vocal audio."""

    @abstractmethod
    def extract_pitch(self, audio_path: str, device: str = "cpu") -> PitchedData:
        """Extracts pitch contour from an audio file.

        Args:
            audio_path: Path to the clean vocal audio file.
            device: Computing device ('cpu' or 'cuda').

        Returns:
            PitchedData: Object containing times (float seconds), frequencies (Hz), and confidence.
        """
        pass
