"""Abstract Base Class for Vocal Separation strategies."""
from abc import ABC, abstractmethod


class BaseVocalSeparator(ABC):
    """Abstract Strategy interface for separating vocals and instrumental tracks."""

    @abstractmethod
    def separate(
        self,
        input_audio_path: str,
        cache_dir: str,
        device: str = "cpu",
        model_name: str = None,
    ) -> tuple[str, str]:
        """Separates vocals and instrumentals from input audio.

        Args:
            input_audio_path: Path to the input audio/video file.
            cache_dir: Destination cache folder.
            device: Computing device ('cpu' or 'cuda').
            model_name: Optional model identifier.

        Returns:
            tuple[str, str]: Absolute paths to (vocals.wav, no_vocals.wav).
        """
        pass
