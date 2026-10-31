"""SwiftF0 Pitch Extraction Strategy."""
import numpy as np
from scipy.io import wavfile

from modules.console_colors import (
    ULTRASINGER_HEAD,
    blue_highlighted,
    red_highlighted,
)
from modules.Pitcher.Extractors.base_pitcher import BasePitchExtractor
from modules.Pitcher.pitched_data import PitchedData


class SwiftF0Pitcher(BasePitchExtractor):
    """Fast, lightweight neural pitch detector using SwiftF0."""

    def __init__(self, fmin: float = 46.875, fmax: float = 2093.75, confidence_threshold: float = 0.9):
        self.fmin = fmin
        self.fmax = fmax
        self.confidence_threshold = confidence_threshold
        self._detector = None

    def _get_detector(self):
        if self._detector is None:
            from swift_f0 import SwiftF0
            self._detector = SwiftF0(
                fmin=self.fmin,
                fmax=self.fmax,
                confidence_threshold=self.confidence_threshold,
            )
        return self._detector

    def extract_pitch(self, audio_path: str, device: str = "cpu") -> PitchedData:
        print(f"{ULTRASINGER_HEAD} Pitching with {blue_highlighted('SwiftF0')}")

        sample_rate, audio = wavfile.read(audio_path)

        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        if audio.dtype == np.uint8:
            audio = (audio.astype(np.float32) - 128.0) / 128.0
        elif audio.dtype in [np.int16, np.int32, np.int64]:
            dtype_info = np.iinfo(audio.dtype)
            max_val = max(abs(dtype_info.min), abs(dtype_info.max))
            audio = audio.astype(np.float32) / float(max_val)
        elif audio.dtype == np.float64:
            audio = audio.astype(np.float32)
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32) / 32768.0

        detector = self._get_detector()
        result = detector.detect_from_array(audio, sample_rate)

        times = [float(t) for t in result.timestamps]
        frequencies = [float(f) for f in result.pitch_hz]
        confidence = [float(c) for c in result.confidence]

        return PitchedData(times, frequencies, confidence)
