"""Pitcher module"""
import numpy as np

from scipy.io import wavfile
from swift_f0 import SwiftF0

from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted
from modules.Pitcher.pitched_data import PitchedData

_swift_f0_detector = None

def _get_detector():
    """Lazy initialize SwiftF0 detector"""
    global _swift_f0_detector
    if _swift_f0_detector is None:
        # Initialize for general music/speech (wide frequency range) fmin=46.875, fmax=2093.75
        # fixme: is this correct?
        # For speech only: fmin=65, fmax=400
        _swift_f0_detector = SwiftF0(fmin=65, fmax=400, confidence_threshold=0.9)
    return _swift_f0_detector


def get_pitch_with_file(
    filename: str
) -> PitchedData:
    """Pitch detection using SwiftF0"""

    print(
        f"{ULTRASINGER_HEAD} Pitching with {blue_highlighted('SwiftF0')}"
    )
    sample_rate, audio = wavfile.read(filename)

    # Convert stereo to mono if needed
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    # Normalize audio to float if needed
    if audio.dtype != np.float32 and audio.dtype != np.float64:
        audio = audio.astype(np.float32) / (2**15)

    return get_pitch_with_swift_f0(audio, sample_rate)


def get_pitch_with_swift_f0(
    audio: np.ndarray, sample_rate: int
) -> PitchedData:
    """Pitch detection using SwiftF0

    SwiftF0 processes audio at 16kHz with 256-sample hop size internally.
    Returns frames at approximately 62.5 ms intervals.
    """
    detector = _get_detector()

    # Detect pitch
    result = detector.detect_from_array(audio, sample_rate)

    # Convert to PitchedData format
    times = [float(t) for t in result.timestamps]
    frequencies = [float(f) for f in result.pitch_hz]
    confidence = [float(c) for c in result.confidence]

    return PitchedData(times, frequencies, confidence)


def get_pitched_data_with_high_confidence(
    pitched_data: PitchedData, threshold=0.4
) -> PitchedData:
    """Get frequency with high confidence"""
    new_pitched_data = PitchedData([], [], [])
    for i, conf in enumerate(pitched_data.confidence):
        if conf > threshold:
            new_pitched_data.times.append(pitched_data.times[i])
            new_pitched_data.frequencies.append(pitched_data.frequencies[i])
            new_pitched_data.confidence.append(pitched_data.confidence[i])

    return new_pitched_data


class Pitcher:
    """Docstring"""
