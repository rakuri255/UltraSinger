"""BPM detection module"""

import numpy as np
import librosa
import soundfile as sf

from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted


def _pick_best_tempo(primary_bpm: float) -> float:
    """Apply half/double-tempo correction to a detected BPM value.

    Tempo detectors sometimes report double or half the actual tempo.
    This function generates half and double variants and picks the one
    in the typical song range (60-200 BPM) that is closest to 120 BPM
    (a common pop/rock tempo).

    Args:
        primary_bpm: The primary tempo estimate from librosa.

    Returns:
        The corrected tempo in BPM.
    """
    if primary_bpm <= 0:
        return 120.0  # Fallback

    # Generate half/double candidates
    candidates = [primary_bpm, primary_bpm / 2, primary_bpm * 2]

    # Prefer candidates in the typical song range (60-200 BPM)
    in_range = [c for c in candidates if 60 <= c <= 200]

    if in_range:
        # Pick the one closest to 120 BPM (common pop/rock tempo)
        return min(in_range, key=lambda x: abs(x - 120))

    # Nothing in range — use the primary as-is
    return primary_bpm


def get_bpm_from_data(data, sampling_rate):
    """Get real BPM from audio data.

    Uses librosa's default tempo estimation and applies half/double-tempo
    correction to avoid common detection errors where the tempo is
    reported as 2x or 0.5x the actual value.
    """
    onset_env = librosa.onset.onset_strength(y=data, sr=sampling_rate)
    wav_tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sampling_rate)
    primary_bpm = float(wav_tempo[0])

    # Apply half/double-tempo correction
    best_bpm = _pick_best_tempo(primary_bpm)

    print(
        f"{ULTRASINGER_HEAD} BPM is {blue_highlighted(str(round(best_bpm, 2)))}"
    )
    return best_bpm


def get_bpm_from_file(wav_file: str) -> float:
    """Get real BPM from audio file."""
    data, sampling_rate = sf.read(wav_file, dtype='float32')
    # Transpose if stereo to match librosa's expected format
    if len(data.shape) > 1:
        data = data.T
    # Convert to mono if stereo
    if data.ndim > 1:
        data = librosa.to_mono(data)
    return get_bpm_from_data(data, sampling_rate)
