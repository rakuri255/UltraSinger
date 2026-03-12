"""BPM detection module"""

import numpy as np
import librosa
import soundfile as sf

from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted


# Candidate ratios ordered by musical simplicity.
# Simpler ratios (identity, half, double) are listed first so that
# they win ties when two candidates are equally close to the target.
_TEMPO_RATIOS: list[float] = [
    1.0,    # identity
    0.5,    # half-tempo
    2.0,    # double-tempo
    1 / 3,  # third-tempo  (e.g. 3/4 waltz detected as compound)
    3.0,    # triple-tempo
    0.25,   # quarter-tempo
    4.0,    # quadruple-tempo
    2 / 3,  # two-thirds  (e.g. compound metre confusion)
    1.5,    # dotted-half  (shuffle / swing feel)
    0.75,   # three-quarters
]


def _pick_best_tempo(
    primary_bpm: float,
    low: float = 60.0,
    high: float = 200.0,
    target: float = 120.0,
) -> float:
    """Apply tempo-ratio correction to a detected BPM value.

    Tempo detectors sometimes report a multiple or fraction of the
    actual tempo.  This function generates candidates by multiplying
    the detected value with a set of common musical ratios (half,
    double, third, quarter, etc.) and picks the candidate in the
    expected range that is closest to *target* BPM.

    If the detected tempo already lies inside ``[low, high]`` it is
    returned unchanged -- ratio correction only applies to values
    outside the expected range.  This prevents legitimate fast or slow
    tempos (e.g. 180 BPM punk rock) from being rescaled.

    When two candidates are equally close to *target*, the one derived
    from the simpler ratio wins (identity > half/double > third, ...).

    Args:
        primary_bpm: The primary tempo estimate from librosa.
        low: Lower bound of the acceptable BPM range (inclusive).
        high: Upper bound of the acceptable BPM range (inclusive).
        target: Preferred tempo centre -- candidates closest to this
            value are selected.  120 BPM is a common pop/rock tempo.

    Returns:
        The corrected tempo in BPM.
    """
    if primary_bpm <= 0:
        return target  # Fallback

    # Already in range -- trust the detector and return as-is.
    # Ratio correction should only kick in for out-of-range values;
    # otherwise legitimate fast/slow songs get their tempo rescaled
    # which breaks all downstream beat-length calculations.
    if low <= primary_bpm <= high:
        return primary_bpm

    # Generate candidates from all ratios, preserving ratio order
    candidates = [primary_bpm * r for r in _TEMPO_RATIOS]

    # Prefer candidates in the typical song range
    in_range = [c for c in candidates if low <= c <= high]

    if in_range:
        # Pick the one closest to target BPM.
        # Because _TEMPO_RATIOS is ordered by simplicity and min()
        # is stable (returns the first minimum), simpler ratios win
        # ties automatically.
        return min(in_range, key=lambda x: abs(x - target))

    # Nothing in range -- use the primary as-is
    return primary_bpm


def get_bpm_from_data(data, sampling_rate):
    """Get real BPM from audio data.

    Uses librosa's default tempo estimation and applies tempo-ratio
    correction to avoid common detection errors where the tempo is
    reported as a multiple or fraction of the actual value.
    """
    onset_env = librosa.onset.onset_strength(y=data, sr=sampling_rate)
    wav_tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sampling_rate)
    primary_bpm = float(wav_tempo[0])

    # Apply tempo-ratio correction
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
