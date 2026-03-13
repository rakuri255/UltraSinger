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
    low: float = 50.0,
    high: float = 500.0,
    target: float | None = None,
) -> float:
    """Apply tempo-ratio correction to a detected BPM value.

    Tempo detectors sometimes report a multiple or fraction of the
    actual tempo.  This function generates candidates by multiplying
    the detected value with a set of common musical ratios (half,
    double, third, quarter, etc.) and picks the candidate in the
    expected range that is closest to a reference value.

    The default range ``[50, 500]`` is intentionally wide because
    UltraStar uses BPM as a **timing-resolution parameter**, not as
    musical tempo.  Higher BPM means a finer beat grid and more
    precise note placement.  The ``get_multiplier()`` function in
    ``ultrastar_writer.py`` already compensates for low BPM values,
    so aggressive downscaling toward a "typical" tempo is harmful.

    If the detected tempo already lies inside ``[low, high]`` it is
    returned unchanged -- ratio correction only applies to values
    outside the expected range.  This prevents legitimate tempos
    from being rescaled.

    When *target* is ``None`` (default), the detected *primary_bpm*
    itself is used as reference.  This means: when correction IS
    needed, the candidate closest to the original value is picked,
    avoiding any bias toward a particular tempo.

    When two candidates are equally close to the reference, the one
    derived from the simpler ratio wins (identity > half/double >
    third, ...).

    Args:
        primary_bpm: The primary tempo estimate from librosa.
        low: Lower bound of the acceptable BPM range (inclusive).
        high: Upper bound of the acceptable BPM range (inclusive).
        target: Reference tempo for candidate selection.  When
            ``None`` (default), the *primary_bpm* itself is used,
            so correction picks the value closest to the detected
            tempo.  Pass an explicit value (e.g. ``120.0``) to bias
            toward a specific tempo centre.

    Returns:
        The corrected tempo in BPM.
    """
    if primary_bpm <= 0:
        return 120.0  # Fallback for invalid input

    # Resolve effective target: default to the detected value itself
    effective_target = target if target is not None else primary_bpm

    # Already in range -- trust the detector and return as-is.
    # Ratio correction should only kick in for out-of-range values;
    # otherwise legitimate fast/slow songs get their tempo rescaled
    # which breaks all downstream beat-length calculations.
    if low <= primary_bpm <= high:
        return primary_bpm

    # Generate candidates from all ratios, preserving ratio order
    candidates = [primary_bpm * r for r in _TEMPO_RATIOS]

    # Prefer candidates in the acceptable range
    in_range = [c for c in candidates if low <= c <= high]

    if in_range:
        # Pick the one closest to the effective target.
        # Because _TEMPO_RATIOS is ordered by simplicity and min()
        # is stable (returns the first minimum), simpler ratios win
        # ties automatically.
        return min(in_range, key=lambda x: abs(x - effective_target))

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
