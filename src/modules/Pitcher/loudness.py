import warnings

import librosa
import numpy as np
from modules.Pitcher.core import CREPE_MODEL_SAMPLE_RATE

###############################################################################
# Constants
###############################################################################

WINDOW_SIZE = 1024
TIMES_DECIMAL_PLACES: int = 3
# Minimum decibel level
MIN_DB = -100.

# Reference decibel level
REF_DB = 20.

def set_confidence_to_zero_in_silent_regions(confidence, audio, threshold=-60, step_size=10, pad=True):
    # Don't modify in-place
    confidence = confidence[:]

    # Compute loudness
    loudness = a_weighted(audio, step_size, pad)

    # Threshold silence
    confidence[loudness < threshold] = 0.

    return confidence, loudness

def a_weighted(audio, step_size=10, pad=True):
    """Retrieve the per-frame loudness"""
    step_size_seconds = round(step_size / 1000, TIMES_DECIMAL_PLACES)
    steps_per_second = 1 / step_size_seconds
    hop_length = int(CREPE_MODEL_SAMPLE_RATE // steps_per_second)

    a_perceptual_weights = perceptual_weights()

    # Take stft
    stft = librosa.stft(audio,
                        n_fft=WINDOW_SIZE,
                        hop_length=hop_length,
                        win_length=WINDOW_SIZE,
                        center=pad,
                        pad_mode='constant')

    # Compute magnitude on db scale
    db = librosa.amplitude_to_db(np.abs(stft))

    # Apply A-weighting
    weighted = db + a_perceptual_weights

    # Threshold
    weighted[weighted < MIN_DB] = MIN_DB

    # Average over weighted frequencies
    return weighted.mean(axis=0)


def perceptual_weights():
    """A-weighted frequency-dependent perceptual loudness weights"""
    frequencies = librosa.fft_frequencies(sr=CREPE_MODEL_SAMPLE_RATE,
                                          n_fft=WINDOW_SIZE)

    # A warning is raised for nearly inaudible frequencies, but it ends up
    # defaulting to -100 db. That default is fine for our purposes.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return librosa.A_weighting(frequencies)[:, None] - REF_DB