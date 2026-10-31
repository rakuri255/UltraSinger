"""Stereo vocal spatial analysis module for discriminating center lead vocals vs side-panned backing vocals / harmonies.

Uses Interaural Cross-Correlation (IACC) and Center-to-Side Energy Ratio (CSR) to classify
lead vocals vs backing choir harmonies and enhance pitch tracking stability.
"""

import os
from typing import Dict, Any, Tuple
import numpy as np
import librosa


def compute_stereo_vocal_spatial_metrics(
    audio_path: str,
    start_time: float = 0.0,
    duration: float | None = None,
    sr: int = 22050,
) -> Dict[str, Any]:
    """Calculate spatial stereo metrics (IACC and CSR) for a segment of vocal audio.

    Args:
        audio_path: Path to vocal audio file (WAV or MP3).
        start_time: Start offset in seconds.
        duration: Duration to analyze in seconds. None analyzes up to 10 seconds.
        sr: Sample rate for analysis.

    Returns:
        dict containing:
            - 'iacc': Interaural cross-correlation coefficient [0.0, 1.0].
            - 'csr_db': Center-to-Side energy ratio in decibels.
            - 'is_lead': True if signal is dominated by centered lead vocals.
            - 'is_backing': True if signal exhibits wide side-panned backing vocal characteristics.
            - 'side_energy_ratio': Ratio of side energy to total energy [0.0, 1.0].
    """
    if not audio_path or not os.path.isfile(audio_path):
        return {
            "iacc": 1.0,
            "csr_db": 30.0,
            "is_lead": True,
            "is_backing": False,
            "side_energy_ratio": 0.0,
        }

    try:
        dur_load = min(duration, 15.0) if duration is not None else 10.0
        y, _ = librosa.load(
            audio_path,
            sr=sr,
            mono=False,
            offset=max(0.0, start_time),
            duration=dur_load,
        )
    except Exception:
        return {
            "iacc": 1.0,
            "csr_db": 30.0,
            "is_lead": True,
            "is_backing": False,
            "side_energy_ratio": 0.0,
        }

    # If audio is mono (1D or single channel), it is centered by definition
    if y.ndim == 1 or y.shape[0] < 2:
        return {
            "iacc": 1.0,
            "csr_db": 30.0,
            "is_lead": True,
            "is_backing": False,
            "side_energy_ratio": 0.0,
        }

    L = y[0].astype(np.float64)
    R = y[1].astype(np.float64)

    if len(L) == 0 or len(R) == 0:
        return {
            "iacc": 1.0,
            "csr_db": 30.0,
            "is_lead": True,
            "is_backing": False,
            "side_energy_ratio": 0.0,
        }

    # 1. Mid-Side Energy Decomposition
    # Mid (center) M = (L + R) / sqrt(2)
    # Side (surround / panned) S = (L - R) / sqrt(2)
    M = (L + R) / np.sqrt(2.0)
    S = (L - R) / np.sqrt(2.0)

    e_m = float(np.sum(M**2))
    e_s = float(np.sum(S**2))
    e_total = e_m + e_s + 1e-9

    csr_db = float(10.0 * np.log10(e_m / (e_s + 1e-9))) if (e_s + 1e-9) > 0 else 30.0
    csr_db = float(np.clip(csr_db, -40.0, 40.0))
    side_energy_ratio = float(np.clip(e_s / e_total, 0.0, 1.0))

    # 2. Interaural Cross-Correlation Coefficient (IACC)
    # Search within standard human interaural time delay window (+/- 1 ms)
    denom = np.sqrt(np.sum(L**2) * np.sum(R**2)) + 1e-9
    if denom <= 1e-6:
        # One channel silent -> hard panned
        iacc = 0.0
    else:
        max_lag = max(1, int(0.001 * sr))
        norm_corr = np.correlate(L, R, mode="full") / denom
        center = len(L) - 1
        window = norm_corr[
            max(0, center - max_lag) : min(len(norm_corr), center + max_lag + 1)
        ]
        iacc = float(np.max(np.abs(window))) if len(window) > 0 else 1.0

    iacc = float(np.clip(iacc, 0.0, 1.0))

    # Decision Rule:
    # Lead vocals in studio productions are centered: IACC >= 0.75 and CSR >= 4.0 dB
    # Backing vocals / harmonies are panned wide or decorrelated: IACC < 0.70 or CSR < 3.0 dB
    is_lead = bool(iacc >= 0.75 and csr_db >= 4.0)
    is_backing = bool(not is_lead and (iacc < 0.70 or csr_db < 3.0))

    return {
        "iacc": iacc,
        "csr_db": csr_db,
        "is_lead": is_lead,
        "is_backing": is_backing,
        "side_energy_ratio": side_energy_ratio,
    }


def separate_lead_and_backing_audio(
    y_stereo: np.ndarray,
    sr: int = 22050,
) -> Tuple[np.ndarray, np.ndarray]:
    """Separate a 2-channel stereo audio signal into center lead vocal and side backing vocals.

    Applies Mid-Side spectral masking to isolate centered phonation from stereo harmonies.

    Args:
        y_stereo: 2D numpy array of shape (2, N).
        sr: Sample rate.

    Returns:
        tuple (y_lead_mono, y_backing_stereo)
    """
    if y_stereo.ndim == 1 or y_stereo.shape[0] < 2:
        # Mono signal is pure lead
        return y_stereo.copy(), np.zeros_like(y_stereo)

    L = y_stereo[0]
    R = y_stereo[1]

    M = (L + R) / np.sqrt(2.0)
    S = (L - R) / np.sqrt(2.0)

    # Spectral STFT masking for soft center extraction
    n_fft = 1024
    hop_length = 256
    S_M = librosa.stft(M, n_fft=n_fft, hop_length=hop_length)
    S_S = librosa.stft(S, n_fft=n_fft, hop_length=hop_length)

    mag_M = np.abs(S_M)
    mag_S = np.abs(S_S)

    # Center mask
    mask_center = mag_M / (mag_M + mag_S + 1e-9)
    mask_side = 1.0 - mask_center

    M_lead = librosa.istft(S_M * mask_center, hop_length=hop_length, length=len(M))
    S_back = librosa.istft(S_S * mask_side, hop_length=hop_length, length=len(S))

    # Reconstruct backing stereo
    L_back = (S_back) / np.sqrt(2.0)
    R_back = (-S_back) / np.sqrt(2.0)
    y_backing = np.stack([L_back, R_back], axis=0)

    return M_lead, y_backing
