"""BIC-based spectral boundary and acoustic changepoint detector (Phase 2).

Uses the Bayesian Information Criterion (BIC) to identify the exact changepoint
between consonant closure/frication and vowel harmonic voicing at 3ms resolution.
"""

import os
from typing import Optional, Tuple
import numpy as np
import librosa

_audio_cache: dict[str, Tuple[np.ndarray, int]] = {}


def _get_audio(audio_path: str, sr: int = 22050) -> Optional[Tuple[np.ndarray, int]]:
    if audio_path in _audio_cache:
        return _audio_cache[audio_path]
    if not os.path.isfile(audio_path):
        return None
    try:
        y, s_actual = librosa.load(audio_path, sr=sr, mono=True)
        _audio_cache[audio_path] = (y, s_actual)
        return y, s_actual
    except Exception:
        return None


def compute_bic_changepoint(features: np.ndarray, penalty: float = 1.0) -> Optional[Tuple[int, float]]:
    """Compute the optimal acoustic changepoint index in a feature matrix using BIC.

    Delta BIC:
      dBIC(t) = 0.5 * N * log|Sigma| - 0.5 * N1 * log|Sigma1| - 0.5 * N2 * log|Sigma2| - cost_penalty
    """
    N, D = features.shape
    if N < 8:
        return None

    cov_all = np.cov(features, rowvar=False)
    if D == 1:
        det_all = float(cov_all) + 1e-8
        log_det_all = np.log(det_all)
    else:
        sign, log_det_all = np.linalg.slogdet(cov_all + 1e-6 * np.eye(D))
        if sign <= 0:
            return None

    cost_penalty = 0.5 * penalty * (D + 0.5 * D * (D + 1)) * np.log(N)
    best_bic = 0.0
    best_idx = None

    for i in range(3, N - 3):
        cov1 = np.cov(features[:i], rowvar=False)
        cov2 = np.cov(features[i:], rowvar=False)
        if D == 1:
            det1 = float(cov1) + 1e-8
            det2 = float(cov2) + 1e-8
            bic = 0.5 * N * log_det_all - 0.5 * i * np.log(det1) - 0.5 * (N - i) * np.log(det2) - cost_penalty
        else:
            s1, ld1 = np.linalg.slogdet(cov1 + 1e-6 * np.eye(D))
            s2, ld2 = np.linalg.slogdet(cov2 + 1e-6 * np.eye(D))
            if s1 <= 0 or s2 <= 0:
                continue
            bic = 0.5 * N * log_det_all - 0.5 * i * ld1 - 0.5 * (N - i) * ld2 - cost_penalty

        if bic > best_bic:
            best_bic = bic
            best_idx = i

    if best_idx is not None:
        return best_idx, best_bic
    return None


def refine_onset_with_bic(
    audio_path: str,
    candidate_t: float,
    window_ms: float = 60.0,
    hop_ms: float = 3.0,
    sr: int = 22050,
    penalty: float = 1.0,
) -> float:
    """Refine an approximate onset using BIC spectral changepoint detection.

    Extracts high-resolution MFCCs in [candidate_t - window_ms, candidate_t + window_ms]
    and locates the point of maximum spectral transition.
    """
    res = _get_audio(audio_path, sr=sr)
    if res is None:
        return candidate_t
    y, sr_actual = res

    dur = len(y) / float(sr_actual)
    win_sec = window_ms / 1000.0
    t_start = max(0.0, candidate_t - win_sec)
    t_end = min(dur, candidate_t + win_sec)
    if (t_end - t_start) < 0.040:
        return candidate_t

    hop_length = int(round(sr_actual * (hop_ms / 1000.0)))
    n_fft = min(512, hop_length * 4)

    s_sample = int(round(t_start * sr_actual))
    e_sample = int(round(t_end * sr_actual))
    y_clip = y[s_sample:e_sample]
    if len(y_clip) < n_fft:
        return candidate_t

    try:
        mfcc = librosa.feature.mfcc(
            y=y_clip,
            sr=sr_actual,
            n_mfcc=6,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        rms = librosa.feature.rms(y=y_clip, hop_length=hop_length, frame_length=n_fft)
        min_cols = min(mfcc.shape[1], rms.shape[1])
        feats = np.vstack([mfcc[:, :min_cols], rms[:, :min_cols]]).T

        cp = compute_bic_changepoint(feats, penalty=penalty)
        if cp is not None:
            best_idx, score = cp
            cp_time = t_start + float(best_idx * hop_ms) / 1000.0
            # Only accept if within tolerance (e.g. within 40ms of candidate)
            if abs(cp_time - candidate_t) <= 0.040:
                return round(cp_time, 3)
    except Exception:
        pass

    return candidate_t

