"""Onset snapping: refine syllable boundaries using spectral onset detection."""

import os
import re
from typing import Optional
import numpy as np
import scipy.signal
import librosa

from modules.console_colors import (
    ULTRASINGER_HEAD,
    blue_highlighted,
    green_highlighted,
    red_highlighted,
    cyan_highlighted,
)
from modules.Speech_Recognition.TranscribedData import TranscribedData

_onset_cache: dict[str, np.ndarray] = {}


def get_consonant_onset_lead(syllable_text: str) -> float:
    """Return lead time in seconds for the initial consonant before the vowel onset (Phase 5).

    Reproduces human UltraStar transcription convention where note begins with the consonant:
    - plosives (p, t, k, b, d, g, c, q): 30 ms
    - fricatives / affricates (s, f, z, j, v, h, x, sh, ch, th): 50 ms
    - nasals / liquids / glides (m, n, l, r, w, y): 20 ms
    - vowels: 0 ms
    """
    s = (syllable_text or "").strip().lower()
    if not s or s.startswith("~"):
        return 0.0
    s = re.sub(r"^[^a-záéíóúñ]+", "", s)
    if not s:
        return 0.0
    first = s[0]
    first_two = s[:2]
    if first in "aeiouyáéíóú":
        return 0.0
    if first_two in ("sh", "ch", "th", "ph") or first in "sfzjvhx":
        return 0.050
    if first in "ptkcbtdgq":
        return 0.030
    if first in "mnlrw":
        return 0.020
    return 0.025


def detect_vocal_onsets(
    audio_path: str,
    sr: int = 22050,
    delta: float = 0.07,
) -> np.ndarray:
    """Detect spectral onsets in vocal audio track using VOP & SuperFlux (Phase 5).

    Combines:
    - 300-3400 Hz formant band Hilbert envelope derivative (vowel onset point)
    - SuperFlux mel spectrogram-based onset strength (harmonic vocal attacks)
    - Backtracking to the nearest preceding energy local minimum
    """
    if audio_path in _onset_cache:
        return _onset_cache[audio_path]

    if not os.path.isfile(audio_path):
        print(f"{ULTRASINGER_HEAD} {red_highlighted('Warning:')} Audio file not found for onset detection: {audio_path}")
        return np.array([], dtype=float)

    try:
        y, sr_actual = librosa.load(audio_path, sr=sr, mono=True)
    except Exception as e:
        print(f"{ULTRASINGER_HEAD} {red_highlighted('Error loading audio for onsets:')} {e}")
        return np.array([], dtype=float)

    if len(y) == 0:
        return np.array([], dtype=float)

    # Harmonic-Percussive Source Separation (HPSS) to isolate clean harmonic vocal attacks
    try:
        y_harm = librosa.effects.harmonic(y)
    except Exception:
        y_harm = y

    # Compute onset strength envelope with SuperFlux (Böck & Widmer 2013)
    # max_size=3 applies local maximum filtering across frequency bins to suppress false vibrato onsets.
    onset_env = librosa.onset.onset_strength(
        y=y_harm,
        sr=sr_actual,
        hop_length=256,
        lag=2,
        max_size=3,
        aggregate=np.median,
    )

    # VOP: 300-3400 Hz formant band Hilbert envelope derivative
    try:
        nyq = sr_actual * 0.5
        sos = scipy.signal.butter(4, [max(10.0, 300.0 / nyq), min(0.99, 3400.0 / nyq)], btype="bandpass", output="sos")
        y_formant = scipy.signal.sosfilt(sos, y)
        env_formant = np.abs(scipy.signal.hilbert(y_formant))
        # Frame-wise max or mean matching hop_length=256
        n_frames = len(onset_env)
        pad_len = n_frames * 256 - len(env_formant)
        if pad_len > 0:
            env_formant = np.pad(env_formant, (0, pad_len))
        elif pad_len < 0:
            env_formant = env_formant[:n_frames * 256]
        framed_formant = env_formant.reshape((n_frames, 256)).max(axis=1)
        d_formant = np.maximum(0.0, np.diff(framed_formant, prepend=framed_formant[0]))
        # 95th percentile normalization
        p95_on = np.percentile(onset_env, 95) or 1.0
        p95_form = np.percentile(d_formant, 95) or 1.0
        comb_env = 0.60 * (onset_env / p95_on) + 0.40 * (d_formant / p95_form)
    except Exception:
        comb_env = onset_env

    # Detect onset frames with backtrack to onset beginning
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=comb_env,
        sr=sr_actual,
        hop_length=256,
        backtrack=True,
        units="frames",
        delta=delta,
    )

    onset_times = librosa.frames_to_time(onset_frames, sr=sr_actual, hop_length=256)
    _onset_cache[audio_path] = onset_times
    return onset_times


def snap_transcribed_data_to_onsets(
    audio_path: str,
    transcribed_data: list[TranscribedData],
    snap_window_ms: float = 45.0,
    phrase_snap_window_ms: Optional[float] = None,
    min_confidence: float = 0.0,
    rhythm_grid_mode: str = "straight",
) -> list[TranscribedData]:
    """Snap transcribed syllable start times to the nearest detected spectral onset.

    Args:
        audio_path: Path to vocal audio file
        transcribed_data: List of TranscribedData syllable objects
        snap_window_ms: Maximum distance (ms) to snap inside phrases (strict 40ms to 60ms)
        phrase_snap_window_ms: Maximum distance (ms) to snap phrase starters / post-silence words (up to 120-150ms).
            If None, defaults to snap_window_ms.
        min_confidence: Minimum syllable confidence to snap
        rhythm_grid_mode: Genre rhythmic grid mode ("straight", "dembow_332", "swing_ternary")

    Returns:
        TranscribedData list with aligned start times.
    """
    if not transcribed_data:
        return transcribed_data

    # Requirement 3: Strictly clamp onset snapping window to safety range (<= 50ms) to avoid drum bleed
    clamped_snap_window = min(50.0, max(10.0, float(snap_window_ms)))
    if phrase_snap_window_ms is not None:
        effective_phrase_window_ms = min(50.0, max(10.0, float(phrase_snap_window_ms)))
    else:
        effective_phrase_window_ms = clamped_snap_window

    grid_mode_desc = f" | grid: {cyan_highlighted(rhythm_grid_mode)}" if rhythm_grid_mode != "straight" else ""

    print(
        f"{ULTRASINGER_HEAD} Refining syllable timing with {blue_highlighted('Onset Snapping')} "
        f"(phrase start: {effective_phrase_window_ms:.0f}ms, intra-phrase: {clamped_snap_window:.0f}ms{grid_mode_desc})"
    )

    try:
        onset_times = detect_vocal_onsets(audio_path)
    except Exception as e:
        print(f"{ULTRASINGER_HEAD} {red_highlighted('Onset detection failed:')} {e}")
        return transcribed_data

    if len(onset_times) == 0:
        print(f"{ULTRASINGER_HEAD} No vocal onsets detected; skipping onset snapping.")
        return transcribed_data

    snap_count = 0
    total_snap_ms = 0.0

    for i, data in enumerate(transcribed_data):
        if data.confidence < min_confidence:
            continue

        start = data.start
        duration = data.end - data.start

        # Detect if this syllable starts a phrase or follows a pause (tarareos, intro, verse starts)
        is_phrase_start = False
        prev_end_limit = 0.0
        if i == 0:
            is_phrase_start = True
        else:
            prev_data = transcribed_data[i - 1]
            prev_end_limit = prev_data.end + 0.02
            if (start - prev_data.end) >= 0.180 or getattr(data, "is_line_start", False):
                is_phrase_start = True

        window_sec = (effective_phrase_window_ms if is_phrase_start else clamped_snap_window) / 1000.0
        if rhythm_grid_mode == "dembow_332" and not is_phrase_start:
            window_sec = max(window_sec, 0.055)  # Adaptive capture for 3-3-2 syncopated off-beat hits

        left_idx = int(np.searchsorted(onset_times, start - window_sec))
        right_idx = int(np.searchsorted(onset_times, start + window_sec, side="right"))

        best_onset: Optional[float] = None
        best_dist = float("inf")

        for candidate_idx in range(left_idx, min(right_idx, len(onset_times))):
            cand_time = float(onset_times[candidate_idx])
            if cand_time < prev_end_limit:
                continue
            raw_dist = cand_time - start
            dist = abs(raw_dist)

            # Ecuación 3 (Swing Ratio 2/3:1/3): In swing mode, off-beat delayed onsets are favored over premature onsets
            if rhythm_grid_mode == "swing_ternary" and raw_dist > 0:
                dist *= 0.85

            if dist < best_dist:
                best_dist = dist
                best_onset = cand_time

        if best_onset is not None and best_dist <= window_sec:
            target_start = max(best_onset, prev_end_limit)

            # Phase 2: Refine acoustic boundary with Bayesian Information Criterion (BIC)
            try:
                from modules.Audio.bic_segmenter import refine_onset_with_bic
                refined_t = refine_onset_with_bic(audio_path, target_start, window_ms=40.0)
                if refined_t >= prev_end_limit:
                    target_start = refined_t
            except Exception:
                pass

            snap_ms = (target_start - start) * 1000.0
            data.start = round(target_start, 3)
            data.end = round(max(target_start + 0.03, target_start + duration), 3)

            snap_count += 1
            total_snap_ms += abs(snap_ms)

    # Prevent overlaps between consecutive syllables and preserve positive durations
    min_dur = 0.03  # 30ms minimum duration
    for i in range(1, len(transcribed_data)):
        prev = transcribed_data[i - 1]
        curr = transcribed_data[i]
        if curr.start < prev.end:
            if (curr.end - prev.start) >= 2 * min_dur:
                mid = (prev.end + curr.start) / 2.0
                mid = max(prev.start + min_dur, min(curr.end - min_dur, mid))
                prev.end = mid
                curr.start = mid
            else:
                curr.start = max(curr.start, prev.end)
                if curr.end <= curr.start:
                    curr.end = curr.start + min_dur

    for data in transcribed_data:
        if data.end <= data.start:
            data.end = data.start + min_dur

    if snap_count > 0:
        avg_snap = total_snap_ms / snap_count
        print(
            f"{ULTRASINGER_HEAD} {green_highlighted('Onset Snapping:')} aligned "
            f"{blue_highlighted(str(snap_count))}/{len(transcribed_data)} syllables "
            f"(avg shift: {blue_highlighted(f'{avg_snap:.1f}ms')})"
        )
    else:
        print(f"{ULTRASINGER_HEAD} Onset Snapping: no syllables within tolerance of an onset.")

    return transcribed_data
