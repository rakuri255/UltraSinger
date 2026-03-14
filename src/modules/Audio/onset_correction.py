"""Onset correction module — snaps note start times to audio onsets.

After WhisperX forced alignment, note start times may be offset from
the actual vocal onset by up to ~100ms.  This module detects audio
onsets using librosa and snaps nearby note starts to improve timing
accuracy.
"""

from __future__ import annotations

import numpy as np
import librosa
import soundfile as sf

from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted
from modules.Speech_Recognition.TranscribedData import TranscribedData


def detect_vocal_onsets(
    audio_path: str,
    hop_length: int = 512,
) -> np.ndarray:
    """Detect onset times in the vocal audio.

    Args:
        audio_path: Path to the audio file (vocals track).
        hop_length: Number of samples between onset strength frames.

    Returns:
        1-D array of onset times in seconds, sorted ascending.
    """
    y, sr = sf.read(audio_path, dtype="float32")
    if y.ndim > 1:
        y = librosa.to_mono(y.T)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        backtrack=True,  # snap to nearest preceding minimum
    )
    return librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)


def snap_to_onsets(
    transcribed_data: list[TranscribedData],
    onset_times: np.ndarray,
    max_snap_ms: float = 80.0,
) -> list[TranscribedData]:
    """Snap note start times to nearby detected audio onsets.

    For each note, finds the nearest onset within *max_snap_ms*.
    If found, adjusts the note's ``start`` to the onset time.
    End times are **not** modified (they are either derived from the
    next note's start or kept as-is).

    The function preserves chronological order: a snapped start will
    never move past the note's own end time (with a 10 ms safety
    margin to avoid zero-duration notes).

    Args:
        transcribed_data: List of transcribed segments.
        onset_times: Sorted array of onset times in seconds.
        max_snap_ms: Maximum snap distance in milliseconds.

    Returns:
        The same list, with ``start`` values adjusted in-place.
    """
    if len(onset_times) == 0 or not transcribed_data:
        return transcribed_data

    max_snap_s = max_snap_ms / 1000.0
    snap_count = 0
    min_duration_s = 0.01  # 10 ms minimum note duration

    prev_end = float("-inf")
    for data in transcribed_data:
        # Binary search for nearest onset
        idx = np.searchsorted(onset_times, data.start)

        candidates: list[float] = []
        if idx > 0:
            candidates.append(float(onset_times[idx - 1]))
        if idx < len(onset_times):
            candidates.append(float(onset_times[idx]))

        if not candidates:
            prev_end = data.end
            continue

        nearest = min(candidates, key=lambda t: abs(t - data.start))
        distance = abs(nearest - data.start)

        if distance <= max_snap_s:
            # Don't snap before the previous note's end (overlap prevention)
            candidate = max(nearest, prev_end)
            # Don't create zero/negative duration notes
            if candidate < data.end - min_duration_s:
                data.start = candidate
                snap_count += 1

        prev_end = data.end

    print(
        f"{ULTRASINGER_HEAD} Onset correction: snapped "
        f"{blue_highlighted(str(snap_count))} of "
        f"{blue_highlighted(str(len(transcribed_data)))} notes "
        f"(max {max_snap_ms:.0f}ms)"
    )

    return transcribed_data
