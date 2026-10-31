"""Vocal activity detection and #GAP calibration module."""

import os
import numpy as np
import librosa

from modules.console_colors import (
    ULTRASINGER_HEAD,
    blue_highlighted,
    green_highlighted,
    red_highlighted,
    cyan_highlighted,
)
from modules.Speech_Recognition.TranscribedData import TranscribedData


def detect_first_vocal_onset(
    audio_path: str,
    search_window_sec: float = 60.0,
    energy_threshold_ratio: float = 0.08,
    min_sustained_ms: float = 60.0,
    expected_start: float | None = None,
) -> float | None:
    """Detect the exact timestamp in seconds where singing first begins in isolated vocal audio.

    Args:
        audio_path: Path to vocal audio file (vocals.wav or [Vocals].mp3)
        search_window_sec: Search duration from start of file in seconds
        energy_threshold_ratio: RMS threshold ratio relative to peak vocal energy
        min_sustained_ms: Minimum duration the energy must be sustained to count as vocal onset
        expected_start: Optional expected onset from Whisper/lyrics to constrain search around real vocals

    Returns:
        Timestamp in seconds of the first vocal onset, or None if no vocals detected.
    """
    if not os.path.isfile(audio_path):
        return None

    try:
        offset_sec = max(0.0, expected_start - 0.40) if (expected_start is not None and expected_start > 0.40) else 0.0
        dur_sec = 1.2 if expected_start is not None else search_window_sec
        y, sr = librosa.load(audio_path, sr=22050, mono=True, offset=offset_sec, duration=dur_sec)
    except Exception:
        return None

    if len(y) == 0:
        return None

    hop_length = 128  # ~5.8ms resolution
    frame_length = 512
    D = librosa.stft(y, n_fft=frame_length, hop_length=hop_length)
    mag = np.abs(D)
    fft_freqs = librosa.fft_frequencies(sr=sr, n_fft=frame_length)
    f_formant_mask = (fft_freqs >= 300) & (fft_freqs <= 3400)
    e_formant = np.sum(mag[f_formant_mask, :] ** 2, axis=0)
    e_total = np.sum(mag ** 2, axis=0) + 1e-6
    formant_ratio = e_formant / e_total

    rms = librosa.feature.rms(S=mag, frame_length=frame_length)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length) + offset_sec

    max_rms = float(np.max(rms)) if len(rms) > 0 else 0.0
    if max_rms < 1e-4:
        return None

    # Threshold for vocal presence: adaptive ratio with formant band confirmation
    threshold = max(max_rms * energy_threshold_ratio, 0.008)

    # Required consecutive frames for sustained vocal sound
    frames_per_ms = sr / (hop_length * 1000.0)
    min_frames = max(1, int(round(min_sustained_ms * frames_per_ms)))

    consecutive = 0
    candidate_idx = None
    detected_onsets = []

    for i, val in enumerate(rms):
        is_vocal_formant = (formant_ratio[i] >= 0.28) if i < len(formant_ratio) else True
        if val >= threshold and is_vocal_formant:
            if consecutive == 0:
                candidate_idx = i
            consecutive += 1
            if consecutive == min_frames:
                # Backtrack to start of acoustic rise
                start_frame = max(0, candidate_idx - 1)
                t_onset = float(rms_times[start_frame])
                detected_onsets.append(t_onset)
        else:
            consecutive = 0
            candidate_idx = None

    if not detected_onsets:
        return None

    if expected_start is None:
        return detected_onsets[0]

    # Select the onset closest to expected_start within valid boundary
    closest = min(detected_onsets, key=lambda t: abs(t - expected_start))
    return closest


def validate_and_calibrate_gap(
    audio_path: str,
    transcribed_data: list[TranscribedData],
    auto_calibrate: bool = True,
    max_calibrate_ms: float = 500.0,
) -> tuple[float | None, float, float]:
    """Validate and optionally calibrate the #GAP (first syllable start) against the real vocal onset."""
    if not transcribed_data:
        return None, 0.0, 0.0

    first_syllable = transcribed_data[0]
    whisper_start = float(first_syllable.start)
    original_duration = float(first_syllable.end) - whisper_start

    vocal_onset = detect_first_vocal_onset(audio_path, expected_start=whisper_start)
    if vocal_onset is None:
        print(f"{ULTRASINGER_HEAD} Vocal onset: {blue_highlighted('Could not detect distinct onset in vocal audio')}")
        return None, whisper_start, 0.0

    diff_ms = (whisper_start - vocal_onset) * 1000.0

    if abs(diff_ms) <= 50.0:
        status_text = green_highlighted("EXCELLENT (< 50ms)")
    elif abs(diff_ms) <= 150.0:
        status_text = cyan_highlighted("GOOD (< 150ms)")
    elif abs(diff_ms) <= max_calibrate_ms:
        status_text = blue_highlighted(f"CALIBRATABLE (<= {max_calibrate_ms:.0f}ms)")
    else:
        status_text = red_highlighted(f"DISCREPANCY (> {max_calibrate_ms:.0f}ms)")

    calibrated = False
    if auto_calibrate and abs(diff_ms) <= max_calibrate_ms and abs(diff_ms) > 1.0:
        # Calibrate first syllable to acoustic onset
        first_syllable.start = vocal_onset
        if diff_ms > 500.0:
            # Elongated entrance vowel: singer began singing well before Whisper start
            new_end = max(vocal_onset + original_duration, float(first_syllable.end))
        else:
            # Standard timing jitter: preserve original syllable duration
            new_end = vocal_onset + original_duration

        if len(transcribed_data) > 1:
            next_start = float(transcribed_data[1].start)
            if new_end > next_start:
                new_end = max(vocal_onset + 0.03, next_start - 0.01)
        first_syllable.end = new_end
        calibrated = True

    final_start = float(first_syllable.start)

    calib_str = green_highlighted("YES") if calibrated else blue_highlighted("NO")
    print(f"{ULTRASINGER_HEAD} {cyan_highlighted('Vocal Start Validation (Acoustic VAD vs #GAP):')}")
    print(f"{ULTRASINGER_HEAD}   ├── Real vocal onset : {blue_highlighted(f'{vocal_onset:.3f}s')} ({int(vocal_onset * 1000)} ms)")
    print(f"{ULTRASINGER_HEAD}   ├── Transcribed #GAP : {blue_highlighted(f'{whisper_start:.3f}s')} ({int(whisper_start * 1000)} ms)")
    print(f"{ULTRASINGER_HEAD}   ├── Timing offset    : {blue_highlighted(f'{diff_ms:+.1f} ms')} [{status_text}]")
    print(f"{ULTRASINGER_HEAD}   └── Auto-calibrated  : {calib_str} -> Final #GAP: {green_highlighted(f'{int(final_start * 1000)} ms')}")

    return vocal_onset, final_start, diff_ms
