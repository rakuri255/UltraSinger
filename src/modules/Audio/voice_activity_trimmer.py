import os
import re
import numpy as np
import librosa

from modules.console_colors import (
    ULTRASINGER_HEAD,
    blue_highlighted,
    green_highlighted,
)
from modules.Speech_Recognition.TranscribedData import TranscribedData

# Regex para consonantes sordas, oclusivas y fricativas (s, d, t, k, z, p, c, f, j, x, ch)
UNVOICED_CONSONANT_PATTERN = re.compile(r"[sdtkzcpfjx]\s*$", re.IGNORECASE)
UNVOICED_INITIAL_CONSONANT_PATTERN = re.compile(r"^[sdtkzcpfjx]", re.IGNORECASE)


def trim_syllable_ends_with_energy(
    audio_path: str,
    transcribed_data: list[TranscribedData],
    energy_threshold_ratio: float = 0.05,
    min_syllable_dur: float = 0.06,
    max_trim_sec: float = 0.5,
) -> list[TranscribedData]:
    """Trim excess leading and tail silence from syllables using Hilbert envelope VAD.

    Prevents notes from starting early in dead silence or hanging past the end of singing.
    """
    if not transcribed_data or not os.path.isfile(audio_path):
        return transcribed_data

    try:
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
    except Exception as e:
        return transcribed_data

    if len(y) == 0:
        return transcribed_data

    # Pilar 3: Analytic Signal Instantaneous Envelope via Hilbert Transform z(t) = s(t) + j*H{s(t)}
    try:
        from scipy.signal import hilbert
        analytic_signal = hilbert(y)
        amplitude_envelope = np.abs(analytic_signal)
        w_env = max(5, int(round(sr * 0.015)))  # ~15ms smoothing
        smoothed_env = np.convolve(amplitude_envelope, np.ones(w_env) / w_env, mode="same")
        hop_length = 64  # Fine ~2.9ms temporal resolution
        env = smoothed_env[::hop_length]
        env_times = librosa.frames_to_time(np.arange(len(env)), sr=sr, hop_length=hop_length)
    except Exception:
        hop_length = 128
        env = librosa.feature.rms(y=y, frame_length=512, hop_length=hop_length)[0]
        env_times = librosa.frames_to_time(np.arange(len(env)), sr=sr, hop_length=hop_length)

    trim_tail_count = 0
    trim_lead_count = 0
    total_trimmed_ms = 0.0

    for i, data in enumerate(transcribed_data):
        duration = data.end - data.start
        if duration <= min_syllable_dur:
            continue

        start_idx = int(np.searchsorted(env_times, data.start))
        end_idx = int(np.searchsorted(env_times, data.end))

        if start_idx >= end_idx or end_idx >= len(env):
            continue

        syl_env = env[start_idx:end_idx]
        if len(syl_env) == 0:
            continue

        peak_energy = float(np.max(syl_env))
        if peak_energy <= 1e-5:
            continue

        threshold_energy = peak_energy * energy_threshold_ratio

        # Ecuación II (Leading Silence Gate): Scan from start_idx forward to find where vocal sound actually attacks
        # Enable leading trim on line starts, after pauses, or whenever the initial frames are silent (< threshold_energy)
        prev_end_t = transcribed_data[i - 1].end if i > 0 else 0.0
        is_prev_line_end = getattr(transcribed_data[i - 1], "is_line_end", False) if i > 0 else True
        starts_in_silence = bool(len(syl_env) > 3 and float(np.mean(syl_env[:3])) < threshold_energy)
        can_trim_lead = (
            (i == 0)
            or is_prev_line_end
            or getattr(data, "is_line_start", False)
            or ((data.start - prev_end_t) >= 0.04)
            or starts_in_silence
        )

        new_start_time = data.start
        if can_trim_lead:
            vocal_start_frame = start_idx
            for f_idx in range(start_idx, end_idx):
                if env[f_idx] >= threshold_energy:
                    vocal_start_frame = f_idx
                    break

            if vocal_start_frame > start_idx + 2:
                onset_t = float(env_times[vocal_start_frame])
                word_clean = str(data.word).strip()
                lead_pad = 0.025 if UNVOICED_INITIAL_CONSONANT_PATTERN.search(word_clean) else 0.010
                cand_start = max(data.start, onset_t - lead_pad)
                if cand_start < data.end - min_syllable_dur and (cand_start - data.start) > 0.020:
                    lead_trimmed = cand_start - data.start
                    total_trimmed_ms += lead_trimmed * 1000.0
                    data.start = cand_start
                    new_start_time = cand_start
                    trim_lead_count += 1
                    start_idx = int(np.searchsorted(env_times, data.start))

        # Causal Tail Gate: NEVER count trailing silence frames until AFTER vocal energy has first risen above threshold_energy.
        # This prevents leading breathing pauses from tricking the tail trimmer into crushing words at start_idx.
        silence_hangover_frames = int(round(0.100 / (hop_length / sr)))
        consecutive_silent = 0
        vocal_end_frame = end_idx - 1
        has_vocal_started = False

        for f_idx in range(start_idx, end_idx):
            if env[f_idx] >= threshold_energy:
                has_vocal_started = True
                consecutive_silent = 0
                vocal_end_frame = f_idx
            elif has_vocal_started:
                consecutive_silent += 1
                if consecutive_silent >= silence_hangover_frames:
                    vocal_end_frame = f_idx - consecutive_silent
                    break

        new_end_time = float(env_times[max(start_idx, vocal_end_frame)])

        # INYECCIÓN DE PADDING: Proteger consonantes sordas/fricativas finales (ej. 's', 'd', 't', 'z')
        word_str = str(data.word)
        if UNVOICED_CONSONANT_PATTERN.search(word_str):
            new_end_time = min(data.end, new_end_time + 0.050)

        # Respect minimum duration and trim excess silence tail
        trimmed_sec = data.end - new_end_time
        if trimmed_sec > 0.02 and new_end_time >= new_start_time + min_syllable_dur:
            total_trimmed_ms += trimmed_sec * 1000.0
            data.end = new_end_time
            trim_tail_count += 1

    if (trim_tail_count + trim_lead_count) > 0:
        print(
            f"{ULTRASINGER_HEAD} {green_highlighted('Energy Trimmer (Leading & Tail Gate):')} trimmed "
            f"{blue_highlighted(str(trim_lead_count))} early attacks and {blue_highlighted(str(trim_tail_count))} silence tails "
            f"({blue_highlighted(f'{total_trimmed_ms:.1f}ms')} total silence removed)"
        )

    return transcribed_data
