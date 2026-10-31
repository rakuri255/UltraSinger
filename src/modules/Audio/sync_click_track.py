"""Generate a sync click track audio file for acoustic verification."""

import os
import tempfile
import numpy as np
import librosa
import soundfile as sf

from modules.Audio.convert_audio import convert_audio_format
from modules.console_colors import (
    ULTRASINGER_HEAD,
    blue_highlighted,
    green_highlighted,
)
from modules.Midi.MidiSegment import MidiSegment


def generate_click_sample(
    sample_rate: int = 44100,
    duration_ms: float = 16.0,
    freq_hz: float = 1800.0,
    amplitude: float = 0.85,
) -> np.ndarray:
    """Synthesize a sharp, percussive click sound with exponential decay.

    Args:
        sample_rate: Audio sample rate
        duration_ms: Click length in milliseconds
        freq_hz: Tone frequency in Hz
        amplitude: Peak amplitude (0.0 to 1.0)

    Returns:
        1D float32 numpy array representing the click waveform.
    """
    num_samples = int(sample_rate * (duration_ms / 1000.0))
    if num_samples <= 0:
        return np.zeros(0, dtype=np.float32)

    t = np.linspace(0, duration_ms / 1000.0, num_samples, endpoint=False, dtype=np.float32)
    # Fast decaying sine burst
    sine = np.sin(2.0 * np.pi * freq_hz * t)
    decay_rate = 1000.0 / (duration_ms / 3.0)
    envelope = np.exp(-t * decay_rate)

    click = (sine * envelope * amplitude).astype(np.float32)
    return click


def create_sync_click_track(
    vocal_audio_path: str,
    output_audio_path: str,
    midi_segments: list[MidiSegment],
    accent_first_syllable: bool = True,
) -> str | None:
    """Mix an audible metronome click onto the vocal track at every note start.

    Args:
        vocal_audio_path: Path to vocal audio file
        output_audio_path: Destination path for sync test audio (.mp3 or .wav)
        midi_segments: List of MidiSegment objects containing note start times
        accent_first_syllable: If True, uses higher pitch (2400Hz) after pauses > 0.35s

    Returns:
        output_audio_path on success, or None on failure.
    """
    if not os.path.isfile(vocal_audio_path) or not midi_segments:
        return None

    try:
        y, sr = librosa.load(vocal_audio_path, sr=44100, mono=True)
    except Exception as e:
        print(f"{ULTRASINGER_HEAD} Warning: Could not load audio for click track: {e}")
        return None

    if len(y) == 0:
        return None

    accent_click = generate_click_sample(sr, duration_ms=18.0, freq_hz=2400.0, amplitude=0.90)
    regular_click = generate_click_sample(sr, duration_ms=14.0, freq_hz=1600.0, amplitude=0.75)

    audio_len = len(y)
    prev_end = 0.0
    clicks_added = 0

    for segment in midi_segments:
        start_time = float(segment.start)
        start_idx = int(round(start_time * sr))
        if start_idx < 0 or start_idx >= audio_len:
            continue

        # Check if this syllable starts a new phrase (pause > 350ms)
        is_phrase_start = (start_time - prev_end) > 0.35
        prev_end = float(segment.end)

        click = accent_click if (accent_first_syllable and is_phrase_start) else regular_click
        click_len = len(click)

        end_idx = min(audio_len, start_idx + click_len)
        actual_len = end_idx - start_idx
        if actual_len > 0:
            y[start_idx:end_idx] += click[:actual_len]
            clicks_added += 1

    # Prevent digital clipping
    y = np.clip(y, -1.0, 1.0)

    # Save output
    output_ext = os.path.splitext(output_audio_path)[1].lower()
    temp_wav = None
    try:
        if output_ext == ".wav":
            sf.write(output_audio_path, y, sr)
        else:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_wav = tmp.name
            sf.write(temp_wav, y, sr)
            convert_audio_format(temp_wav, output_audio_path)
    finally:
        if temp_wav and os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
            except OSError:
                pass

    print(
        f"{ULTRASINGER_HEAD} Generated Sync Test Audio with {blue_highlighted(str(clicks_added))} clicks -> "
        f"{green_highlighted(output_audio_path)}"
    )
    return output_audio_path
