"""Key detection and pitch quantization to musical scale"""

import librosa
import numpy as np

from modules.console_colors import (
    ULTRASINGER_HEAD,
blue_highlighted)

# scales (in semitones relative to root note)
MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def detect_key_from_audio(audio_path: str) -> tuple[str, str]:
    """
    Detect the key and mode (major/minor) of a song from audio file.

    Args:
        audio_path: Path to audio file

    Returns:
        Tuple of (key_note, mode) e.g., ('C', 'major') or ('A', 'minor')
    """
    print(f"{ULTRASINGER_HEAD} Detecting musical key")

    y, sr = librosa.load(audio_path, sr=None, duration=60.0)  # Analyze first 60 seconds
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_avg = np.mean(chroma, axis=1)
    chroma_avg = chroma_avg / np.sum(chroma_avg)

    major_template = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])  # Major scale pattern
    minor_template = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])  # Minor scale pattern
    major_template = major_template / np.sum(major_template)
    minor_template = minor_template / np.sum(minor_template)

    best_correlation = -1
    best_key = None
    best_mode = None

    # Try all 12 keys
    for shift in range(12):
        shifted_major = np.roll(major_template, shift)
        shifted_minor = np.roll(minor_template, shift)

        major_corr = np.corrcoef(chroma_avg, shifted_major)[0, 1]
        minor_corr = np.corrcoef(chroma_avg, shifted_minor)[0, 1]

        if major_corr > best_correlation:
            best_correlation = major_corr
            best_key = NOTE_NAMES[shift]
            best_mode = 'major'

        if minor_corr > best_correlation:
            best_correlation = minor_corr
            best_key = NOTE_NAMES[shift]
            best_mode = 'minor'

    print(f"{ULTRASINGER_HEAD} Detected key: {blue_highlighted(best_key)} {blue_highlighted(best_mode)}")
    return best_key, best_mode


def get_allowed_notes_for_key(key_note: str, mode: str) -> set[str]:
    """
    Get all allowed note names for a given key across all octaves.

    Args:
        key_note: Root note (e.g., 'C', 'D#', 'F')
        mode: 'major' or 'minor'

    Returns:
        Set of allowed note names (e.g., {'C', 'D', 'E', ...})
    """
    root_idx = NOTE_NAMES.index(key_note)
    scale = MAJOR_SCALE if mode == 'major' else MINOR_SCALE

    allowed_notes = set()
    for interval in scale:
        note_idx = (root_idx + interval) % 12
        allowed_notes.add(NOTE_NAMES[note_idx])

    return allowed_notes


def quantize_note_to_key(note: str, allowed_notes: set[str]) -> str:
    """
    Quantize a note to the nearest allowed note in the key.

    Args:
        note: MIDI note name (e.g., 'C4', 'D#5')
        allowed_notes: Set of allowed note names without octave

    Returns:
        Quantized note name
    """
    print(f"{ULTRASINGER_HEAD} Quantizing note {blue_highlighted(note)} to key with allowed notes")
    # Parse note and octave
    if len(note) == 0:
        return note

    # Extract note name and octave
    if len(note) >= 2 and note[-2] == '#':
        note_name = note[:-1]
        octave = note[-1]
    else:
        note_name = note[:-1] if note[-1].isdigit() else note
        octave = note[-1] if note[-1].isdigit() else ''

    # If already in key, return as is
    if note_name in allowed_notes:
        return note

    # Find nearest allowed note
    try:
        note_midi = librosa.note_to_midi(note)
    except:
        return note

    # Try all allowed notes in nearby octaves
    min_distance = float('inf')
    best_note = note

    for allowed_note_name in allowed_notes:
        # Try same octave and adjacent octaves
        for oct_offset in [-1, 0, 1]:
            if not octave:
                test_octave = 4 + oct_offset
            else:
                test_octave = int(octave) + oct_offset

            if test_octave < 0 or test_octave > 8:
                continue

            test_note = f"{allowed_note_name}{test_octave}"

            try:
                test_midi = librosa.note_to_midi(test_note)
                distance = abs(test_midi - note_midi)

                if distance < min_distance:
                    min_distance = distance
                    best_note = test_note
            except:
                continue

    #print(f"{ULTRASINGER_HEAD} Moved {blue_highlighted(note)} to {blue_highlighted(best_note)}")
    return best_note