"""Key detection and pitch quantization to musical scale"""

try:
    import librosa
except Exception:
    librosa = None
import numpy as np

from modules.console_colors import (
    ULTRASINGER_HEAD,
blue_highlighted)

# scales (in semitones relative to root note)
MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Krumhansl-Schmuckler Psychoacoustic Key Profiles
KRUMHANSL_MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    dtype=np.float64,
)
KRUMHANSL_MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    dtype=np.float64,
)


def compute_pitch_class_profile(pitch_bins_matrix: np.ndarray) -> np.ndarray:
    """Módulo 3 MIR: Perfil de Clase de Tono (Cromagrama de 12 clases p in [0..11]):
      C(p) = sum_o |X(p + 12*o)|
    Acepta una matriz de bins semitonales (N_bins, T) o un cromagrama (12, T) y colapsa
    las octavas sumando la magnitud absoluta de cada clase de tono.
    """
    arr = np.abs(np.asarray(pitch_bins_matrix, dtype=np.float64))
    if arr.ndim == 1:
        arr = arr[:, None]
    n_bins, n_frames = arr.shape
    chroma = np.zeros((12, n_frames), dtype=np.float64)
    for b in range(n_bins):
        chroma[b % 12, :] += arr[b, :]
    return chroma


def detect_key_from_chroma(chroma: np.ndarray) -> tuple[str, str]:
    """Detecta la tonalidad global (ej. ('C', 'major') o ('A', 'minor')) a partir de un cromagrama (12, T)
    empleando correlación con los perfiles psicoacústicos de Krumhansl-Schmuckler.
    """
    c_mat = compute_pitch_class_profile(chroma)
    chroma_avg = np.mean(c_mat, axis=1)
    total = float(np.sum(chroma_avg))
    if total <= 1e-9:
        return "C", "major"
    chroma_avg = chroma_avg / total

    major_template = KRUMHANSL_MAJOR_PROFILE / np.sum(KRUMHANSL_MAJOR_PROFILE)
    minor_template = KRUMHANSL_MINOR_PROFILE / np.sum(KRUMHANSL_MINOR_PROFILE)

    best_correlation = -2.0
    best_key = "C"
    best_mode = "major"

    for shift in range(12):
        shifted_major = np.roll(major_template, shift)
        shifted_minor = np.roll(minor_template, shift)

        major_corr = float(np.corrcoef(chroma_avg, shifted_major)[0, 1])
        minor_corr = float(np.corrcoef(chroma_avg, shifted_minor)[0, 1])

        if major_corr > best_correlation:
            best_correlation = major_corr
            best_key = NOTE_NAMES[shift]
            best_mode = "major"

        if minor_corr > best_correlation:
            best_correlation = minor_corr
            best_key = NOTE_NAMES[shift]
            best_mode = "minor"

    return best_key, best_mode


def detect_dominant_chords_timeline(
    chroma: np.ndarray,
    times: np.ndarray,
    window_sec: float = 2.0,
) -> list[dict]:
    """Módulo 3 MIR: Identificación de Acordes Dominantes por ventana temporal (ej. 'C', 'Am', 'C#m', 'G').
    Evalúa las tríadas mayores [0, 4, 7] y menores [0, 3, 7] sobre el Perfil de Clase de Tono C(p).
    """
    c_mat = compute_pitch_class_profile(chroma)
    t_arr = np.asarray(times, dtype=np.float64)
    if c_mat.shape[1] == 0 or len(t_arr) == 0:
        return []

    maj_triad = np.zeros(12, dtype=np.float64)
    maj_triad[[0, 4, 7]] = [1.0, 0.85, 0.90]
    min_triad = np.zeros(12, dtype=np.float64)
    min_triad[[0, 3, 7]] = [1.0, 0.85, 0.90]

    t_min = float(t_arr[0])
    t_max = float(t_arr[-1])
    events = []
    curr_t = t_min

    while curr_t < t_max:
        next_t = min(t_max, curr_t + max(0.25, window_sec))
        mask = (t_arr >= curr_t) & (t_arr <= next_t)
        if np.any(mask):
            c_vec = np.mean(c_mat[:, mask], axis=1)
            norm = float(np.linalg.norm(c_vec))
            if norm > 1e-6:
                c_unit = c_vec / norm
                best_score = -1.0
                best_chord = "C"
                for root in range(12):
                    s_maj = float(np.dot(c_unit, np.roll(maj_triad, root)) / np.linalg.norm(maj_triad))
                    s_min = float(np.dot(c_unit, np.roll(min_triad, root)) / np.linalg.norm(min_triad))
                    if s_maj >= s_min and s_maj > best_score:
                        best_score = s_maj
                        best_chord = NOTE_NAMES[root]
                    elif s_min > s_maj and s_min > best_score:
                        best_score = s_min
                        best_chord = f"{NOTE_NAMES[root]}m"
                events.append({
                    "timestamp_start": round(curr_t, 3),
                    "timestamp_end": round(next_t, 3),
                    "dimension": "Acorde Dominante",
                    "quantitative_value": best_chord,
                    "confidence": round(best_score, 3),
                })
        curr_t = next_t

    return events


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
    best_key, best_mode = detect_key_from_chroma(chroma)

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