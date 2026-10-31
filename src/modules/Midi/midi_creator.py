"""Midi creator module"""

import math
import os
from collections import Counter

try:
    import librosa
except Exception:
    librosa = None
import numpy as np
try:
    import pretty_midi
except Exception:
    pretty_midi = None
try:
    import unidecode
except Exception:
    unidecode = None

from modules.Midi.MidiSegment import MidiSegment
from modules.Speech_Recognition.TranscribedData import TranscribedData
from modules.console_colors import (
    ULTRASINGER_HEAD, blue_highlighted,
)
from modules.Ultrastar.ultrastar_txt import UltrastarTxtValue
from modules.Pitcher.pitched_data import PitchedData
from modules.Pitcher.pitched_data_helper import get_frequencies_with_high_confidence
from modules.Audio.key_detector import quantize_note_to_key
from modules.Audio.ultrastar_autocorrect import estimate_pitch_kde_mode, evaluate_vibrato_sinusoid

def create_midi_instrument(
    midi_segments: list[MidiSegment],
    audio_path: str = None,
    pitched_data: PitchedData = None,
) -> object:
    """Converts an Ultrastar data to a midi instrument with dynamic RMS velocity and pitch bends."""

    print(f"{ULTRASINGER_HEAD} Creating midi instrument (dynamic velocity & pitch bends)")

    instrument = pretty_midi.Instrument(program=0, name="Vocals")

    # 1. Compute dynamic RMS velocity per segment if audio_path is provided
    velocities = [100] * len(midi_segments)
    if audio_path and os.path.exists(audio_path):
        try:
            import soundfile as sf
            y, sr = sf.read(audio_path)
            if y.ndim > 1:
                y = np.mean(y, axis=1)

            segment_rms = []
            for seg in midi_segments:
                start_sample = max(0, int(round(seg.start * sr)))
                end_sample = min(len(y), int(round(seg.end * sr)))
                if end_sample > start_sample:
                    chunk = y[start_sample:end_sample]
                    rms = float(np.sqrt(np.mean(chunk**2)))
                else:
                    rms = 0.0
                segment_rms.append(rms)

            rms_max = max(segment_rms) if segment_rms else 0.0
            if rms_max > 1e-6:
                for idx, rms_val in enumerate(segment_rms):
                    norm_rms = rms_val / rms_max
                    # Dynamic acoustic mapping: Velocity = clip(35 + 85 * sqrt(norm_rms), 30, 127)
                    vel = int(np.clip(round(35 + 85 * np.sqrt(norm_rms)), 30, 127))
                    velocities[idx] = vel
        except Exception as e:
            print(f"{ULTRASINGER_HEAD} [Warning] Could not extract dynamic RMS velocity: {e}")
            velocities = [100] * len(midi_segments)

    for i, midi_segment in enumerate(midi_segments):
        note_pitch = librosa.note_to_midi(midi_segment.note)
        vel = velocities[i] if i < len(velocities) else 100
        note = pretty_midi.Note(vel, note_pitch, midi_segment.start, midi_segment.end)
        instrument.notes.append(note)

    # 2. Inject continuous Pitch Bends (14-bit wheel [-8192, 8191]) if pitched_data is provided
    if pitched_data and pitched_data.times and pitched_data.frequencies:
        times_arr = np.array(pitched_data.times)
        freqs_arr = np.array(pitched_data.frequencies)
        confs_arr = np.array(pitched_data.confidence) if pitched_data.confidence else np.ones_like(freqs_arr)

        for midi_segment in midi_segments:
            center_midi = librosa.note_to_midi(midi_segment.note)
            t_start = midi_segment.start
            t_end = midi_segment.end

            idx_start = np.searchsorted(times_arr, t_start, side="left")
            idx_end = np.searchsorted(times_arr, t_end, side="right")

            for k in range(idx_start, idx_end):
                if k >= len(times_arr):
                    break
                f0 = freqs_arr[k]
                conf = confs_arr[k]
                t_frame = times_arr[k]

                if f0 > 60.0 and conf >= 0.3:
                    # Pitch in continuous fractional semitones: m(t) = 69 + 12 * log2(f0 / 440)
                    continuous_midi = 69.0 + 12.0 * np.log2(f0 / 440.0)
                    delta_semitones = continuous_midi - center_midi
                    # Standard MIDI Pitch Wheel range is +/- 2 semitones mapped to [-8192, 8191]
                    if -2.0 <= delta_semitones <= 2.0:
                        bend_val = int(np.clip(round((delta_semitones / 2.0) * 8191), -8192, 8191))
                        instrument.pitch_bends.append(pretty_midi.PitchBend(pitch=bend_val, time=float(t_frame)))

            # Reset pitch bend to 0 at the end of the note
            instrument.pitch_bends.append(pretty_midi.PitchBend(pitch=0, time=float(t_end)))

        # Sort pitch bends chronologically
        instrument.pitch_bends.sort(key=lambda pb: pb.time)

    return instrument

def sanitize_for_midi(text):
    """
    Sanitize text for MIDI compatibility.
    Uses unidecode to approximate characters to ASCII.
    """
    return unidecode.unidecode(text)

def __create_midi(instruments: list[object], bpm: float, midi_output: str, midi_segments: list[MidiSegment]) -> None:
    """Write instruments to midi file"""

    print(f"{ULTRASINGER_HEAD} Creating midi file -> {midi_output}")

    midi_data = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    for i, midi_segment in enumerate(midi_segments):
        sanitized_word = sanitize_for_midi(midi_segment.word)
        midi_data.lyrics.append(pretty_midi.Lyric(text=sanitized_word, time=midi_segment.start))
    for instrument in instruments:
        midi_data.instruments.append(instrument)
    midi_data.write(midi_output)


class MidiCreator:
    """Docstring"""


_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _hz_to_midi_safe(frequency: float) -> float:
    if librosa is not None:
        return float(librosa.hz_to_midi(float(frequency)))
    return 69.0 + 12.0 * math.log2(max(float(frequency), 1e-6) / 440.0)


def safe_midi_to_note(midi_number: int) -> str:
    """Converts midi number to note string guaranteed to use ASCII characters (# and b)."""
    m_int = int(round(midi_number))
    if librosa is not None:
        return librosa.midi_to_note(m_int, unicode=False).replace("♯", "#").replace("♭", "b")
    octave = (m_int // 12) - 1
    note = _NOTE_NAMES[m_int % 12]
    return f"{note}{octave}"


def safe_hz_to_note(frequency: float) -> str:
    """Converts hz to note string guaranteed to use ASCII characters (# and b)."""
    if librosa is not None:
        return librosa.hz_to_note(float(frequency), unicode=False).replace("♯", "#").replace("♭", "b")
    return safe_midi_to_note(int(round(_hz_to_midi_safe(frequency))))


def convert_frequencies_to_notes(frequency: [str]) -> list[list[str]]:
    """Converts frequencies to notes"""
    notes = []
    for freq in frequency:
        notes.append(safe_hz_to_note(float(freq)))
    return notes


def most_frequent(array: [str]) -> list[tuple[str, int]]:
    """Get most frequent item in array"""
    return Counter(array).most_common(1)


def find_nearest_index(array: list[float], value: float) -> int:
    """Nearest index in array"""
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (
        idx == len(array)
        or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])
    ):
        return idx - 1

    return idx


def segment_melismas_from_f0(
    start_time: float,
    end_time: float,
    word: str,
    pitched_data: PitchedData,
    allowed_notes: set[str] = None,
    min_segment_duration: float = 0.200,
    pitch_semitone_threshold: float = 1.0,
    max_melismas_per_syllable: int = 2,
    vibrato_freq_range: tuple[float, float] = (4.5, 7.5),
    vibrato_max_amplitude_st: float = 1.2,
) -> list[MidiSegment]:
    """Evalúa la curva F0 dentro de la sílaba. Si hay cambio >= pitch_semitone_threshold sostenido,
    divide en [MidiSegment(word), MidiSegment('~'), ...].
    Filtra el vibrato natural y micro-fluctuaciones para evitar cascadas de ~.
    """
    start_idx = find_nearest_index(pitched_data.times, start_time)
    end_idx = find_nearest_index(pitched_data.times, end_time)

    if start_idx >= end_idx:
        end_idx = start_idx + 1

    times = np.array(pitched_data.times[start_idx:end_idx], dtype=np.float64)
    freqs = np.array(pitched_data.frequencies[start_idx:end_idx], dtype=np.float64)
    confs = np.array(pitched_data.confidence[start_idx:end_idx], dtype=np.float64)

    # Filtrar frecuencias al rango vocal humano (65 Hz a 1100 Hz) con confianza suficiente
    valid_mask = (freqs >= 65.0) & (freqs <= 1100.0) & (confs >= 0.4)
    if not np.any(valid_mask):
        fallback_freq = freqs[0] if len(freqs) > 0 and freqs[0] > 0 else 440.0
        note = safe_hz_to_note(fallback_freq)
        if allowed_notes:
            note = quantize_note_to_key(note, allowed_notes)
        return [MidiSegment(note, start_time, end_time, word)]

    # Convertir frecuencias válidas a semitonos continuos (MIDI pitch)
    raw_midi_pitches = np.full(len(freqs), np.nan, dtype=np.float64)
    for i in range(len(freqs)):
        if valid_mask[i]:
            raw_midi_pitches[i] = _hz_to_midi_safe(freqs[i])

    valid_indices = np.where(~np.isnan(raw_midi_pitches))[0]

    # Guard 1: Comprobar si toda la trayectoria vocal es un vibrato sinusoidal biológico
    if len(valid_indices) >= 20:
        is_vib, _, _ = evaluate_vibrato_sinusoid(
            raw_midi_pitches[valid_indices],
            times[valid_indices],
            f_min=vibrato_freq_range[0],
            f_max=vibrato_freq_range[1],
            max_amplitude_st=vibrato_max_amplitude_st,
        )
        if is_vib:
            weights = confs[valid_indices] ** 2
            mode_midi = estimate_pitch_kde_mode(raw_midi_pitches[valid_indices], weights, bandwidth_semitones=0.35)
            note = safe_midi_to_note(int(round(mode_midi)))
            if allowed_notes:
                note = quantize_note_to_key(note, allowed_notes)
            return [MidiSegment(note, start_time, end_time, word)]

    midi_pitches = raw_midi_pitches.copy()

    # Suavizado anti-vibrato: aplicar filtro de mediana temporal adaptive
    if len(valid_indices) >= 5:
        try:
            from scipy.signal import medfilt
            k = min(5 if len(valid_indices) <= 25 else 11, len(valid_indices))
            if k % 2 == 0:
                k -= 1
            if k >= 3:
                smoothed_values = medfilt(midi_pitches[valid_indices], kernel_size=k)
                for idx, sm_val in zip(valid_indices, smoothed_values):
                    midi_pitches[idx] = sm_val
        except Exception:
            pass

    # Detectar cambios sostenidos >= pitch_semitone_threshold (requiere persistencia en >= 3 frames)
    blocks = []
    current_block_start = start_time
    current_pitches = []

    for i in range(len(midi_pitches)):
        p = midi_pitches[i]
        t = times[i]
        if np.isnan(p):
            continue

        if not current_pitches:
            current_pitches.append(p)
            continue

        pitch_arr = np.array(current_pitches, dtype=np.float64)
        kde_current = estimate_pitch_kde_mode(pitch_arr, np.ones(len(pitch_arr)))
        # Comprobar persistencia: al menos 3 frames válidos consecutivos deben sostener el salto
        lookahead_valid = [
            midi_pitches[j]
            for j in range(i, min(len(midi_pitches), i + 6))
            if not np.isnan(midi_pitches[j])
        ][:3]
        is_sustained_shift = (
            len(lookahead_valid) >= 3
            and all(abs(lp - kde_current) >= pitch_semitone_threshold for lp in lookahead_valid)
            and all(np.sign(lp - kde_current) == np.sign(p - kde_current) for lp in lookahead_valid)
        )

        if is_sustained_shift and (t - current_block_start) >= min_segment_duration:
            blocks.append((current_block_start, t, round(kde_current)))
            current_block_start = t
            current_pitches = [p]
        else:
            current_pitches.append(p)

    if current_pitches:
        pitch_arr = np.array(current_pitches, dtype=np.float64)
        kde_pitch = estimate_pitch_kde_mode(pitch_arr, np.ones(len(pitch_arr)))
        blocks.append((current_block_start, end_time, round(kde_pitch)))

    # Merge consecutive blocks with identical midi pitch
    merged_blocks = []
    for b_start, b_end, b_midi in blocks:
        if merged_blocks and merged_blocks[-1][2] == b_midi:
            merged_blocks[-1] = (merged_blocks[-1][0], b_end, b_midi)
        else:
            merged_blocks.append((b_start, b_end, b_midi))
    blocks = merged_blocks

    # Filter out blocks shorter than min_segment_duration, merging into adjacent
    if len(blocks) > 1:
        filtered = []
        for b_start, b_end, b_midi in blocks:
            if (b_end - b_start) >= min_segment_duration:
                filtered.append((b_start, b_end, b_midi))
            elif filtered:
                prev = filtered[-1]
                filtered[-1] = (prev[0], b_end, prev[2])
            else:
                filtered.append((b_start, b_end, b_midi))
        blocks = filtered

    # Cap maximum number of melismas per syllable to prevent visual clutter in UltraStar
    if len(blocks) > max_melismas_per_syllable:
        blocks.sort(key=lambda b: (b[1] - b[0]), reverse=True)
        top_blocks = sorted(blocks[:max_melismas_per_syllable], key=lambda b: b[0])
        top_blocks[0] = (start_time, top_blocks[0][1], top_blocks[0][2])
        for k in range(len(top_blocks) - 1):
            top_blocks[k] = (top_blocks[k][0], top_blocks[k + 1][0], top_blocks[k][2])
        top_blocks[-1] = (top_blocks[-1][0], end_time, top_blocks[-1][2])
        blocks = top_blocks

    if not blocks:
        note = safe_hz_to_note(440.0)
        return [MidiSegment(note, start_time, end_time, word)]

    # Construir MidiSegments: el primero lleva el texto, los subsecuentes llevan estrictamente '~'
    segments = []
    has_trailing_space = word.endswith(" ")
    clean_word = word.rstrip(" ")

    for idx, (b_start, b_end, b_midi) in enumerate(blocks):
        note_name = safe_midi_to_note(int(b_midi))
        if allowed_notes:
            note_name = quantize_note_to_key(note_name, allowed_notes)

        if idx == 0:
            seg_word = clean_word if len(blocks) > 1 else word
        else:
            seg_word = "~ " if (idx == len(blocks) - 1 and has_trailing_space) else "~"

        segments.append(MidiSegment(note_name, float(b_start), float(b_end), seg_word))

    if has_trailing_space and len(segments) > 1 and not segments[-1].word.endswith(" "):
        segments[-1].word += " "

    return segments


def correct_octave_jumps(midi_segments: list[MidiSegment]) -> list[MidiSegment]:
    """Corrects spurious octave jumps caused by harmonic locking (e.g. pitch 24 on consonants or cymbals)."""
    if not midi_segments:
        return midi_segments

    # Calculate global song median to protect against isolated single-note phrases
    all_midi_nums = []
    for seg in midi_segments:
        try:
            clean_n = str(seg.note).replace("♯", "#").replace("♭", "b")
            all_midi_nums.append(float(librosa.note_to_midi(clean_n)))
        except Exception:
            pass
    global_median = float(np.median(all_midi_nums)) if all_midi_nums else 60.0

    # Group segments into phrases by silence (>= 0.4s) or explicit is_line_end
    phrases: list[list[MidiSegment]] = []
    current_phrase: list[MidiSegment] = []

    for i, seg in enumerate(midi_segments):
        current_phrase.append(seg)
        is_break = getattr(seg, "is_line_end", False)
        if not is_break and i < len(midi_segments) - 1:
            if (midi_segments[i + 1].start - seg.end) >= 0.4:
                is_break = True
        if is_break or i == len(midi_segments) - 1:
            phrases.append(current_phrase)
            current_phrase = []

    for phrase in phrases:
        if not phrase:
            continue
        midi_nums = []
        valid_indices = []
        for idx, seg in enumerate(phrase):
            try:
                clean_n = str(seg.note).replace("♯", "#").replace("♭", "b")
                val = float(librosa.note_to_midi(clean_n))
                midi_nums.append(val)
                valid_indices.append(idx)
            except Exception:
                pass

        if not midi_nums:
            continue

        phrase_median = float(np.median(midi_nums)) if len(valid_indices) >= 3 else global_median

        for k, seg_idx in enumerate(valid_indices):
            curr = midi_nums[k]
            # 1. Clamp against global song vocal center first
            while (curr - global_median) >= 14.0:
                curr -= 12.0
            while (global_median - curr) >= 14.0:
                curr += 12.0

            # 2. Fold notes deviating by >= 12 semitones from phrase median
            while (curr - phrase_median) >= 12.0:
                curr -= 12.0
            while (phrase_median - curr) >= 12.0:
                curr += 12.0

            # 3. Check isolated single-note spikes relative to neighbors
            if len(valid_indices) >= 3:
                prev_p = midi_nums[k - 1] if k > 0 else midi_nums[1]
                next_p = midi_nums[k + 1] if k < len(valid_indices) - 1 else midi_nums[-2]
                neighbor_avg = (prev_p + next_p) / 2.0
                if abs(prev_p - next_p) <= 7.0:
                    while (curr - neighbor_avg) >= 11.0:
                        curr -= 12.0
                    while (neighbor_avg - curr) >= 11.0:
                        curr += 12.0

            phrase[seg_idx].note = safe_midi_to_note(int(round(curr)))

    return midi_segments


def create_midi_notes_from_pitched_data(
    start_times: list[float],
    end_times: list[float],
    words: list[str],
    pitched_data: PitchedData,
    allowed_notes: set[str] = None,
    disable_melismas: bool = False,
    melisma_threshold: float = 1.8,
    melisma_min_duration: float = 0.160,
) -> list[MidiSegment]:
    """Create midi notes from pitched data"""
    print(f"{ULTRASINGER_HEAD} Creating midi_segments")

    midi_segments = []

    for index, start_time in enumerate(start_times):
        end_time = end_times[index]
        word = str(words[index])
        dur = end_time - start_time

        if dur >= 0.28 and not disable_melismas:
            segs = segment_melismas_from_f0(
                start_time, end_time, word, pitched_data, allowed_notes,
                min_segment_duration=melisma_min_duration,
                pitch_semitone_threshold=melisma_threshold,
            )
            midi_segments.extend(segs)
        else:
            midi_segment = create_midi_note_from_pitched_data(start_time, end_time, pitched_data, word, allowed_notes)
            midi_segments.append(midi_segment)

    return correct_octave_jumps(midi_segments)


def create_midi_note_from_pitched_data(start_time: float, end_time: float, pitched_data: PitchedData, word: str,
                                        allowed_notes: set[str] = None) -> MidiSegment:
    """Create midi note from pitched data"""

    start = find_nearest_index(pitched_data.times, start_time)
    end = find_nearest_index(pitched_data.times, end_time)

    if start >= end:
        freqs = [pitched_data.frequencies[start]] if len(pitched_data.frequencies) > start else [440.0]
        confs = [pitched_data.confidence[start]] if len(pitched_data.confidence) > start else [1.0]
    else:
        freqs = pitched_data.frequencies[start:end]
        confs = pitched_data.confidence[start:end]

    valid_midi_pitches = []
    valid_weights = []
    for f, c in zip(freqs, confs):
        f_val = float(f)
        c_val = float(c)
        if 65.0 <= f_val <= 1100.0 and c_val >= 0.20:
            valid_midi_pitches.append(librosa.hz_to_midi(f_val))
            valid_weights.append(c_val ** 2)

    if valid_midi_pitches:
        if len(valid_midi_pitches) >= 2:
            mode_pitch = estimate_pitch_kde_mode(
                np.array(valid_midi_pitches),
                np.array(valid_weights),
                bandwidth_semitones=0.35,
            )
            note = safe_midi_to_note(int(round(mode_pitch)))
        else:
            note = safe_midi_to_note(int(round(valid_midi_pitches[0])))
    else:
        conf_f = get_frequencies_with_high_confidence(freqs, confs)
        valid_fallback = [float(f) for f in conf_f if 65.0 <= float(f) <= 1100.0]
        if valid_fallback:
            note = safe_hz_to_note(float(np.median(valid_fallback)))
        else:
            fallback_freq = pitched_data.frequencies[start] if len(pitched_data.frequencies) > start else 440.0
            note = safe_hz_to_note(float(fallback_freq))

    if allowed_notes is not None:
        note = quantize_note_to_key(note, allowed_notes)

    return MidiSegment(note, start_time, end_time, word)


def create_midi_segments_from_transcribed_data(
    transcribed_data: list[TranscribedData],
    pitched_data: PitchedData,
    allowed_notes: set[str] = None,
    disable_melismas: bool = False,
    melisma_threshold: float = 1.8,
    melisma_min_duration: float = 0.160,
    vibrato_freq_range: tuple[float, float] = (4.5, 7.5),
    vibrato_max_amplitude_st: float = 1.2,
) -> list[MidiSegment]:
    """Create MIDI segments from transcribed data (supporting melismas and line endings)."""
    print(f"{ULTRASINGER_HEAD} Creating midi_segments from transcribed data")
    midi_segments: list[MidiSegment] = []

    for data in transcribed_data:
        start_time = float(data.start)
        end_time = float(data.end)
        word = str(data.word)
        dur = end_time - start_time
        is_word_end = getattr(data, "is_word_end", word.endswith(" "))
        is_line_end = getattr(data, "is_line_end", False)

        if dur >= 0.28 and not disable_melismas:
            segs = segment_melismas_from_f0(
                start_time, end_time, word, pitched_data, allowed_notes,
                min_segment_duration=melisma_min_duration,
                pitch_semitone_threshold=melisma_threshold,
                vibrato_freq_range=vibrato_freq_range,
                vibrato_max_amplitude_st=vibrato_max_amplitude_st,
            )
        else:
            seg = create_midi_note_from_pitched_data(start_time, end_time, pitched_data, word, allowed_notes)
            segs = [seg]

        num_segs = len(segs)
        for s_idx, s in enumerate(segs):
            if s_idx < num_segs - 1:
                s.is_word_end = False
                s.is_line_end = False
                s.word = s.word.rstrip(" ")
            else:
                s.is_word_end = is_word_end
                s.is_line_end = is_line_end
                if is_word_end and not s.word.endswith(" "):
                    s.word += " "
                elif not is_word_end and s.word.endswith(" "):
                    s.word = s.word.rstrip(" ")

        midi_segments.extend(segs)

    midi_segments = correct_octave_jumps(midi_segments)
    return midi_segments


def create_repitched_midi_segments_from_ultrastar_txt(pitched_data: PitchedData, ultrastar_txt: UltrastarTxtValue) -> list[MidiSegment]:
    start_times = []
    end_times = []
    words = []

    for i, note_lines in enumerate(ultrastar_txt.UltrastarNoteLines):
        start_times.append(note_lines.startTime)
        end_times.append(note_lines.endTime)
        words.append(note_lines.word)
    midi_segments = create_midi_notes_from_pitched_data(start_times, end_times, words, pitched_data)
    return midi_segments


def create_midi_file(
        real_bpm: float,
        song_output: str,
        midi_segments: list[MidiSegment],
        basename_without_ext: str,
        audio_path: str = None,
        pitched_data: PitchedData = None,
) -> None:
    """Create midi file"""
    print(f"{ULTRASINGER_HEAD} Creating Midi with {blue_highlighted('pretty_midi')}")

    voice_instrument = [
        create_midi_instrument(midi_segments, audio_path=audio_path, pitched_data=pitched_data)
    ]

    midi_output = os.path.join(song_output, f"{basename_without_ext}.mid")
    __create_midi(voice_instrument, real_bpm, midi_output, midi_segments)
