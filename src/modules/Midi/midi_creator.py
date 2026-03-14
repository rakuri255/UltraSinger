"""Midi creator module"""

import math
import os

import librosa
import numpy as np
import pretty_midi
import unidecode

from modules.Midi.MidiSegment import MidiSegment
from modules.Speech_Recognition.TranscribedData import TranscribedData
from modules.console_colors import (
    ULTRASINGER_HEAD, blue_highlighted,
)
from modules.Ultrastar.ultrastar_txt import UltrastarTxtValue
from modules.Pitcher.pitched_data import PitchedData
from modules.Pitcher.pitched_data_helper import get_frequencies_with_high_confidence
from modules.Audio.key_detector import quantize_note_to_key

def create_midi_instrument(midi_segments: list[MidiSegment]) -> object:
    """Converts an Ultrastar data to a midi instrument"""

    print(f"{ULTRASINGER_HEAD} Creating midi instrument")

    instrument = pretty_midi.Instrument(program=0, name="Vocals")
    velocity = 100

    for i, midi_segment in enumerate(midi_segments):
        note = pretty_midi.Note(velocity, librosa.note_to_midi(midi_segment.note), midi_segment.start, midi_segment.end)
        instrument.notes.append(note)

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


def confidence_weighted_median_note(
    frequencies: list[float], weights: list[float]
) -> str:
    """Select a note using the confidence-weighted median of frequencies.

    Instead of converting frequencies to note names first and picking the
    mode (most common), this operates on raw Hz values so that the median
    is computed in a continuous space.  This avoids the problem where
    pitch jitter across note boundaries causes the mode to pick a rare
    outlier note.

    Args:
        frequencies: Detected frequencies in Hz (already confidence-filtered).
        weights: Confidence values corresponding to each frequency.

    Returns:
        Note name string (e.g. ``"C4"``).

    Raises:
        ValueError: If *frequencies* or *weights* are empty, have
            mismatched lengths, or all weights are zero.
    """
    if not frequencies or not weights:
        raise ValueError("frequencies and weights must be non-empty")
    if len(frequencies) != len(weights):
        raise ValueError(
            f"frequencies ({len(frequencies)}) and weights ({len(weights)}) "
            "must have the same length"
        )

    freqs = np.asarray(frequencies, dtype=float)
    wts = np.asarray(weights, dtype=float)

    total_weight = wts.sum()
    if total_weight == 0:
        raise ValueError("total weight must be > 0")

    # Sort by frequency
    order = np.argsort(freqs)
    sorted_freqs = freqs[order]
    sorted_wts = wts[order]

    # Weighted median: first cumulative weight index >= half total weight
    cumulative = np.cumsum(sorted_wts)
    half = total_weight / 2.0
    median_idx = int(np.searchsorted(cumulative, half))
    # Clamp to valid index range
    median_idx = min(median_idx, len(sorted_freqs) - 1)
    median_freq = sorted_freqs[median_idx]

    return librosa.hz_to_note(float(median_freq))


def find_nearest_index(array: list[float], value: float) -> int:
    """Nearest index in array"""
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (
        idx == len(array)
        or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])
    ):
        return idx - 1

    return idx


def create_midi_notes_from_pitched_data(start_times: list[float], end_times: list[float], words: list[str],
                                         pitched_data: PitchedData, allowed_notes: set[str] = None) -> list[MidiSegment]:
    """Create midi notes from pitched data

    Args:
        start_times: List of start times
        end_times: List of end times
        words: List of words/syllables
        pitched_data: Pitched data containing frequencies and confidence
        allowed_notes: Optional set of allowed note names for key quantization

    Returns:
        List of MidiSegments
    """
    print(f"{ULTRASINGER_HEAD} Creating midi_segments")

    midi_segments = []

    for index, start_time in enumerate(start_times):
        end_time = end_times[index]
        word = str(words[index])

        midi_segment = create_midi_note_from_pitched_data(start_time, end_time, pitched_data, word, allowed_notes)
        midi_segments.append(midi_segment)

        # todo: Progress?
        # print(filename + " f: " + str(mean))
    return midi_segments


def create_midi_note_from_pitched_data(start_time: float, end_time: float, pitched_data: PitchedData, word: str,
                                        allowed_notes: set[str] = None) -> MidiSegment:
    """Create midi note from pitched data

    Args:
        start_time: Start time of the note
        end_time: End time of the note
        pitched_data: Pitched data containing frequencies and confidence
        word: The word/syllable for this note
        allowed_notes: Optional set of allowed note names (without octave) for key quantization

    Returns:
        One MidiSegment
    """

    start = find_nearest_index(pitched_data.times, start_time)
    end = find_nearest_index(pitched_data.times, end_time)

    if start == end:
        freqs = [pitched_data.frequencies[start]]
        confs = [pitched_data.confidence[start]]
    else:
        freqs = pitched_data.frequencies[start:end]
        confs = pitched_data.confidence[start:end]

    conf_f, conf_weights = get_frequencies_with_high_confidence(freqs, confs)

    if not conf_f:
        # No valid frequencies found; fall back to a neutral middle note
        note = "C4"
    else:
        note = confidence_weighted_median_note(conf_f, conf_weights)

    if allowed_notes is not None:
        note = quantize_note_to_key(note, allowed_notes)

    return MidiSegment(note, start_time, end_time, word)


def create_midi_segments_from_transcribed_data(transcribed_data: list[TranscribedData], pitched_data: PitchedData,
                                                allowed_notes: set[str] = None) -> list[MidiSegment]:
    """Create MIDI segments from transcribed data

    Args:
        transcribed_data: List of transcribed data segments
        pitched_data: Pitched data containing frequencies and confidence
        allowed_notes: Optional set of allowed note names for key quantization

    Returns:
        List of MidiSegments
    """
    start_times = []
    end_times = []
    words = []

    if transcribed_data:
        for i, midi_segment in enumerate(transcribed_data):
            start_times.append(midi_segment.start)
            end_times.append(midi_segment.end)
            words.append(midi_segment.word)
        midi_segments = create_midi_notes_from_pitched_data(start_times, end_times, words,
                                                            pitched_data, allowed_notes)
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


def correct_vocal_center(
    midi_segments: list[MidiSegment],
    low_threshold: int = 55,
    high_threshold: int = 79,
    concentration_pct: float = 0.80,
) -> list[MidiSegment]:
    """Safety-net octave correction for consistently wrong-octave detection.

    :func:`correct_global_octave` catches cases where the median falls
    *outside* the broad vocal range (MIDI 48-84).  However, when *all*
    detected pitches are consistently one octave too low (or too high),
    the median can land just inside that range, and
    :func:`correct_octave_outliers` cannot help either because there
    are no outliers to compare against.

    This function uses a **tighter** expected centre band (default
    MIDI 55-79, roughly G3-G5) and a concentration threshold.  If more
    than *concentration_pct* of all notes cluster below *low_threshold*
    (or above *high_threshold*), it shifts **all** notes by one octave
    toward the centre.

    Run this **after** both :func:`correct_global_octave` and
    :func:`correct_octave_outliers` as a final safety net.

    Args:
        midi_segments: List of MIDI segments with ``.note`` attributes.
        low_threshold: Notes below this are considered suspiciously low.
        high_threshold: Notes above this are considered suspiciously high.
        concentration_pct: Minimum fraction of notes that must be
            outside the threshold to trigger a shift (0.0-1.0).

    Returns:
        The same list, with notes shifted in-place when triggered.
    """
    if not midi_segments:
        return midi_segments

    midi_values: list[int] = []
    for seg in midi_segments:
        try:
            midi_values.append(librosa.note_to_midi(seg.note))
        except (ValueError, TypeError):
            pass

    if not midi_values:
        return midi_segments

    median_midi = float(np.median(midi_values))

    # Already in the expected centre band — nothing to do
    if low_threshold <= median_midi <= high_threshold:
        return midi_segments

    total = len(midi_values)

    # Check for suspiciously low concentration
    if median_midi < low_threshold:
        below = sum(1 for v in midi_values if v < low_threshold)
        if below / total > concentration_pct:
            shift = 12
            print(
                f"{ULTRASINGER_HEAD} Vocal-centre correction: "
                f"shifting all notes by {blue_highlighted('+12')} semitones "
                f"(median MIDI {median_midi:.0f}, "
                f"{below}/{total} notes below {low_threshold})"
            )
            for seg in midi_segments:
                try:
                    current = librosa.note_to_midi(seg.note)
                    seg.note = librosa.midi_to_note(current + shift)
                except (ValueError, TypeError):
                    pass
            return midi_segments

    # Check for suspiciously high concentration
    if median_midi > high_threshold:
        above = sum(1 for v in midi_values if v > high_threshold)
        if above / total > concentration_pct:
            shift = -12
            print(
                f"{ULTRASINGER_HEAD} Vocal-centre correction: "
                f"shifting all notes by {blue_highlighted('-12')} semitones "
                f"(median MIDI {median_midi:.0f}, "
                f"{above}/{total} notes above {high_threshold})"
            )
            for seg in midi_segments:
                try:
                    current = librosa.note_to_midi(seg.note)
                    seg.note = librosa.midi_to_note(current + shift)
                except (ValueError, TypeError):
                    pass
            return midi_segments

    return midi_segments


def apply_octave_shift(
    midi_segments: list[MidiSegment],
    octaves: int,
) -> list[MidiSegment]:
    """Shift all notes by a fixed number of octaves.

    This is a manual override for cases where automatic octave correction
    fails (e.g. when the pitch detector consistently detects the wrong
    octave but the median still falls in the expected vocal range).

    Args:
        midi_segments: List of MIDI segments with ``.note`` attributes.
        octaves: Number of octaves to shift (positive = up, negative = down).

    Returns:
        The same list, with notes shifted in-place.
    """
    if not midi_segments or octaves == 0:
        return midi_segments

    shift = octaves * 12

    print(
        f"{ULTRASINGER_HEAD} Manual octave shift: "
        f"shifting all notes by {blue_highlighted(f'{octaves:+d}')} octave(s) "
        f"({shift:+d} semitones)"
    )

    for seg in midi_segments:
        try:
            current = librosa.note_to_midi(seg.note)
            new_midi = current + shift
            if 0 <= new_midi <= 127:
                seg.note = librosa.midi_to_note(new_midi)
        except (ValueError, TypeError):
            print(f"{ULTRASINGER_HEAD} Skipping invalid note in octave shift: {seg.note!r}")

    return midi_segments


def correct_global_octave(
    midi_segments: list[MidiSegment],
    low: int = 48,
    high: int = 84,
) -> list[MidiSegment]:
    """Shift all notes by whole octaves so the median lies in the vocal range.

    Pitch detectors sometimes lock onto a sub-harmonic (e.g. half the
    fundamental frequency), pushing *every* note into an implausibly low
    register.  :func:`correct_octave_outliers` only fixes *local*
    outliers relative to their neighbours and cannot detect such a
    *global* shift.

    This function computes the median MIDI value of the entire song.
    If that median falls outside the expected vocal range
    (default ``48``-``84``, i.e. C3-C6), it shifts **all** notes by
    the smallest multiple of 12 semitones needed to bring the median
    inside the range.

    Args:
        midi_segments: List of MIDI segments with ``.note`` attributes.
        low: Lower bound of the expected vocal MIDI range (inclusive).
        high: Upper bound of the expected vocal MIDI range (inclusive).

    Returns:
        The same list, with notes corrected in-place.
    """
    if not midi_segments:
        return midi_segments

    midi_values = []
    for seg in midi_segments:
        try:
            midi_values.append(librosa.note_to_midi(seg.note))
        except (ValueError, TypeError):
            pass

    if not midi_values:
        return midi_segments

    median_midi = float(np.median(midi_values))

    # Already in range — nothing to do
    if low <= median_midi <= high:
        return midi_segments

    # Find the smallest octave shift that brings the median into range
    if median_midi < low:
        shift = int(np.ceil((low - median_midi) / 12)) * 12
    else:
        shift = -int(np.ceil((median_midi - high) / 12)) * 12

    print(
        f"{ULTRASINGER_HEAD} Global octave correction: "
        f"shifting all notes by {blue_highlighted(f'{shift:+d}')} semitones "
        f"(median was MIDI {median_midi:.0f}, target range {low}-{high})"
    )

    for seg in midi_segments:
        try:
            current = librosa.note_to_midi(seg.note)
            seg.note = librosa.midi_to_note(current + shift)
        except (ValueError, TypeError):
            pass

    return midi_segments


def correct_octave_outliers(
    midi_segments: list[MidiSegment],
    window: int = 5,
    passes: int = 2,
) -> list[MidiSegment]:
    """Correct octave outliers by comparing each note to its neighbours.

    Pitch detectors occasionally jump a note into the wrong octave
    (e.g. C2 instead of C4).  This post-processing step shifts any note
    that is more than 11 semitones away from the local median back to the
    closest octave.

    The correction operates in two phases:

    **Phase 1 — Local passes:**  Multiple passes compare each note to
    the median of its *window* neighbours.  Notes more than 11 semitones
    away are shifted by whole octaves toward the local median.  When
    several candidates are equally valid, the one closest to the *global*
    median wins.  Multiple passes help because corrections from earlier
    passes update neighbour values for later passes.

    **Phase 2 — Global consensus:**  After local passes, some clusters
    may remain uncorrected because the local neighbourhood was dominated
    by wrong-octave notes (contaminated median).  When the majority of
    notes agree on a particular octave region, this phase shifts notes
    that are a full octave or more (≥ 12 semitones) from the global
    median by whole octaves toward it.  Legitimate melody intervals
    (< 12 semitones, e.g. a fifth or sixth) are never touched.
    This only activates when a clear majority exists (> 50 % of notes
    near the global median).

    Args:
        midi_segments: List of MIDI segments with ``.note`` attributes.
        window: Number of neighbours on each side to consider.
        passes: Number of local correction passes (default 2).  Use
            ``1`` for the legacy single-pass behaviour.

    Returns:
        The same list, with outlier notes corrected in-place.
    """
    if len(midi_segments) < 3:
        return midi_segments

    print(f"{ULTRASINGER_HEAD} Correcting octave outliers ({passes} pass{'es' if passes != 1 else ''})")

    # ── Phase 1: Local outlier correction ────────────────────────────
    for _pass_num in range(passes):
        # Rebuild MIDI values from current notes at the start of each pass
        midi_values: list[int | None] = []
        for seg in midi_segments:
            try:
                midi_values.append(librosa.note_to_midi(seg.note))
            except (ValueError, KeyError):
                midi_values.append(None)

        # Compute global median once per pass as tie-breaker reference
        valid_values = [v for v in midi_values if v is not None]
        if not valid_values:
            break
        global_median = float(np.median(valid_values))

        # Collect all corrections first, then apply them after the scan.
        # This prevents earlier corrections from leaking into the
        # neighbourhood of later notes within the same pass.
        pending_updates: list[tuple[int, int]] = []

        for i, seg in enumerate(midi_segments):
            if midi_values[i] is None:
                continue

            # Collect valid neighbour MIDI values
            lo = max(0, i - window)
            hi = min(len(midi_segments), i + window + 1)
            neighbours = [
                midi_values[j]
                for j in range(lo, hi)
                if j != i and midi_values[j] is not None
            ]

            if not neighbours:
                continue

            median_neighbour = float(np.median(neighbours))
            current = midi_values[i]

            # Check if note is an outlier (> 11 semitones from local median)
            if abs(current - median_neighbour) <= 11:
                continue

            # Generate all octave-shifted candidates within 11 ST of local median
            candidates = []
            shifted = current
            # Shift down
            while shifted > median_neighbour - 12:
                if abs(shifted - median_neighbour) <= 11:
                    candidates.append(shifted)
                shifted -= 12
            # Shift up from original
            shifted = current + 12
            while shifted < median_neighbour + 12:
                if abs(shifted - median_neighbour) <= 11:
                    candidates.append(shifted)
                shifted += 12

            if not candidates:
                # Fallback: simple shift toward local median
                shifted = current
                while abs(shifted - median_neighbour) > 11:
                    if shifted > median_neighbour:
                        shifted -= 12
                    else:
                        shifted += 12
                candidates = [shifted]

            # Pick candidate closest to global median (tie-breaker)
            best = min(candidates, key=lambda x: abs(x - global_median))

            if best != midi_values[i]:
                pending_updates.append((i, best))

        # Apply all corrections from this pass at once
        for i, best in pending_updates:
            midi_segments[i].note = librosa.midi_to_note(best)

        if not pending_updates:
            # No changes this pass — further passes won't help
            break

    # ── Phase 2: Global consensus correction ─────────────────────────
    # Fixes clusters that local correction couldn't handle because the
    # local neighbourhood median was contaminated by wrong-octave notes.
    midi_values_ph2: list[int | None] = []
    for seg in midi_segments:
        try:
            midi_values_ph2.append(librosa.note_to_midi(seg.note))
        except (ValueError, KeyError):
            midi_values_ph2.append(None)

    valid_values = [v for v in midi_values_ph2 if v is not None]
    if len(valid_values) >= 3:
        global_median = float(np.median(valid_values))

        # Only apply when there is a clear majority near the global median
        near_global = sum(
            1 for v in valid_values if abs(v - global_median) <= 6
        )
        if near_global / len(valid_values) > 0.5:
            consensus_corrections = 0
            for i, seg in enumerate(midi_segments):
                if midi_values_ph2[i] is None:
                    continue
                current = midi_values_ph2[i]

                # Only target notes that are a full octave or more from the
                # global median — these are true octave errors.  Notes
                # within 11 semitones are legitimate melody intervals
                # (e.g. G4 is 7 ST from C4 and must NOT be shifted).
                if abs(current - global_median) < 12:
                    continue

                # Try octave shifts (±12, ±24) to get closer to global median
                best = current
                for shift in [-24, -12, 12, 24]:
                    candidate = current + shift
                    if abs(candidate - global_median) < abs(best - global_median):
                        best = candidate

                # Only apply if the shift brings the note reasonably close
                if best != current and abs(best - global_median) <= 11:
                    seg.note = librosa.midi_to_note(best)
                    midi_values_ph2[i] = best
                    consensus_corrections += 1

            if consensus_corrections > 0:
                print(
                    f"{ULTRASINGER_HEAD} Global consensus: corrected "
                    f"{blue_highlighted(str(consensus_corrections))} "
                    f"cluster note{'s' if consensus_corrections != 1 else ''}"
                )

    return midi_segments


def create_midi_file(
        real_bpm: float,
        song_output: str,
        midi_segments: list[MidiSegment],
        basename_without_ext: str,
) -> None:
    """Create midi file"""
    print(f"{ULTRASINGER_HEAD} Creating Midi with {blue_highlighted('pretty_midi')}")

    voice_instrument = [
        create_midi_instrument(midi_segments)
    ]

    midi_output = os.path.join(song_output, f"{basename_without_ext}.mid")
    __create_midi(voice_instrument, real_bpm, midi_output, midi_segments)
