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
