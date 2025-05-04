"""Midi creator module"""

import math
import os
from collections import Counter

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


def create_midi_instrument(midi_segments: list[MidiSegment]) -> object:
    """Converts an Ultrastar data to a midi instrument"""

    print(f"{ULTRASINGER_HEAD} Creating midi instrument")

    instrument = pretty_midi.Instrument(program=0, name="Vocals")
    velocity = 100

    for i, midi_segment in enumerate(midi_segments):
        # Skip if the note is empty or potentially invalid
        if not midi_segment.note:
            print(f"{ULTRASINGER_HEAD} [Warning] Skipping empty note at index {i}, time {midi_segment.start:.2f}s")
            continue

        try:
            midi_note_number = librosa.note_to_midi(midi_segment.note)
            note = pretty_midi.Note(velocity, midi_note_number, midi_segment.start, midi_segment.end)
            instrument.notes.append(note)
        except librosa.util.exceptions.ParameterError:
            print(f"{ULTRASINGER_HEAD} [Warning] Skipping invalid note format '{midi_segment.note}' at index {i}, time {midi_segment.start:.2f}s")
            continue

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


def convert_frequencies_to_notes(frequency: [str]) -> list[list[str]]:
    """Converts frequencies to notes"""
    notes = []
    for freq in frequency:
        notes.append(librosa.hz_to_note(float(freq)))
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


def create_midi_notes_from_pitched_data(start_times: list[float], end_times: list[float], words: list[str], pitched_data: PitchedData) -> list[
    MidiSegment]:
    """Create midi notes from pitched data"""
    print(f"{ULTRASINGER_HEAD} Creating midi_segments")

    midi_segments = []

    for index, start_time in enumerate(start_times):
        end_time = end_times[index]
        word = str(words[index])

        midi_segment = create_midi_note_from_pitched_data(start_time, end_time, pitched_data, word)
        if midi_segment is not None: # Filter out None results
            midi_segments.append(midi_segment)

        # todo: Progress?
        # print(filename + " f: " + str(mean))
    return midi_segments


def _get_pitch_indices(times: list[float], start_time: float, end_time: float) -> tuple[int | None, int | None]:
    """Find nearest indices in time array for start and end times."""
    start = find_nearest_index(times, start_time)
    end = find_nearest_index(times, end_time)
    return start, end

def _extract_pitch_data(pitched_data: PitchedData, start_index: int, end_index: int) -> tuple[list[float], list[float]]:
    """Extract frequencies and confidences for the given index range."""
    # Handle edge case where start and end indices are the same
    if start_index == end_index:
        # Ensure index is within bounds before accessing
        if start_index < len(pitched_data.frequencies):
            freqs = [pitched_data.frequencies[start_index]]
            confs = [pitched_data.confidence[start_index]]
        else:
            return [], [] # Return empty lists if index is out of bounds
    else:
        # Ensure slice indices are valid
        freqs = pitched_data.frequencies[start_index:end_index]
        confs = pitched_data.confidence[start_index:end_index]
    return freqs, confs

def _get_confident_frequencies(frequencies: list[float], confidences: list[float]) -> list[float]:
    """Filter frequencies based on high confidence."""
    return get_frequencies_with_high_confidence(frequencies, confidences)

def _frequencies_to_notes(frequencies: list[float], word: str) -> list[str]:
    """Convert a list of frequencies to musical notes, handling errors."""
    valid_notes = []
    for freq in frequencies:
        try:
            # Ensure frequency is positive before conversion
            if freq > 0:
                note = librosa.hz_to_note(float(freq))
                # Basic validation for note format (e.g., 'C4', 'A#3')
                if note and len(note) >= 2 and note[0].isalpha() and note[-1].isdigit():
                    valid_notes.append(note)
            # else: # Optional: Log zero/negative frequencies if needed
            #     print(f"{ULTRASINGER_HEAD} [Debug] Skipping non-positive frequency {freq} for word '{word}'.")
        except (ValueError, TypeError) as e:
            print(f"{ULTRASINGER_HEAD} [Warning] Error converting frequency {freq} to note for word '{word}': {e}. Skipping this frequency.")
            continue
    return valid_notes

def _determine_most_common_note(notes: list[str], word: str) -> str | None:
    """Determine the most common note from a list of notes."""
    if not notes:
        # print(f"{ULTRASINGER_HEAD} [Debug] No valid notes provided for word '{word}'. Cannot determine most common note.")
        return None

    most_common = most_frequent(notes)
    # Ensure most_common is not empty and has a valid note string
    if not most_common or not most_common[0] or not most_common[0][0]:
        # print(f"{ULTRASINGER_HEAD} [Debug] Could not determine most common note for word '{word}'.")
        return None
    return most_common[0][0]

def _get_midi_number(note: str, word: str) -> int | None:
    """Convert a note string to its MIDI number."""
    try:
        return librosa.note_to_midi(note)
    except librosa.util.exceptions.ParameterError:
        print(f"{ULTRASINGER_HEAD} [Warning] Could not convert determined note '{note}' to MIDI number for word '{word}'. Skipping MIDI number assignment.")
        return None

def create_midi_note_from_pitched_data(start_time: float, end_time: float, pitched_data: PitchedData, word: str) -> MidiSegment | None:
    """Create midi note from pitched data. Returns None if no valid note can be determined."""

    start_index, end_index = _get_pitch_indices(pitched_data.times, start_time, end_time)

    # Check if indices are valid
    if start_index is None or end_index is None or start_index >= len(pitched_data.frequencies) or end_index > len(pitched_data.frequencies) or start_index < 0 or end_index < 0:
        print(f"{ULTRASINGER_HEAD} [Warning] Invalid time range or indices for word '{word}' ({start_time:.2f}s - {end_time:.2f}s). Skipping note creation.")
        return None

    frequencies, confidences = _extract_pitch_data(pitched_data, start_index, end_index)

    # Check if frequency/confidence lists are empty
    if not frequencies or not confidences:
        print(f"{ULTRASINGER_HEAD} [Warning] No frequency data found for word '{word}' in time range ({start_time:.2f}s - {end_time:.2f}s). Skipping note creation.")
        return None

    # Get frequencies with high confidence
    confident_frequencies = _get_confident_frequencies(frequencies, confidences)
    if not confident_frequencies:
        # print(f"{ULTRASINGER_HEAD} [Debug] No confident frequencies found for word '{word}'. Skipping note creation.")
        return None

    # Convert frequencies to notes
    valid_notes = _frequencies_to_notes(confident_frequencies, word)
    if not valid_notes:
        # print(f"{ULTRASINGER_HEAD} [Debug] No valid notes derived from confident frequencies for word '{word}'. Skipping note creation.")
        return None

    # Determine the most common note
    final_note = _determine_most_common_note(valid_notes, word)
    if final_note is None:
        # print(f"{ULTRASINGER_HEAD} [Debug] Could not determine final note for word '{word}'. Skipping note creation.")
        return None

    # Calculate MIDI note number
    midi_note_number = _get_midi_number(final_note, word)

    # print(f"{ULTRASINGER_HEAD} [Debug] Created note '{final_note}' for word '{word}' ({start_time:.2f}s - {end_time:.2f}s)")
    return MidiSegment(final_note, start_time, end_time, word, midi_note=midi_note_number)


def create_midi_segments_from_transcribed_data(transcribed_data: list[TranscribedData], pitched_data: PitchedData) -> list[MidiSegment]:
    start_times = []
    end_times = []
    words = []

    if transcribed_data:
        for i, midi_segment in enumerate(transcribed_data):
            start_times.append(midi_segment.start)
            end_times.append(midi_segment.end)
            words.append(midi_segment.word)
        midi_segments = create_midi_notes_from_pitched_data(start_times, end_times, words,
                                                            pitched_data)
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
