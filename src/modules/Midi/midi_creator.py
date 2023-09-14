"""Midi creator module"""

import math
from collections import Counter

import librosa
import numpy as np
import pretty_midi

from modules.Pitcher.pitcher import get_frequencies_with_high_confidence
from modules.Ultrastar.ultrastar_converter import (
    get_end_time_from_ultrastar,
    get_start_time_from_ultrastar,
    ultrastar_note_to_midi_note,
)
from modules.console_colors import (
    ULTRASINGER_HEAD,
    red_highlighted,
)
from modules.Ultrastar.ultrastar_txt import UltrastarTxtValue
from modules.Pitcher.pitched_data import PitchedData


def convert_ultrastar_to_midi_instrument(ultrastar_class: UltrastarTxtValue) -> object:
    """Converts an Ultrastar data to a midi instrument"""

    print(f"{ULTRASINGER_HEAD} Creating midi instrument from Ultrastar txt")

    instrument = pretty_midi.Instrument(program=0)
    velocity = 100

    for i in enumerate(ultrastar_class.words):
        pos = i[0]
        start_time = get_start_time_from_ultrastar(ultrastar_class, pos)
        end_time = get_end_time_from_ultrastar(ultrastar_class, pos)
        pitch = ultrastar_note_to_midi_note(int(ultrastar_class.pitches[pos]))

        note = pretty_midi.Note(velocity, pitch, start_time, end_time)
        instrument.notes.append(note)

    return instrument


def instruments_to_midi(instruments: list[object], bpm: float, midi_output: str) -> None:
    """Write instruments to midi file"""

    print(f"{ULTRASINGER_HEAD} Creating midi file -> {midi_output}")

    midi_data = pretty_midi.PrettyMIDI(initial_tempo=bpm)
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


def create_midi_notes_from_pitched_data(start_times: list[float], end_times: list[float], pitched_data: PitchedData) -> list[str]:
    """Create midi notes from pitched data"""
    print(f"{ULTRASINGER_HEAD} Creating midi notes from pitched data")

    midi_notes = []

    for i in enumerate(start_times):
        pos = i[0]
        start_time = start_times[pos]
        end_time = end_times[pos]

        note = create_midi_note_from_pitched_data(
            start_time, end_time, pitched_data
        )

        midi_notes.append(note)
        # todo: Progress?
        # print(filename + " f: " + str(mean))
    return midi_notes


def create_midi_note_from_pitched_data(start_time: float, end_time: float, pitched_data: PitchedData) -> str:
    """Create midi note from pitched data"""

    start = find_nearest_index(pitched_data.times, start_time)
    end = find_nearest_index(pitched_data.times, end_time)

    if start == end:
        freqs = [pitched_data.frequencies[start]]
        confs = [pitched_data.confidence[start]]
    else:
        freqs = pitched_data.frequencies[start:end]
        confs = pitched_data.confidence[start:end]

    conf_f = get_frequencies_with_high_confidence(freqs, confs)

    notes = convert_frequencies_to_notes(conf_f)

    note = most_frequent(notes)[0][0]

    return note
