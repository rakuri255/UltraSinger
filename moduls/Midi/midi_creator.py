import math
from collections import Counter

import librosa
import numpy as np
import pretty_midi

from moduls.Pitcher.pitcher import get_frequency_with_high_confidence
from moduls.Ultrastar.ultrastar_converter import ultrastar_note_to_midi_note, get_start_time_from_ultrastar, \
    get_end_time_from_ultrastar
from moduls.Log import PRINT_ULTRASTAR

def convert_ultrastar_to_midi_instrument(ultrastar_class):
    print(PRINT_ULTRASTAR + " Creating midi instrument from Ultrastar txt")

    instrument = pretty_midi.Instrument(program=0)
    velocity = 100

    for i in range(len(ultrastar_class.words)):
        start_time = get_start_time_from_ultrastar(ultrastar_class, i)
        end_time = get_end_time_from_ultrastar(ultrastar_class, i)
        pitch = ultrastar_note_to_midi_note(int(ultrastar_class.pitches[i]))

        note = pretty_midi.Note(velocity, pitch, start_time, end_time)
        instrument.notes.append(note)

    return instrument


def instruments_to_midi(instruments, bpm, midi_output):
    print(PRINT_ULTRASTAR + " Creating midi file -> {}".format(midi_output))

    midi_data = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    for instrument in instruments:
        midi_data.instruments.append(instrument)
    midi_data.write(midi_output)


class MidiCreator:
    pass


def convert_frequencies_to_notes(frequency):
    notes = []
    for f in frequency:
        notes.append(librosa.hz_to_note(float(f)))
    return notes


def most_frequent(ar):
    return Counter(ar).most_common(1)


def find_nearest_index(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
        return idx - 1
    else:
        return idx


def create_midi_notes_from_pitched_data(start_times, end_times, pitched_data):
    print(PRINT_ULTRASTAR + " Creating midi notes from pitched data")

    midi_notes = []

    for i in range(len(start_times)):
        start_time = start_times[i]
        end_time = end_times[i]

        note = create_midi_note_from_pitched_data(start_time, end_time, pitched_data)

        midi_notes.append(note)
        # todo: Progress?
        # print(filename + " f: " + str(mean))
    return midi_notes


def create_midi_note_from_pitched_data(start_time, end_time, pitched_data):
    s = find_nearest_index(pitched_data.times, start_time)
    e = find_nearest_index(pitched_data.times, end_time)

    if s == e:
        f = [pitched_data.frequencies[s]]
        c = [pitched_data.confidence[s]]
    else:
        f = pitched_data.frequencies[s:e]
        c = pitched_data.confidence[s:e]

    conf_f = get_frequency_with_high_confidence(f, c)

    notes = convert_frequencies_to_notes(conf_f)

    note = most_frequent(notes)[0][0]

    return note
