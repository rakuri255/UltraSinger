import pretty_midi
from moduls.Ultrastar.ultrastar_converter import ultrastar_note_to_midi_note, get_start_time_from_ultrastar, get_end_time_from_ultrastar


def convert_ultrastar_to_midi_instrument(ultrastar_class):
    print("Creating midi instrument from Ultrastar txt")

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
    print("Creating midi file -> {}".format(midi_output))

    midi_data = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    for instrument in instruments:
        midi_data.instruments.append(instrument)
    midi_data.write(midi_output)


class MidiCreator:
    pass
