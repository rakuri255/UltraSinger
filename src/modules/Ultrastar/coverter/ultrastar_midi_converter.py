import librosa
import pretty_midi

from modules.Midi.MidiSegment import MidiSegment
from modules.Ultrastar.coverter.ultrastar_converter import midi_note_to_ultrastar_note, ultrastar_note_to_midi_note, \
    get_start_time_from_ultrastar, get_end_time_from_ultrastar
from modules.Ultrastar.ultrastar_txt import UltrastarTxtValue
from modules.console_colors import ULTRASINGER_HEAD


def convert_midi_note_to_ultrastar_note(midi_segment: MidiSegment) -> int:
    """Convert midi notes to ultrastar notes"""

    note_number_librosa = librosa.note_to_midi(midi_segment.note)
    ultrastar_note = midi_note_to_ultrastar_note(note_number_librosa)
    return ultrastar_note


def convert_ultrastar_to_midi_instrument(ultrastar_class: UltrastarTxtValue) -> object:
    """Converts an Ultrastar data to a midi instrument"""
    # Todo: delete?

    print(f"{ULTRASINGER_HEAD} Creating midi instrument from Ultrastar txt")

    instrument = pretty_midi.Instrument(program=0, name="Vocals")
    velocity = 100

    for i, note_line in enumerate(ultrastar_class.UltrastarNoteLines):
        note = pretty_midi.Note(velocity, ultrastar_note_to_midi_note(note_line.pitch), note_line.startTime, note_line.endTime)
        instrument.notes.append(note)

    return instrument


def convert_midi_notes_to_ultrastar_notes(midi_segments: list[MidiSegment]) -> list[int]:
    """Convert midi notes to ultrastar notes"""
    print(f"{ULTRASINGER_HEAD} Creating Ultrastar notes from midi data")

    ultrastar_note_numbers = []
    for i, midi_segment in enumerate(midi_segments):
        ultrastar_note = convert_midi_note_to_ultrastar_note(midi_segment)
        ultrastar_note_numbers.append(ultrastar_note)
        # todo: Progress?
        # print(
        #    f"Note: {midi_notes[i]} midi_note: {str(note_number_librosa)} pitch: {str(pitch)}"
        # )
    return ultrastar_note_numbers


def ultrastar_to_midi_segments(ultrastar_txt: UltrastarTxtValue) -> list[MidiSegment]:
    """Converts an Ultrastar txt to Midi segments"""
    midi_segments = []
    for i, data in enumerate(ultrastar_txt.UltrastarNoteLines):
        start_time = get_start_time_from_ultrastar(ultrastar_txt, i)
        end_time = get_end_time_from_ultrastar(ultrastar_txt, i)
        midi_segments.append(
            MidiSegment(librosa.midi_to_note(ultrastar_note_to_midi_note(data.pitch)),
                        start_time,
                        end_time,
                        data.word,
                        )
        )
    return midi_segments
