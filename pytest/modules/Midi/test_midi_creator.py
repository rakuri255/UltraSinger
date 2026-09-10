"""Tests for the midi_creator.py module.

Regression tests for degenerate MidiSegments: UltraSinger's syllable splitting and
segment merging can produce segments whose `end` is equal to or earlier than their
`start`. Such a segment used to crash the whole run when writing the MIDI file:

    ValueError: Note end time must be greater than start time
    (pretty_midi/containers.py, Note.__init__)

Note on pretty_midi's semantics (measured, pretty_midi 0.2.11):

    pretty_midi.Note(100, 60, 3.0, 3.0)  -> accepted   (end == start)
    pretty_midi.Note(100, 60, 5.0, 4.0)  -> ValueError (end <  start)

The crash is therefore only triggered by `end < start`, even though the exception
text asks for *greater than*. The MIDI writer guards on `end <= start` on purpose: a
zero-length note carries no information and the same root cause produces invalid
`duration <= 0` lines in the sibling UltraStar output. The MIDI writer must tolerate
both instead of aborting the run.
"""

import unittest

from src.modules.Midi.MidiSegment import MidiSegment
from src.modules.Midi.midi_creator import create_midi_instrument


class TestCreateMidiInstrument(unittest.TestCase):

    def test_keeps_valid_segments(self):
        # Arrange
        midi_segments = [
            MidiSegment(note="C4", start=1.0, end=2.0, word="a"),
            MidiSegment(note="D4", start=2.5, end=4.0, word="b"),
        ]

        # Act
        instrument = create_midi_instrument(midi_segments)

        # Assert
        self.assertEqual(2, len(instrument.notes))
        self.assertEqual([1.0, 2.5], [note.start for note in instrument.notes])

    def test_skips_zero_length_segment(self):
        # Arrange: end == start. pretty_midi accepts this, but a zero-length note is a
        # no-op; the same root cause also yields invalid `duration 0` UltraStar lines.
        midi_segments = [
            MidiSegment(note="C4", start=1.0, end=2.0, word="a"),
            MidiSegment(note="C4", start=3.0, end=3.0, word="b"),
        ]

        # Act
        instrument = create_midi_instrument(midi_segments)

        # Assert: the degenerate note is dropped, the valid one survives
        self.assertEqual(1, len(instrument.notes))
        self.assertEqual(1.0, instrument.notes[0].start)

    def test_skips_negative_length_segment(self):
        # Arrange: end < start (seen in real runs as word='~' start=155.25 end=153.78)
        midi_segments = [
            MidiSegment(note="C4", start=1.0, end=2.0, word="a"),
            MidiSegment(note="E4", start=155.2537142857143, end=153.776, word="~"),
        ]

        # Act
        instrument = create_midi_instrument(midi_segments)

        # Assert
        self.assertEqual(1, len(instrument.notes))

    def test_all_segments_degenerate_does_not_raise(self):
        # Arrange
        midi_segments = [
            MidiSegment(note="C4", start=5.0, end=5.0, word="a"),
            MidiSegment(note="C4", start=9.0, end=8.0, word="b"),
        ]

        # Act
        instrument = create_midi_instrument(midi_segments)

        # Assert
        self.assertEqual(0, len(instrument.notes))


if __name__ == "__main__":
    unittest.main()
