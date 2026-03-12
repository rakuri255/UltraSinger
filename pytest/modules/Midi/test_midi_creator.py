"""Tests for midi_creator.py — global and local octave correction."""

import unittest

import librosa
import numpy as np

from src.modules.Midi.MidiSegment import MidiSegment
from src.modules.Midi.midi_creator import (
    correct_global_octave,
    confidence_weighted_median_note,
)


class TestCorrectGlobalOctave(unittest.TestCase):
    """Tests for correct_global_octave()."""

    # -- edge cases ----------------------------------------------------------

    def test_empty_list_returns_empty(self):
        result = correct_global_octave([])
        self.assertEqual(result, [])

    def test_single_note_in_range_unchanged(self):
        seg = MidiSegment(note="C4", start=0, end=1, word="hello ")
        result = correct_global_octave([seg])
        self.assertEqual(result[0].note, "C4")

    # -- already in range (no shift) -----------------------------------------

    def test_notes_in_range_unchanged(self):
        """Notes whose median is inside 48-84 should not be shifted."""
        segs = [
            MidiSegment(note="C4", start=0, end=1, word="one "),   # MIDI 60
            MidiSegment(note="E4", start=1, end=2, word="two "),   # MIDI 64
            MidiSegment(note="G4", start=2, end=3, word="three "), # MIDI 67
        ]
        result = correct_global_octave(segs)
        notes = [s.note for s in result]
        self.assertEqual(notes, ["C4", "E4", "G4"])

    # -- shift up (sub-harmonic detection) ------------------------------------

    def test_all_notes_two_octaves_low_shifted_up(self):
        """Median MIDI ~36 (C2) -> needs +24 shift to reach 48+ range."""
        segs = [
            MidiSegment(note="C2", start=0, end=1, word="a "),   # MIDI 36
            MidiSegment(note="D2", start=1, end=2, word="b "),   # MIDI 38
            MidiSegment(note="E2", start=2, end=3, word="c "),   # MIDI 40
        ]
        result = correct_global_octave(segs)
        # All should be shifted up by +12 (one octave brings median to 48-50)
        midi_values = [librosa.note_to_midi(s.note) for s in result]
        for val in midi_values:
            self.assertGreaterEqual(val, 48)
            self.assertLessEqual(val, 84)

    def test_all_notes_four_octaves_low_shifted_up(self):
        """Simulate a real-world sub-harmonic detection problem: MIDI 21-26.

        The function shifts all notes so the *median* lands inside the
        vocal range.  Individual notes may still be below ``low`` if
        they were already the lowest in the set.
        """
        segs = [
            MidiSegment(note="D1", start=0, end=1, word="wake "),   # MIDI 26
            MidiSegment(note="A0", start=1, end=2, word="me "),     # MIDI 21
            MidiSegment(note="C1", start=2, end=3, word="up "),     # MIDI 24
            MidiSegment(note="B0", start=3, end=4, word="in "),     # MIDI 23
            MidiSegment(note="D1", start=4, end=5, word="side "),   # MIDI 26
        ]
        result = correct_global_octave(segs)
        midi_values = [librosa.note_to_midi(s.note) for s in result]
        median_after = float(np.median(midi_values))
        # Median must be within the default vocal range
        self.assertGreaterEqual(median_after, 48)
        self.assertLessEqual(median_after, 84)
        # All notes must have been shifted up (original median was 24)
        for val in midi_values:
            self.assertGreater(val, 24)

    # -- shift down (over-detection) ------------------------------------------

    def test_all_notes_above_range_shifted_down(self):
        """Median MIDI ~96 (C7) -> needs shift down into range."""
        segs = [
            MidiSegment(note="C7", start=0, end=1, word="high "),   # MIDI 96
            MidiSegment(note="D7", start=1, end=2, word="note "),   # MIDI 98
            MidiSegment(note="E7", start=2, end=3, word="test "),   # MIDI 100
        ]
        result = correct_global_octave(segs)
        midi_values = [librosa.note_to_midi(s.note) for s in result]
        for val in midi_values:
            self.assertLessEqual(val, 84)

    # -- shift is always a multiple of 12 ------------------------------------

    def test_shift_is_multiple_of_12(self):
        """The shift must be a whole number of octaves (multiple of 12)."""
        segs = [
            MidiSegment(note="C2", start=0, end=1, word="lo "),  # MIDI 36
            MidiSegment(note="G2", start=1, end=2, word="w "),   # MIDI 43
        ]
        original_midis = [librosa.note_to_midi(s.note) for s in segs]
        result = correct_global_octave(segs)
        shifted_midis = [librosa.note_to_midi(s.note) for s in result]

        for orig, shifted in zip(original_midis, shifted_midis, strict=True):
            self.assertEqual((shifted - orig) % 12, 0)

    # -- custom range ---------------------------------------------------------

    def test_custom_range(self):
        """Custom low/high should be respected."""
        segs = [
            MidiSegment(note="C4", start=0, end=1, word="mid "),  # MIDI 60
        ]
        # With a very narrow high range, C4 (60) is above high=55
        result = correct_global_octave(segs, low=40, high=55)
        midi_val = librosa.note_to_midi(result[0].note)
        self.assertLessEqual(midi_val, 55)

    # -- preserves relative intervals -----------------------------------------

    def test_relative_intervals_preserved(self):
        """After shift, the intervals between notes must be identical."""
        segs = [
            MidiSegment(note="C2", start=0, end=1, word="do "),   # MIDI 36
            MidiSegment(note="E2", start=1, end=2, word="mi "),   # MIDI 40
            MidiSegment(note="G2", start=2, end=3, word="sol "),  # MIDI 43
        ]
        original_midis = [librosa.note_to_midi(s.note) for s in segs]
        original_intervals = [
            original_midis[i + 1] - original_midis[i]
            for i in range(len(original_midis) - 1)
        ]

        result = correct_global_octave(segs)
        shifted_midis = [librosa.note_to_midi(s.note) for s in result]
        shifted_intervals = [
            shifted_midis[i + 1] - shifted_midis[i]
            for i in range(len(shifted_midis) - 1)
        ]

        self.assertEqual(original_intervals, shifted_intervals)


class TestConfidenceWeightedMedianNote(unittest.TestCase):
    """Tests for confidence_weighted_median_note()."""

    def test_single_frequency(self):
        note = confidence_weighted_median_note([440.0], [1.0])
        self.assertEqual(note, "A4")

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            confidence_weighted_median_note([], [])

    def test_mismatched_lengths_raises(self):
        with self.assertRaises(ValueError):
            confidence_weighted_median_note([440.0, 880.0], [1.0])

    def test_zero_weights_raises(self):
        with self.assertRaises(ValueError):
            confidence_weighted_median_note([440.0], [0.0])

    def test_high_confidence_dominates(self):
        """A single high-confidence frame should dominate low-confidence noise."""
        freqs = [440.0, 220.0, 220.0, 220.0]
        weights = [0.95, 0.1, 0.1, 0.1]
        note = confidence_weighted_median_note(freqs, weights)
        # The high-weight A4 (440 Hz) should win
        self.assertEqual(note, "A4")


if __name__ == "__main__":
    unittest.main()
