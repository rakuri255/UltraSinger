"""Tests for alignment_and_octave.py"""

import unittest
from modules.Midi.MidiSegment import MidiSegment
from modules.Ultrastar.ultrastar_writer import calculate_optimal_octave_offset
from modules.Ultrastar.coverter.ultrastar_converter import midi_note_to_ultrastar_note
from modules.Speech_Recognition.TranscribedData import TranscribedData
from UltraSinger import align_syllables_by_chars


class AlignmentAndOctaveTest(unittest.TestCase):
    def test_optimal_octave_offset_female_high(self):
        # Song with notes in E4 to G5 (MIDI 64 to 79, median 68)
        # Without offset: note = 68 - 48 = 20 (way above staff)
        # With offset: optimal_offset should be -1 or -2
        segments = [
            MidiSegment("E4", 0.0, 1.0, "word1"),
            MidiSegment("G4", 1.0, 2.0, "word2"),
            MidiSegment("C5", 2.0, 3.0, "word3"),
            MidiSegment("E5", 3.0, 4.0, "word4"),
        ]
        offset = calculate_optimal_octave_offset(segments)
        self.assertEqual(offset, -2)

        # Transposed note for E4 (MIDI 64) with offset -2: (64 - 48) - 24 = -8 (centered on staff!)
        transposed_pitch = midi_note_to_ultrastar_note(64, offset)
        self.assertEqual(transposed_pitch, -8)

    def test_optimal_octave_offset_already_centered(self):
        # Song already centered around C3/C4 (MIDI 48 to 52, median 50)
        # note = 50 - 48 = 2 (already centered)
        segments = [
            MidiSegment("C3", 0.0, 1.0, "word1"),
            MidiSegment("D3", 1.0, 2.0, "word2"),
            MidiSegment("E3", 2.0, 3.0, "word3"),
        ]
        offset = calculate_optimal_octave_offset(segments)
        self.assertEqual(offset, 0)

    def test_align_syllables_by_chars(self):
        # Word "cama", syllables ["ca", "ma"]
        chars = [
            {"char": "c", "start": 1.0, "end": 1.2},
            {"char": "a", "start": 1.2, "end": 1.8},  # "ca" lasts 0.8s
            {"char": "m", "start": 1.8, "end": 1.9},  # "ma" starts at 1.8s
            {"char": "a", "start": 1.9, "end": 2.0},  # "ma" ends at 2.0s
        ]
        data = TranscribedData(word="cama", start=1.0, end=2.0, chars=chars)

        result = align_syllables_by_chars(data, ["ca", "ma"])

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)
        # First syllable "ca" should start at 1.0s and end at 1.8s
        self.assertAlmostEqual(result[0][0], 1.0, places=2)
        self.assertAlmostEqual(result[0][1], 1.8, places=2)
        # Second syllable "ma" should start at 1.8s and end at 2.0s
        self.assertAlmostEqual(result[1][0], 1.8, places=2)
        self.assertAlmostEqual(result[1][1], 2.0, places=2)


if __name__ == "__main__":
    unittest.main()
