"""Tests for midi_creator.py"""

import unittest

from modules.Midi.midi_creator import (
    get_most_confident_note,
    create_midi_notes_from_pitched_data,
)
from modules.Pitcher.pitched_data import PitchedData

A4 = 440.0
E4 = 329.63


class GetMostConfidentNoteTest(unittest.TestCase):
    def test_prefers_frames_above_threshold(self):
        # Many weak A4 frames, few confident E4 frames -> E4
        freqs = [A4] * 6 + [E4] * 2
        confs = [0.3] * 6 + [0.9] * 2
        self.assertEqual("E4", get_most_confident_note(freqs, confs))

    def test_weighted_vote_when_no_frame_above_threshold(self):
        # Fewer frames but higher confidence wins the weighted vote
        freqs = [A4] * 3 + [E4] * 2
        confs = [0.05] * 3 + [0.35] * 2
        self.assertEqual("E4", get_most_confident_note(freqs, confs))

    def test_returns_none_without_usable_frames(self):
        self.assertIsNone(get_most_confident_note([1900.0, 1950.0], [0.0, 0.0]))


class CreateMidiNotesTest(unittest.TestCase):
    def test_segment_without_usable_frames_keeps_previous_note(self):
        times = [i * 0.1 for i in range(20)]
        # first segment: confident E4, second segment: only zero confidence harmonics
        frequencies = [E4] * 10 + [1950.0] * 10
        confidence = [0.9] * 10 + [0.0] * 10
        pitched_data = PitchedData(times, frequencies, confidence)

        segments = create_midi_notes_from_pitched_data(
            [0.0, 1.0], [0.9, 1.9], ["word ", "~ "], pitched_data
        )

        self.assertEqual("E4", segments[0].note)
        self.assertEqual("E4", segments[1].note)


if __name__ == "__main__":
    unittest.main()
