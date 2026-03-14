"""Tests for syllable-level note splitting (--syllable_split)."""

import copy
import unittest

from src.modules.Midi.MidiSegment import MidiSegment
from src.modules.Speech_Recognition.TranscribedData import TranscribedData
from src.UltraSinger import merge_syllable_segments, _absorb_tilde_segments


def _td(word, start, end, is_hyphen=False, is_word_end=True):
    """Helper to create TranscribedData."""
    td = TranscribedData()
    td.word = word
    td.start = start
    td.end = end
    td.is_hyphen = is_hyphen
    td.is_word_end = is_word_end
    return td


def _ms(note, start, end, word):
    """Helper to create MidiSegment."""
    return MidiSegment(note=note, start=start, end=end, word=word)


# BPM=120 → sixteenth note ≈ 0.125s
BPM = 120.0


class TestMergeSyllableSegmentsDefault(unittest.TestCase):
    """Verify default behaviour (preserve_syllables=False) is unchanged."""

    def test_same_pitch_syllables_merged(self):
        """Same-pitch syllables within a word should merge by default."""
        data = [
            _td("hel", 0.0, 0.25, is_hyphen=True, is_word_end=False),
            _td("lo ", 0.25, 0.5, is_hyphen=True, is_word_end=True),
        ]
        midi = [
            _ms("C4", 0.0, 0.25, "hel"),
            _ms("C4", 0.25, 0.5, "lo "),
        ]
        result_midi, result_data = merge_syllable_segments(midi, data, BPM, preserve_syllables=False)
        self.assertEqual(len(result_data), 1)
        self.assertIn("hel", result_data[0].word)
        self.assertIn("lo", result_data[0].word)

    def test_different_pitch_syllables_kept_separate(self):
        """Different-pitch syllables should stay separate even without flag."""
        data = [
            _td("hel", 0.0, 0.25, is_hyphen=True, is_word_end=False),
            _td("lo ", 0.25, 0.5, is_hyphen=True, is_word_end=True),
        ]
        midi = [
            _ms("C4", 0.0, 0.25, "hel"),
            _ms("D4", 0.25, 0.5, "lo "),
        ]
        result_midi, result_data = merge_syllable_segments(midi, data, BPM, preserve_syllables=False)
        self.assertEqual(len(result_data), 2)


class TestMergeSyllableSegmentsPreserve(unittest.TestCase):
    """Tests for preserve_syllables=True (--syllable_split mode)."""

    def test_same_pitch_syllables_preserved(self):
        """Same-pitch syllables should NOT merge when preserve_syllables=True."""
        data = [
            _td("hel", 0.0, 0.25, is_hyphen=True, is_word_end=False),
            _td("lo ", 0.25, 0.5, is_hyphen=True, is_word_end=True),
        ]
        midi = [
            _ms("C4", 0.0, 0.25, "hel"),
            _ms("C4", 0.25, 0.5, "lo "),
        ]
        result_midi, result_data = merge_syllable_segments(midi, data, BPM, preserve_syllables=True)
        self.assertEqual(len(result_data), 2)
        self.assertEqual(result_data[0].word, "hel")
        self.assertEqual(result_data[1].word, "lo ")

    def test_different_pitch_syllables_preserved(self):
        """Different-pitch syllables should stay separate."""
        data = [
            _td("hel", 0.0, 0.25, is_hyphen=True, is_word_end=False),
            _td("lo ", 0.25, 0.5, is_hyphen=True, is_word_end=True),
        ]
        midi = [
            _ms("C4", 0.0, 0.25, "hel"),
            _ms("D4", 0.25, 0.5, "lo "),
        ]
        result_midi, result_data = merge_syllable_segments(midi, data, BPM, preserve_syllables=True)
        self.assertEqual(len(result_data), 2)
        self.assertEqual(result_midi[0].note, "C4")
        self.assertEqual(result_midi[1].note, "D4")

    def test_tilde_same_pitch_still_merged(self):
        """~ segments with same pitch should still merge even in preserve mode."""
        data = [
            _td("hel", 0.0, 0.125, is_hyphen=True, is_word_end=False),
            _td("~", 0.125, 0.25, is_hyphen=True, is_word_end=False),
            _td("lo ", 0.25, 0.5, is_hyphen=True, is_word_end=True),
        ]
        midi = [
            _ms("C4", 0.0, 0.125, "hel"),
            _ms("C4", 0.125, 0.25, "~"),
            _ms("C4", 0.25, 0.5, "lo "),
        ]
        result_midi, result_data = merge_syllable_segments(midi, data, BPM, preserve_syllables=True)
        # ~ should be merged into "hel", "lo" stays separate
        self.assertEqual(len(result_data), 2)
        self.assertEqual(result_data[0].word, "hel")
        self.assertAlmostEqual(result_data[0].end, 0.25)
        self.assertEqual(result_data[1].word, "lo ")

    def test_tilde_different_pitch_absorbed(self):
        """~ with large pitch jump should be absorbed into adjacent segment."""
        data = [
            _td("hel", 0.0, 0.125, is_hyphen=True, is_word_end=False),
            _td("~", 0.125, 0.25, is_hyphen=True, is_word_end=False),
            _td("lo ", 0.25, 0.5, is_hyphen=True, is_word_end=True),
        ]
        midi = [
            _ms("C4", 0.0, 0.125, "hel"),
            _ms("E4", 0.125, 0.25, "~"),  # 4 semitone jump, not a slide
            _ms("E4", 0.25, 0.5, "lo "),
        ]
        result_midi, result_data = merge_syllable_segments(midi, data, BPM, preserve_syllables=True)
        # ~ should be absorbed, no tilde in output
        for d in result_data:
            self.assertNotEqual(d.word.strip(), "~")

    def test_word_boundary_respected(self):
        """Segments across word boundaries should never merge."""
        data = [
            _td("hel", 0.0, 0.25, is_hyphen=True, is_word_end=False),
            _td("lo ", 0.25, 0.5, is_hyphen=True, is_word_end=True),
            _td("world ", 0.5, 1.0, is_hyphen=False, is_word_end=True),
        ]
        midi = [
            _ms("C4", 0.0, 0.25, "hel"),
            _ms("C4", 0.25, 0.5, "lo "),
            _ms("C4", 0.5, 1.0, "world "),
        ]
        result_midi, result_data = merge_syllable_segments(midi, data, BPM, preserve_syllables=True)
        # "hel" and "lo" are within same word, "world" is separate
        # With preserve_syllables, all three stay separate
        self.assertEqual(len(result_data), 3)

    def test_non_hyphenated_word_unchanged(self):
        """Non-hyphenated words should pass through unchanged."""
        data = [
            _td("hello ", 0.0, 0.5, is_hyphen=False, is_word_end=True),
            _td("world ", 0.5, 1.0, is_hyphen=False, is_word_end=True),
        ]
        midi = [
            _ms("C4", 0.0, 0.5, "hello "),
            _ms("D4", 0.5, 1.0, "world "),
        ]
        result_midi, result_data = merge_syllable_segments(midi, data, BPM, preserve_syllables=True)
        self.assertEqual(len(result_data), 2)

    def test_timing_preserved_after_merge(self):
        """Timing should be preserved correctly after tilde absorption."""
        data = [
            _td("hel", 0.0, 0.125, is_hyphen=True, is_word_end=False),
            _td("~ ", 0.125, 0.25, is_hyphen=True, is_word_end=True),
        ]
        midi = [
            _ms("C4", 0.0, 0.125, "hel"),
            _ms("C4", 0.125, 0.25, "~ "),
        ]
        result_midi, result_data = merge_syllable_segments(midi, data, BPM, preserve_syllables=True)
        # ~ merged into hel
        self.assertEqual(len(result_data), 1)
        self.assertAlmostEqual(result_data[0].start, 0.0)
        self.assertAlmostEqual(result_data[0].end, 0.25)
        self.assertTrue(result_data[0].is_word_end)

    def test_three_syllable_word_all_same_pitch(self):
        """Three syllables with same pitch should all stay separate."""
        data = [
            _td("beau", 0.0, 0.2, is_hyphen=True, is_word_end=False),
            _td("ti", 0.2, 0.4, is_hyphen=True, is_word_end=False),
            _td("ful ", 0.4, 0.6, is_hyphen=True, is_word_end=True),
        ]
        midi = [
            _ms("C4", 0.0, 0.2, "beau"),
            _ms("C4", 0.2, 0.4, "ti"),
            _ms("C4", 0.4, 0.6, "ful "),
        ]
        result_midi, result_data = merge_syllable_segments(midi, data, BPM, preserve_syllables=True)
        self.assertEqual(len(result_data), 3)
        self.assertEqual(result_data[0].word, "beau")
        self.assertEqual(result_data[1].word, "ti")
        self.assertEqual(result_data[2].word, "ful ")


class TestAbsorbTildeSegments(unittest.TestCase):
    """Tests for the _absorb_tilde_segments helper."""

    def test_empty_input(self):
        midi, data = _absorb_tilde_segments([], [])
        self.assertEqual(midi, [])
        self.assertEqual(data, [])

    def test_no_tildes(self):
        data = [_td("hello ", 0.0, 0.5)]
        midi = [_ms("C4", 0.0, 0.5, "hello ")]
        result_midi, result_data = _absorb_tilde_segments(midi, data)
        self.assertEqual(len(result_data), 1)
        self.assertEqual(result_data[0].word, "hello ")

    def test_tilde_absorbed_into_previous(self):
        data = [
            _td("hel", 0.0, 0.25, is_hyphen=True, is_word_end=False),
            _td("~", 0.25, 0.5, is_hyphen=True, is_word_end=False),
        ]
        midi = [
            _ms("C4", 0.0, 0.25, "hel"),
            _ms("E4", 0.25, 0.5, "~"),
        ]
        result_midi, result_data = _absorb_tilde_segments(midi, data)
        self.assertEqual(len(result_data), 1)
        self.assertEqual(result_data[0].word, "hel")
        self.assertAlmostEqual(result_data[0].end, 0.5)

    def test_tilde_with_trailing_space_absorbed(self):
        data = [
            _td("hel", 0.0, 0.25, is_hyphen=True, is_word_end=False),
            _td("~ ", 0.25, 0.5, is_hyphen=True, is_word_end=True),
        ]
        midi = [
            _ms("C4", 0.0, 0.25, "hel"),
            _ms("E4", 0.25, 0.5, "~ "),
        ]
        result_midi, result_data = _absorb_tilde_segments(midi, data)
        self.assertEqual(len(result_data), 1)
        self.assertTrue(result_data[0].word.endswith(" "))
        self.assertTrue(result_data[0].is_word_end)

    def test_multiple_tildes_absorbed(self):
        data = [
            _td("hel", 0.0, 0.1, is_hyphen=True, is_word_end=False),
            _td("~", 0.1, 0.2, is_hyphen=True, is_word_end=False),
            _td("~", 0.2, 0.3, is_hyphen=True, is_word_end=False),
            _td("lo ", 0.3, 0.5, is_hyphen=True, is_word_end=True),
        ]
        midi = [
            _ms("C4", 0.0, 0.1, "hel"),
            _ms("D4", 0.1, 0.2, "~"),
            _ms("E4", 0.2, 0.3, "~"),
            _ms("E4", 0.3, 0.5, "lo "),
        ]
        result_midi, result_data = _absorb_tilde_segments(midi, data)
        # Both ~ absorbed into "hel"
        self.assertEqual(len(result_data), 2)
        self.assertEqual(result_data[0].word, "hel")
        self.assertAlmostEqual(result_data[0].end, 0.3)
        self.assertEqual(result_data[1].word, "lo ")


if __name__ == "__main__":
    unittest.main()
