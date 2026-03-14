"""Tests for vocal gap fill module."""

import unittest

from src.modules.Audio.vocal_gap_fill import (
    fill_vocal_gaps,
    _find_gaps,
    _has_vocal_content,
    _find_first_ge,
)
from src.modules.Pitcher.pitched_data import PitchedData
from src.modules.Speech_Recognition.TranscribedData import TranscribedData


def _td(word, start, end):
    """Helper to create TranscribedData."""
    td = TranscribedData()
    td.word = word
    td.start = start
    td.end = end
    td.is_word_end = True
    return td


def _pd(times, frequencies, confidence):
    """Helper to create PitchedData."""
    return PitchedData(times=times, frequencies=frequencies, confidence=confidence)


class TestFindFirstGe(unittest.TestCase):
    def test_exact_match(self):
        self.assertEqual(_find_first_ge([1.0, 2.0, 3.0], 2.0), 1)

    def test_between_values(self):
        self.assertEqual(_find_first_ge([1.0, 2.0, 3.0], 1.5), 1)

    def test_before_all(self):
        self.assertEqual(_find_first_ge([1.0, 2.0, 3.0], 0.5), 0)

    def test_after_all(self):
        self.assertEqual(_find_first_ge([1.0, 2.0, 3.0], 4.0), 3)

    def test_empty(self):
        self.assertEqual(_find_first_ge([], 1.0), 0)


class TestFindGaps(unittest.TestCase):
    def test_no_gaps(self):
        data = [_td("a", 0.0, 0.5), _td("b", 0.5, 1.0)]
        gaps = _find_gaps(data, min_gap_s=0.15)
        self.assertEqual(gaps, [])

    def test_small_gap_ignored(self):
        data = [_td("a", 0.0, 0.5), _td("b", 0.6, 1.0)]
        gaps = _find_gaps(data, min_gap_s=0.15)
        self.assertEqual(gaps, [])

    def test_gap_detected(self):
        data = [_td("a", 0.0, 0.5), _td("b", 1.0, 1.5)]
        gaps = _find_gaps(data, min_gap_s=0.15)
        self.assertEqual(len(gaps), 1)
        self.assertAlmostEqual(gaps[0][0], 0.5)
        self.assertAlmostEqual(gaps[0][1], 1.0)

    def test_multiple_gaps(self):
        data = [
            _td("a", 0.0, 0.5),
            _td("b", 1.0, 1.5),
            _td("c", 2.0, 2.5),
        ]
        gaps = _find_gaps(data, min_gap_s=0.15)
        self.assertEqual(len(gaps), 2)

    def test_single_word_no_gaps(self):
        data = [_td("a", 0.0, 0.5)]
        gaps = _find_gaps(data, min_gap_s=0.15)
        self.assertEqual(gaps, [])


class TestHasVocalContent(unittest.TestCase):
    def test_all_confident(self):
        pd = _pd(
            times=[0.5, 0.6, 0.7, 0.8, 0.9],
            frequencies=[440.0] * 5,
            confidence=[0.9] * 5,
        )
        self.assertTrue(_has_vocal_content(pd, 0.5, 1.0, 0.5, 0.3))

    def test_no_confidence(self):
        pd = _pd(
            times=[0.5, 0.6, 0.7, 0.8, 0.9],
            frequencies=[440.0] * 5,
            confidence=[0.1] * 5,
        )
        self.assertFalse(_has_vocal_content(pd, 0.5, 1.0, 0.5, 0.3))

    def test_partial_confidence_above_threshold(self):
        pd = _pd(
            times=[0.5, 0.6, 0.7, 0.8, 0.9],
            frequencies=[440.0] * 5,
            confidence=[0.9, 0.9, 0.1, 0.1, 0.1],
        )
        # 2/5 = 0.4 > 0.3 threshold
        self.assertTrue(_has_vocal_content(pd, 0.5, 1.0, 0.5, 0.3))

    def test_partial_confidence_below_threshold(self):
        pd = _pd(
            times=[0.5, 0.6, 0.7, 0.8, 0.9],
            frequencies=[440.0] * 5,
            confidence=[0.9, 0.1, 0.1, 0.1, 0.1],
        )
        # 1/5 = 0.2 < 0.3 threshold
        self.assertFalse(_has_vocal_content(pd, 0.5, 1.0, 0.5, 0.3))

    def test_empty_range(self):
        pd = _pd(times=[0.0, 0.1], frequencies=[440.0, 440.0], confidence=[0.9, 0.9])
        self.assertFalse(_has_vocal_content(pd, 5.0, 6.0, 0.5, 0.3))


class TestFillVocalGaps(unittest.TestCase):
    def test_empty_input(self):
        result = fill_vocal_gaps([], _pd([], [], []))
        self.assertEqual(result, [])

    def test_no_gaps(self):
        data = [_td("a ", 0.0, 0.5), _td("b ", 0.5, 1.0)]
        pd = _pd([0.0, 0.1, 0.2], [440.0] * 3, [0.9] * 3)
        result = fill_vocal_gaps(data, pd)
        self.assertEqual(len(result), 2)

    def test_gap_filled_with_vocal(self):
        data = [_td("a ", 0.0, 0.5), _td("b ", 1.0, 1.5)]
        # Pitch frames covering the gap 0.5-1.0 with high confidence
        pd = _pd(
            times=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            frequencies=[440.0] * 11,
            confidence=[0.9] * 11,
        )
        result = fill_vocal_gaps(data, pd)
        self.assertEqual(len(result), 3)
        # Check the gap-fill entry
        gap_entry = result[1]
        self.assertEqual(gap_entry.word.strip(), "~")
        self.assertAlmostEqual(gap_entry.start, 0.5)
        self.assertAlmostEqual(gap_entry.end, 1.0)
        self.assertTrue(gap_entry.is_word_end)

    def test_gap_not_filled_without_vocal(self):
        data = [_td("a ", 0.0, 0.5), _td("b ", 1.0, 1.5)]
        # Pitch frames covering the gap but with low confidence (silence)
        pd = _pd(
            times=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            frequencies=[440.0] * 11,
            confidence=[0.9, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9],
        )
        result = fill_vocal_gaps(data, pd)
        self.assertEqual(len(result), 2)

    def test_result_sorted_by_start(self):
        data = [
            _td("a ", 0.0, 0.3),
            _td("b ", 0.8, 1.0),
            _td("c ", 1.5, 2.0),
        ]
        pd = _pd(
            times=[i * 0.1 for i in range(21)],
            frequencies=[440.0] * 21,
            confidence=[0.9] * 21,
        )
        result = fill_vocal_gaps(data, pd)
        # Should have gap-fills inserted and be sorted
        for i in range(len(result) - 1):
            self.assertLessEqual(result[i].start, result[i + 1].start)

    def test_gap_fill_entry_properties(self):
        data = [_td("a ", 0.0, 0.5), _td("b ", 1.0, 1.5)]
        pd = _pd(
            times=[i * 0.1 for i in range(16)],
            frequencies=[440.0] * 16,
            confidence=[0.9] * 16,
        )
        result = fill_vocal_gaps(data, pd)
        gap = [d for d in result if d.word.strip() == "~"][0]
        self.assertTrue(gap.is_word_end)
        self.assertFalse(gap.is_hyphen)
        self.assertEqual(gap.confidence, 0.0)

    def test_short_gap_ignored(self):
        data = [_td("a ", 0.0, 0.5), _td("b ", 0.6, 1.0)]
        pd = _pd(
            times=[i * 0.1 for i in range(11)],
            frequencies=[440.0] * 11,
            confidence=[0.9] * 11,
        )
        # Gap is only 0.1s, below default 0.15s threshold
        result = fill_vocal_gaps(data, pd)
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
