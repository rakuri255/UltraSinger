"""Tests for onset_snapping.py"""

import unittest
from unittest.mock import patch
import numpy as np

from modules.Audio.onset_snapping import (
    snap_transcribed_data_to_onsets,
    detect_vocal_onsets,
)
from modules.Speech_Recognition.TranscribedData import TranscribedData


class OnsetSnappingTest(unittest.TestCase):
    @patch("modules.Audio.onset_snapping.detect_vocal_onsets")
    def test_snap_within_window(self, mock_detect_onsets):
        # Onset at 1.00s, syllable at 1.035s (35ms distance, within 45ms strict window)
        mock_detect_onsets.return_value = np.array([1.00, 2.50])

        data = [
            TranscribedData(word="test ", confidence=0.9, start=1.035, end=1.535),
        ]

        result = snap_transcribed_data_to_onsets("mock_vocals.wav", data, snap_window_ms=45.0)

        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0].start, 1.00, places=4)
        # Duration (0.50s) should be preserved
        self.assertAlmostEqual(result[0].end, 1.50, places=4)

    @patch("modules.Audio.onset_snapping.detect_vocal_onsets")
    def test_snap_outside_window(self, mock_detect_onsets):
        # Onset at 1.00s, syllable at 1.10s (100ms distance, beyond 45ms/50ms window)
        mock_detect_onsets.return_value = np.array([1.00])

        data = [
            TranscribedData(word="test ", confidence=0.9, start=1.10, end=1.60),
        ]

        result = snap_transcribed_data_to_onsets("mock_vocals.wav", data, snap_window_ms=45.0)

        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0].start, 1.10, places=4)
        self.assertAlmostEqual(result[0].end, 1.60, places=4)

    @patch("modules.Audio.onset_snapping.detect_vocal_onsets")
    def test_overlap_prevention(self, mock_detect_onsets):
        # If syllable 1 ends at 1.60 and syllable 2 is snapped backwards to 1.50,
        # they must not overlap: midpoint resolution
        mock_detect_onsets.return_value = np.array([1.00, 1.50])

        data = [
            TranscribedData(word="syl1 ", confidence=0.9, start=1.02, end=1.60),
            TranscribedData(word="syl2 ", confidence=0.9, start=1.53, end=2.00),
        ]

        result = snap_transcribed_data_to_onsets("mock_vocals.wav", data, snap_window_ms=45.0)

        self.assertEqual(len(result), 2)
        # Verify no overlap
        self.assertAlmostEqual(result[0].end, result[1].start, places=4)
        self.assertLessEqual(result[0].end, result[1].start)

    def test_empty_transcribed_data(self):
        result = snap_transcribed_data_to_onsets("mock_vocals.wav", [])
        self.assertEqual(result, [])

    def test_nonexistent_audio_file(self):
        # detect_vocal_onsets on missing file returns empty array without raising
        onsets = detect_vocal_onsets("/nonexistent/path/vocals.wav")
        self.assertEqual(len(onsets), 0)


if __name__ == "__main__":
    unittest.main()
