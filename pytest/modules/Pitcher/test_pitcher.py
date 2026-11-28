"""Tests for pitcher.py"""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import src.modules.Pitcher.pitcher as test_subject
import pytest
from src.modules.plot import plot


class PitcherTest(unittest.TestCase):
    @pytest.mark.skip(reason="Skipping this FUNCTION level test, can be used for manual tests")
    def test_get_pitch_with_crepe_file(self):
        # Arrange
        test_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(test_dir + "/../../..")
        # test_file_abs_path = os.path.abspath(root_dir + "/test_input/audio_denoised.wav")
        test_file_abs_path = os.path.abspath(root_dir + "/test_input/test_denoised.wav")
        test_output = root_dir + "/test_output"

        # Act
        pitched_data = test_subject.get_pitch_with_file(test_file_abs_path)
        # test_subject.get_pitch_with_crepe_file(test_file_abs_path, 'full', 'cpu', batch_size=1024)
        plot(pitched_data, test_output, title="pitching test")
        print("done")


class SwiftF0RangeTest(unittest.TestCase):
    def test_frames_outside_vocal_range_get_zero_confidence(self):
        result = SimpleNamespace(
            timestamps=np.array([0.0, 0.016, 0.032, 0.048]),
            pitch_hz=np.array([440.0, 1950.0, 50.0, 1046.5]),
            confidence=np.array([0.9, 0.8, 0.7, 0.6]),
        )
        detector = mock.Mock()
        detector.detect_from_array.return_value = result

        with mock.patch.object(test_subject, "_get_detector", return_value=detector):
            pitched_data = test_subject.get_pitch_with_swift_f0(np.zeros(16000, dtype=np.float32), 16000)

        self.assertEqual([440.0, 1950.0, 50.0, 1046.5], pitched_data.frequencies)
        self.assertEqual([0.9, 0.0, 0.0, 0.6], [round(c, 2) for c in pitched_data.confidence])


if __name__ == "__main__":
    unittest.main()
