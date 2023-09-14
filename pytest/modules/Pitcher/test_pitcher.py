"""Tests for pitcher.py"""

import os
import unittest
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
        pitched_data = test_subject.get_pitch_with_crepe_file(test_file_abs_path, 'full', device="cuda")
        # test_subject.get_pitch_with_crepe_file(test_file_abs_path, 'full', 'cpu', batch_size=1024)
        plot(pitched_data, test_output, title="pitching test")
        print("done")

if __name__ == "__main__":
    unittest.main()
