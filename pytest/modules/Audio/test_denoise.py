"""Tests for whisper.py"""

import os
import unittest
from src.modules.Audio.denoise import denoise_vocal_audio
import pytest


class DenoiseTest(unittest.TestCase):
    @pytest.mark.skip(reason="Skipping this FUNCTION level test, can be used for manual tests")
    def test_ffmpeg_reduce_noise(self):
        # Arrange
        test_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(test_dir + "/../../..")
        test_file_abs_path = os.path.abspath(root_dir + "/test_input/vocals.wav")
        test_file_name = os.path.basename(test_file_abs_path)
        test_output = test_dir + "/test_output"

        # Act
        denoise_vocal_audio(test_file_abs_path, test_output + "/output_" + test_file_name)


if __name__ == "__main__":
    unittest.main()
