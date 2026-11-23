"""Tests for key_detector.py"""

import os
import shutil
import tempfile
import unittest

import librosa
import numpy as np
from scipy.io import wavfile

from modules.Audio.key_detector import detect_key_from_audio, get_allowed_notes_for_key

SAMPLE_RATE = 22050


def _tone(note: str, seconds: float) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return 0.3 * np.sin(2 * np.pi * librosa.note_to_hz(note) * t)


def _melody(notes: list[tuple[str, float]], repeats: int) -> np.ndarray:
    return np.concatenate([_tone(note, seconds) for note, seconds in notes] * repeats)


class KeyDetectorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write(self, audio: np.ndarray) -> str:
        path = os.path.join(self.tmp, "vocals.wav")
        wavfile.write(path, SAMPLE_RATE, audio.astype(np.float32))
        return path

    def test_detects_minor_key_after_muted_intro(self):
        # A natural minor melody, tonic and fifth emphasized, after a 20 second muted intro
        a_minor = [("A3", 1.0), ("C4", 0.5), ("E4", 0.75), ("D4", 0.25), ("B3", 0.25),
                   ("A3", 0.75), ("G3", 0.25), ("F3", 0.25), ("E3", 0.5), ("A3", 1.0)]
        audio = np.concatenate([np.zeros(20 * SAMPLE_RATE), _melody(a_minor, 6)])

        self.assertEqual(("A", "minor"), detect_key_from_audio(self._write(audio)))

    def test_detects_major_key(self):
        g_major = [("G3", 1.0), ("B3", 0.5), ("D4", 0.75), ("C4", 0.25), ("A3", 0.25),
                   ("G3", 0.75), ("F#3", 0.25), ("E3", 0.25), ("D3", 0.5), ("G3", 1.0)]

        self.assertEqual(("G", "major"), detect_key_from_audio(self._write(_melody(g_major, 6))))

    def test_allowed_notes_for_key(self):
        self.assertEqual({"A", "B", "C", "D", "E", "F", "G"}, get_allowed_notes_for_key("A", "minor"))


if __name__ == "__main__":
    unittest.main()
