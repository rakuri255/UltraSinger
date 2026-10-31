"""Tests for vocal_validator.py and sync_click_track.py"""

import os
import tempfile
import unittest
import numpy as np
import soundfile as sf

from modules.Audio.vocal_validator import (
    detect_first_vocal_onset,
    validate_and_calibrate_gap,
)
from modules.Audio.sync_click_track import (
    generate_click_sample,
    create_sync_click_track,
)
from modules.Speech_Recognition.TranscribedData import TranscribedData
from modules.Midi.MidiSegment import MidiSegment


class VocalValidatorTest(unittest.TestCase):
    def test_generate_click_sample(self):
        sr = 44100
        click = generate_click_sample(sr, duration_ms=16.0, freq_hz=1800.0, amplitude=0.8)
        self.assertIsInstance(click, np.ndarray)
        self.assertGreater(len(click), 0)
        # Peak amplitude should be close to 0.8
        self.assertAlmostEqual(float(np.max(np.abs(click))), 0.8, delta=0.05)
        # End should decay close to 0
        self.assertLess(abs(float(click[-1])), 0.1)

    def test_detect_first_vocal_onset_synthetic(self):
        sr = 22050
        duration = 3.0
        # 1.0s of silence, then 1.0s of 440Hz sine wave, then silence
        audio = np.zeros(int(sr * duration), dtype=np.float32)
        vocal_start_idx = int(sr * 1.0)
        vocal_end_idx = int(sr * 2.0)
        t = np.linspace(0, 1.0, vocal_end_idx - vocal_start_idx, endpoint=False)
        audio[vocal_start_idx:vocal_end_idx] = 0.5 * np.sin(2 * np.pi * 440 * t)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            sf.write(tmp_path, audio, sr)
            onset = detect_first_vocal_onset(tmp_path, search_window_sec=5.0)
            self.assertIsNotNone(onset)
            # Should detect onset near 1.0s within 50ms tolerance
            self.assertAlmostEqual(onset, 1.0, delta=0.08)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_detect_first_vocal_onset_silence_returns_none(self):
        sr = 22050
        audio = np.zeros(int(sr * 2.0), dtype=np.float32)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            sf.write(tmp_path, audio, sr)
            onset = detect_first_vocal_onset(tmp_path, search_window_sec=5.0)
            self.assertIsNone(onset)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_validate_and_calibrate_gap_within_tolerance(self):
        sr = 22050
        audio = np.zeros(int(sr * 3.0), dtype=np.float32)
        vocal_start_idx = int(sr * 1.20)
        vocal_end_idx = int(sr * 2.20)
        t = np.linspace(0, 1.0, vocal_end_idx - vocal_start_idx, endpoint=False)
        audio[vocal_start_idx:vocal_end_idx] = 0.5 * np.sin(2 * np.pi * 440 * t)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            sf.write(tmp_path, audio, sr)
            data = [
                TranscribedData(word="test", confidence=0.9, start=1.28, end=1.60),
                TranscribedData(word="song", confidence=0.9, start=1.80, end=2.20),
            ]

            onset, final_start, diff_ms = validate_and_calibrate_gap(
                tmp_path, data, auto_calibrate=True, max_calibrate_ms=250.0
            )

            self.assertIsNotNone(onset)
            # Syllable start should be calibrated to acoustic onset (~1.20s)
            self.assertAlmostEqual(data[0].start, onset, places=4)
            # Duration should be preserved (~0.32s)
            self.assertAlmostEqual(data[0].end - data[0].start, 0.32, places=2)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_create_sync_click_track_execution(self):
        sr = 44100
        audio = np.zeros(int(sr * 2.0), dtype=np.float32)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as in_tmp:
            in_path = in_tmp.name
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out_tmp:
            out_path = out_tmp.name

        try:
            sf.write(in_path, audio, sr)
            segments = [
                MidiSegment(note="C4", start=0.5, end=0.8, word="te"),
                MidiSegment(note="D4", start=1.2, end=1.5, word="st"),
            ]
            result = create_sync_click_track(in_path, out_path, segments)
            self.assertEqual(result, out_path)
            self.assertTrue(os.path.exists(out_path))

            # Verify that clicks were added into the output audio
            y_out, _ = sf.read(out_path)
            self.assertGreater(float(np.max(np.abs(y_out))), 0.5)
        finally:
            for p in (in_path, out_path):
                if os.path.exists(p):
                    os.remove(p)


if __name__ == "__main__":
    unittest.main()
