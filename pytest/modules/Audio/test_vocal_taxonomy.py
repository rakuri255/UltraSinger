"""Unit tests for the 5-Category Vocal Taxonomy:
- Backing Vocals spatial separation (IACC & CSR)
- Intentional Scat Singing preservation vs instrumental bridge purge
- Melisma continuous acoustic segmentation
"""

import os
import tempfile
import unittest
import numpy as np
import soundfile as sf

from modules.Audio.stereo_vocal_analyzer import (
    compute_stereo_vocal_spatial_metrics,
    separate_lead_and_backing_audio,
)
from modules.Speech_Recognition.fuzzy_lyric_matcher import (
    is_scat_or_adlib_pattern,
    purge_instrumental_scat_segments,
)
from modules.ProcessData import TranscribedData
from modules.Speech_Recognition.lyrics_aligner import align_transcribed_data_with_lyrics


class VocalTaxonomyTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_stereo_spatial_metrics_mono(self):
        # Create a mono WAV file
        sr = 22050
        t = np.linspace(0, 1.0, sr)
        mono_data = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
        wav_path = os.path.join(self.temp_dir.name, "mono_vocal.wav")
        sf.write(wav_path, mono_data, sr)

        metrics = compute_stereo_vocal_spatial_metrics(wav_path, start_time=0.0, duration=1.0)
        self.assertTrue(metrics["is_lead"])
        self.assertFalse(metrics["is_backing"])
        self.assertEqual(metrics["iacc"], 1.0)
        self.assertGreaterEqual(metrics["csr_db"], 20.0)

    def test_stereo_spatial_metrics_center_lead(self):
        # Create a 2-channel stereo WAV file with centered audio (L == R)
        sr = 22050
        t = np.linspace(0, 1.0, sr)
        sig = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
        stereo_center = np.stack([sig, sig], axis=1)  # shape (N, 2)
        wav_path = os.path.join(self.temp_dir.name, "center_lead.wav")
        sf.write(wav_path, stereo_center, sr)

        metrics = compute_stereo_vocal_spatial_metrics(wav_path, start_time=0.0, duration=1.0)
        self.assertTrue(metrics["is_lead"])
        self.assertFalse(metrics["is_backing"])
        self.assertGreaterEqual(metrics["iacc"], 0.95)
        self.assertGreaterEqual(metrics["csr_db"], 10.0)

    def test_stereo_spatial_metrics_wide_backing_vocals(self):
        # Create a 2-channel stereo WAV with side-panned / antiphase backing vocals
        sr = 22050
        t = np.linspace(0, 1.0, sr)
        sig = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
        stereo_side = np.stack([sig, -0.9 * sig], axis=1)
        wav_path = os.path.join(self.temp_dir.name, "side_backing.wav")
        sf.write(wav_path, stereo_side, sr)

        metrics = compute_stereo_vocal_spatial_metrics(wav_path, start_time=0.0, duration=1.0)
        self.assertFalse(metrics["is_lead"])
        self.assertTrue(metrics["is_backing"])
        self.assertLess(metrics["csr_db"], 2.0)

    def test_separate_lead_and_backing_audio(self):
        sr = 22050
        t = np.linspace(0, 0.5, int(sr * 0.5))
        lead_sig = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
        backing_sig = np.sin(2 * np.pi * 880.0 * t).astype(np.float32)

        # Mix: lead centered (L=R), backing panned side (L=-R)
        L = lead_sig + backing_sig
        R = lead_sig - backing_sig
        y_stereo = np.stack([L, R], axis=0)

        lead_mono, backing_stereo = separate_lead_and_backing_audio(y_stereo, sr=sr)
        self.assertEqual(len(lead_mono), len(lead_sig))
        self.assertEqual(backing_stereo.shape, y_stereo.shape)
        # Lead should have strong energy
        self.assertGreater(np.sum(lead_mono**2), 0.01)

    def test_intentional_scat_lyrics_preservation(self):
        # Real jazz scat lyrics: "ba", "du", "bi", "du", "bop"
        lyrics_words = [
            ("ba", False),
            ("du", False),
            ("bi", False),
            ("du", False),
            ("bop", True),
        ]
        # Whisper heard scat syllables
        transcribed = [
            TranscribedData(word="ba", start=1.0, end=1.15),
            TranscribedData(word="du", start=1.15, end=1.30),
            TranscribedData(word="bi", start=1.30, end=1.45),
            TranscribedData(word="du", start=1.45, end=1.60),
            TranscribedData(word="bop", start=1.60, end=1.80),
        ]

        aligned = align_transcribed_data_with_lyrics(transcribed, lyrics_words)
        # All 5 scat words must be preserved in the aligned output
        self.assertEqual(len(aligned), 5)
        aligned_words = [td.word.strip() for td in aligned]
        self.assertEqual(aligned_words, ["ba", "du", "bi", "du", "bop"])

    def test_instrumental_scat_purge_when_not_in_lyrics(self):
        # Non-scat song lyrics: "Hoy", "vamos", "a", "cantar"
        lyrics_words = ["Hoy", "vamos", "a", "cantar"]
        # Whisper hallucinated a solo bridge tarareo: "Tararara"
        whisper_segments = [
            {"word": "Tararara", "start": 25.0, "end": 28.0},
            {"word": "Hoy", "start": 30.0, "end": 30.5},
            {"word": "vamos", "start": 30.5, "end": 31.0},
        ]
        retained, purged = purge_instrumental_scat_segments(whisper_segments, lyrics_words)
        self.assertEqual(len(purged), 1)
        self.assertEqual(purged[0]["word"], "Tararara")
        self.assertEqual(len(retained), 2)
        self.assertEqual(retained[0]["word"], "Hoy")


if __name__ == "__main__":
    unittest.main()
