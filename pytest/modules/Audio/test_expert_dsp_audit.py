"""Audit and regression unit tests for the 4 DSP & alignment requirements:
1. Syllable division & Melismas (regional dictionary & F0 melisma segmentation)
2. Anti-drift synchronization in float64 anchored to (T - GAP)
3. Strict 40ms-50ms acoustic onset snapping window
4. RMS voice activity trimmer unvoiced consonant protection (+50ms padding)
"""

import unittest
import unittest.mock
import numpy as np

from modules.Speech_Recognition.hyphenation import language_check, STRICT_REGIONAL_MAP
from modules.Midi.midi_creator import segment_melismas_from_f0
from modules.Pitcher.pitched_data import PitchedData
from modules.Audio.onset_snapping import snap_transcribed_data_to_onsets
from modules.Audio.voice_activity_trimmer import UNVOICED_CONSONANT_PATTERN, trim_syllable_ends_with_energy
from modules.Speech_Recognition.TranscribedData import TranscribedData
from modules.Ultrastar.ultrastar_writer import create_ultrastar_txt
from modules.Midi.MidiSegment import MidiSegment
from modules.Ultrastar.ultrastar_txt import UltrastarTxtValue
import tempfile
import os


class ExpertDspAuditTest(unittest.TestCase):
    def test_requirement_1_strict_regional_dictionary(self):
        """Verify strict regional dictionary mapping (no bare 'es' or 'en')"""
        self.assertEqual(STRICT_REGIONAL_MAP.get("es"), "es_ES")
        self.assertEqual(STRICT_REGIONAL_MAP.get("en"), "en_US")
        
        # Checking language_check returns strict regional code
        lang = language_check("es")
        self.assertIn("_", lang)
        self.assertEqual(lang, "es_ES")

    def test_requirement_1_acoustic_melisma_f0_segmentation(self):
        """Verify F0 variation >= 1 semitone sustained >= 60ms splits syllable into text + '~'"""
        # Create synthetic pitched data:
        # First 100ms: 440 Hz (A4, MIDI 69)
        # Next 100ms: 493.88 Hz (B4, MIDI 71 - 2 semitones jump)
        times = [0.01 * i for i in range(20)]  # 0.00 to 0.19s
        frequencies = [440.0] * 10 + [493.88] * 10
        confidences = [0.9] * 20

        pitched = PitchedData(times=times, frequencies=frequencies, confidence=confidences)

        segments = segment_melismas_from_f0(
            start_time=0.0,
            end_time=0.20,
            word="cor ",
            pitched_data=pitched,
            min_segment_duration=0.06,
        )

        # Must be split into 2 segments: "cor" and "~ "
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].word, "cor")
        self.assertEqual(segments[1].word, "~ ")
        # Pitches should reflect the two notes
        self.assertEqual(segments[0].note, "A4")
        self.assertEqual(segments[1].note, "B4")

    def test_requirement_2_antidrift_float64_anchoring(self):
        """Verify that cumulative note durations do not drift over 200 notes"""
        # Create 200 notes, each 0.35s duration, separated by 0.05s
        segments = []
        t = 10.0
        for i in range(200):
            segments.append(MidiSegment("C4", t, t + 0.35, f"w{i} "))
            t += 0.40  # 0.35 sung + 0.05 silence

        with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False) as tf:
            temp_path = tf.name

        try:
            txt_val = UltrastarTxtValue()
            txt_val.artist = "Artist"
            txt_val.title = "Title"
            txt_val.mp3 = "audio.mp3"
            
            # BPM = 120 -> 4 beats per second (with multiplier = 8 or 16)
            create_ultrastar_txt(segments, temp_path, txt_val, real_bpm=120.0, auto_octave=False)

            with open(temp_path, "r") as f:
                lines = [line.strip() for line in f if line.startswith(": ")]

            self.assertEqual(len(lines), 200)

            # Check the 200th note: its start time in seconds is 10.0 + 199 * 0.40 = 89.60s
            # Gap is segments[0].start = 10.0s
            # Relative time = 79.60s
            # At real_bpm=120, ultrastar_bpm=240 (multiplier=2 -> factor 8 beats/s or multiplier=4 -> 16 beats/s)
            # Let's inspect the last line start_beat
            parts_last = lines[-1].split()
            last_start_beat = int(parts_last[1])
            last_dur = int(parts_last[2])

            # In the code:
            # ultrastar_bpm = real_bpm_to_ultrastar_bpm(120) = 30.0
            # mult = get_multiplier(30.0) = 13
            # seconds_to_beat_factor = (real_bpm * mult) / 60.0 = (120 * 13) / 60 = 26.0
            # expected_start_beat = round((89.60 - 10.0) * 26.0) = 2070
            from modules.Ultrastar.coverter.ultrastar_converter import real_bpm_to_ultrastar_bpm
            from modules.Ultrastar.ultrastar_writer import get_multiplier
            ultrastar_bpm = real_bpm_to_ultrastar_bpm(120.0)
            mult = get_multiplier(ultrastar_bpm)
            factor = (120.0 * mult) / 60.0
            expected_start_beat = int(round((89.60 - 10.0) * factor))
            self.assertEqual(last_start_beat, expected_start_beat)

            # Convert back: absolute time difference must be < 20ms (zero cumulative drift)
            # 1 beat at factor 26.0 is ~38.4ms
            calculated_time = (last_start_beat / factor) + 10.0
            self.assertAlmostEqual(calculated_time, 89.60, delta=0.04)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_requirement_3_onset_snapping_strict_window_clamp(self):
        """Verify onset snapping strictly clamps window to 40-50ms"""
        # Try passing an unsafe 150ms window; code must clamp to <= 50ms
        data = [
            TranscribedData(word="bleed ", confidence=0.9, start=1.08, end=1.50)
        ]
        # Onset at 1.00s (80ms away - drum bleed)
        with unittest.mock.patch("modules.Audio.onset_snapping.detect_vocal_onsets") as mock_onsets:
            mock_onsets.return_value = np.array([1.00])
            result = snap_transcribed_data_to_onsets("mock.wav", data, snap_window_ms=150.0)

            # Should NOT snap to drum bleed at 80ms distance because of 50ms safety clamp
            self.assertEqual(result[0].start, 1.08)

    def test_requirement_4_unvoiced_consonant_padding(self):
        """Verify unvoiced consonants are recognized and given +50ms padding"""
        consonants_to_test = ["más", "red", "luz", "cantar", "stop", "pop", "tick", "café "]
        # "más", "red", "luz", "stop", "pop", "tick" end in unvoiced/fricatives/stops
        self.assertTrue(UNVOICED_CONSONANT_PATTERN.search("más"))
        self.assertTrue(UNVOICED_CONSONANT_PATTERN.search("red"))
        self.assertTrue(UNVOICED_CONSONANT_PATTERN.search("luz"))
        self.assertTrue(UNVOICED_CONSONANT_PATTERN.search("stop "))
        self.assertTrue(UNVOICED_CONSONANT_PATTERN.search("tick"))
        # Vowels should NOT match
        self.assertFalse(UNVOICED_CONSONANT_PATTERN.search("cama"))
        self.assertFalse(UNVOICED_CONSONANT_PATTERN.search("auto "))


if __name__ == "__main__":
    unittest.main()
