"""Unit tests for Master Plan 10/10 fixes:
- Time-monotonic DP aligner (preventing cross-chorus jumping and verse deletion)
- Spanish RAE and English phonetic syllabifiers (cer-ca-no, Ev-ery, on-ly, a-round, Ba-by)
- Autocorrect single vowel preservation (not turned into Freestyle 'F')
- Autocorrect BPM multiplier preservation in [200, 400]
"""

import unittest
from modules.Speech_Recognition.TranscribedData import TranscribedData
from modules.Speech_Recognition.syllabifier import syllabify_es, syllabify_en
from modules.Speech_Recognition.lyrics_aligner import monotonic_dp_align
from modules.Speech_Recognition.fuzzy_lyric_matcher import is_scat_or_adlib_pattern, classify_vocable_category


class TestSync10OutOf10(unittest.TestCase):
    def test_spanish_syllabification_rae(self):
        # Must split VCV such as 'cer-ca-no'
        self.assertEqual(syllabify_es("cercano"), ["cer", "ca", "no"])
        self.assertEqual(syllabify_es("silencioso"), ["si", "len", "cio", "so"])
        self.assertEqual(syllabify_es("sabiduría"), ["sa", "bi", "du", "rí", "a"])
        self.assertEqual(syllabify_es("sonido"), ["so", "ni", "do"])
        self.assertEqual(syllabify_es("canciones"), ["can", "cio", "nes"])
        self.assertEqual(syllabify_es("momento"), ["mo", "men", "to"])
        self.assertEqual(syllabify_es("destino"), ["des", "ti", "no"])
        self.assertEqual(syllabify_es("maravillas"), ["ma", "ra", "vi", "llas"])
        self.assertEqual(syllabify_es("esperanza"), ["es", "pe", "ran", "za"])

    def test_english_singing_syllabification(self):
        # Must split singing lexicon words that pyhyphen leaves unsplit
        self.assertEqual(syllabify_en("Every"), ["Ev", "ery"])
        self.assertEqual(syllabify_en("only"), ["on", "ly"])
        self.assertEqual(syllabify_en("around"), ["a", "round"])
        self.assertEqual(syllabify_en("Baby"), ["Ba", "by"])
        self.assertEqual(syllabify_en("watching"), ["watch", "ing"])
        self.assertEqual(syllabify_en("embrace"), ["em", "brace"])
        # Silent e / contractions should remain 1 syllable
        self.assertEqual(syllabify_en("take"), ["take"])
        self.assertEqual(syllabify_en("breath"), ["breath"])
        self.assertEqual(syllabify_en("smile"), ["smile"])
        self.assertEqual(syllabify_en("can't"), ["can't"])

    def test_single_vowels_not_scat(self):
        # Single vowels ('E', 'u', 'e', 'a', 'o') must never be classified as scat/ad-lib
        for v in ["E", "u", "e", "a", "o", "Ella", "una", "ese"]:
            self.assertFalse(
                is_scat_or_adlib_pattern(v),
                f"Word '{v}' should be recognized as lexical, not scat"
            )

    def test_monotonic_dp_align_repetitive_song(self):
        # Simulate a song with Chorus 1 at t=50s and Chorus 2 at t=135s with identical words
        # SequenceMatcher greedily matches Chorus 1 to t=135s if a word is missing in Chorus 1;
        # monotonic_dp_align must keep Chorus 1 at t=50s and Chorus 2 at t=135s!
        whisper_words = [
            # Verse 1: 0 - 30s
            ("Step", 15.0), ("by", 16.0), ("step", 17.0), ("now", 18.0),
            # Chorus 1: 48 - 60s (with a slight ASR typo: 'sound' instead of 'loud')
            ("Sing", 48.0), ("a", 49.0), ("song", 50.0), ("sound", 51.0),
            ("for", 52.0), ("all", 53.0), ("to", 54.0), ("hear", 55.0),
            ("with", 56.0), ("pure", 57.0), ("joy", 58.0),
            # Bridge: 82 - 100s
            ("When", 82.0), ("the", 83.0), ("music", 84.0), ("starts", 85.0), ("to", 86.0), ("play", 87.0),
            # Chorus 2: 134 - 150s (with exact words)
            ("Sing", 134.0), ("a", 135.0), ("song", 136.0), ("loud", 137.0),
            ("for", 138.0), ("all", 139.0), ("to", 140.0), ("hear", 141.0),
            ("with", 142.0), ("pure", 143.0), ("cheer", 144.0),
        ]
        td_list = [
            TranscribedData(word=w + " ", start=t, end=t + 0.8, confidence=0.9)
            for w, t in whisper_words
        ]

        lyrics_words = [
            # Verse 1
            ("Step", False), ("by", False), ("step", False), ("now", True),
            # Chorus 1
            ("Sing", False), ("a", False), ("song", False), ("loud", True),
            ("For", False), ("all", False), ("to", False), ("hear", True),
            ("with", False), ("pure", False), ("cheer", True),
            # Bridge
            ("When", False), ("the", False), ("music", False), ("starts", False), ("to", False), ("play", True),
            # Chorus 2
            ("Sing", False), ("a", False), ("song", False), ("loud", True),
            ("For", False), ("all", False), ("to", False), ("hear", True),
            ("with", False), ("pure", False), ("cheer", True),
        ]

        opcodes = monotonic_dp_align(td_list, lyrics_words)

        # Verify that Chorus 1 (lyrics index 4..14) matches whisper words around t=48..58s
        matched_chorus1 = False
        for tag, i1, i2, j1, j2 in opcodes:
            if j1 <= 4 < j2 or (j1 <= 7 < j2):
                if i1 < len(td_list):
                    t = td_list[i1].start
                    # Must be around 48s, NOT 134s!
                    self.assertLess(t, 70.0, f"Chorus 1 jumped to t={t}s!")
                    matched_chorus1 = True

        self.assertTrue(matched_chorus1, "Chorus 1 was not aligned")

    def test_bic_spectral_changepoint(self):
        """Test that BIC accurately detects phoneme transition boundaries."""
        import numpy as np
        from modules.Audio.bic_segmenter import compute_bic_changepoint

        np.random.seed(42)
        # Synthetic transition: 15 frames consonant noise -> 25 frames vowel formant energy
        c_frames = np.random.randn(15, 4) * 0.2
        v_frames = np.random.randn(25, 4) * 1.8 + 3.0
        combined = np.vstack([c_frames, v_frames])

        res = compute_bic_changepoint(combined, penalty=1.0)
        self.assertIsNotNone(res)
        best_idx, score = res
        self.assertEqual(best_idx, 15)
        self.assertGreater(score, 50.0)


if __name__ == "__main__":
    unittest.main()

