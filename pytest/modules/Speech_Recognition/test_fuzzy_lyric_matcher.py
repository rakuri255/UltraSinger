"""Tests for fuzzy_lyric_matcher and ad-lib rejection layer."""

import unittest
from modules.Speech_Recognition.TranscribedData import TranscribedData
from modules.Speech_Recognition.fuzzy_lyric_matcher import (
    character_level_similarity,
    normalize_text_for_phonetic_matching,
    is_scat_or_adlib_pattern,
    filter_intro_adlibs,
    fuzzy_lyric_matcher,
)
from modules.Speech_Recognition.lyrics_aligner import align_transcribed_data_with_lyrics


class TestFuzzyLyricMatcher(unittest.TestCase):
    """Test suite for Character-Level Fuzzy Matching and Intro Ad-lib Rejection."""

    def test_normalization_and_similarity(self):
        # Accents and case invariance
        self.assertEqual(character_level_similarity("corazón", "corazon"), 1.0)
        self.assertEqual(character_level_similarity("¡Canté!", "cante"), 1.0)
        self.assertEqual(character_level_similarity("Qué", "que"), 1.0)

        # Ad-lib vs real lyric: Null similarity
        sim_ah_yo = character_level_similarity("Ahhh", "Yo")
        self.assertEqual(sim_ah_yo, 0.0)

        sim_mmm_yo = character_level_similarity("Mmmm", "Yo")
        self.assertEqual(sim_mmm_yo, 0.0)

        # Monosyllables with partial phonetic overlap
        sim_yo_lo = character_level_similarity("Yo", "lo")
        self.assertGreaterEqual(sim_yo_lo, 0.50)

    def test_scat_and_adlib_detection(self):
        # Common singing ad-libs and warmups
        self.assertTrue(is_scat_or_adlib_pattern("Ahhh"))
        self.assertTrue(is_scat_or_adlib_pattern("Mmmm"))
        self.assertTrue(is_scat_or_adlib_pattern("uhm"))
        self.assertTrue(is_scat_or_adlib_pattern("tadarara"))
        self.assertTrue(is_scat_or_adlib_pattern("tarara"))
        self.assertTrue(is_scat_or_adlib_pattern("ohhh"))
        self.assertTrue(is_scat_or_adlib_pattern("la la"))

        # Real Spanish dictionary words
        self.assertFalse(is_scat_or_adlib_pattern("Yo"))
        self.assertFalse(is_scat_or_adlib_pattern("sé"))
        self.assertFalse(is_scat_or_adlib_pattern("puerta"))
        self.assertFalse(is_scat_or_adlib_pattern("amor"))
        self.assertFalse(is_scat_or_adlib_pattern("cantar"))

    def test_filter_intro_adlibs_with_transcribed_data(self):
        # Simulated Whisper output with initial vocalizations (Ahhh, Mmmm) before "Yo"
        whisper_data = [
            TranscribedData(word="Ahhh ", start=2.10, end=4.30, confidence=0.85),
            TranscribedData(word="Mmmm ", start=5.20, end=6.00, confidence=0.80),
            TranscribedData(word="Yo ", start=8.40, end=8.85, confidence=0.95),
            TranscribedData(word="sé ", start=8.90, end=9.30, confidence=0.95),
            TranscribedData(word="que ", start=9.35, end=9.60, confidence=0.92),
        ]
        target_lyrics = ["Yo", "sé", "que", "te", "vas"]

        valid, quarantined = filter_intro_adlibs(whisper_data, target_lyrics, similarity_threshold=0.45)

        # Verification: First 2 ad-libs rejected, anchor is "Yo" at 8.40s
        self.assertEqual(len(quarantined), 2)
        self.assertEqual(len(valid), 3)
        self.assertEqual(valid[0].word.strip(), "Yo")
        self.assertEqual(valid[0].start, 8.40)

    def test_filter_intro_adlibs_dict_mode(self):
        # Dict format as used by autocorrect
        whisper_dicts = [
            {"word": "Ahhh", "clean": "ahhh", "start": 1.5, "end": 3.0, "is_vocablo": False},
            {"word": "Yo", "clean": "yo", "start": 7.0, "end": 7.5, "is_vocablo": False},
            {"word": "sé", "clean": "se", "start": 7.6, "end": 8.0, "is_vocablo": False},
        ]
        target_words = ["Yo", "sé"]

        valid, quarantined = filter_intro_adlibs(whisper_dicts, target_words, similarity_threshold=0.45)

        self.assertEqual(len(quarantined), 1)
        self.assertTrue(quarantined[0]["is_vocablo"])
        self.assertTrue(quarantined[0]["adlib_rejected"])
        self.assertEqual(valid[0]["word"], "Yo")
        self.assertEqual(valid[0]["start"], 7.0)

    def test_end_to_end_lyrics_aligner_adlib_rejection(self):
        # End-to-end integration: align_transcribed_data_with_lyrics with intro ad-lib
        whisper_data = [
            TranscribedData(word="Ahhh ", start=2.00, end=4.00, confidence=0.88),  # Ad-lib
            TranscribedData(word="Yo ", start=7.50, end=7.80, confidence=0.94),
            TranscribedData(word="no ", start=7.85, end=8.10, confidence=0.95),
            TranscribedData(word="sé ", start=8.15, end=8.50, confidence=0.92),
        ]
        lyrics_words = [
            ("Yo", False),
            ("no", False),
            ("sé", True),
        ]

        aligned = align_transcribed_data_with_lyrics(whisper_data, lyrics_words)

        # Output must contain exactly 3 words: "Yo ", "no ", "sé "
        # and t_start of "Yo" MUST be 7.50s, NOT 2.00s!
        self.assertEqual(len(aligned), 3)
        self.assertEqual(aligned[0].word.strip(), "Yo")
        self.assertEqual(aligned[0].start, 7.50)
        self.assertEqual(aligned[1].word.strip(), "no")
        self.assertEqual(aligned[2].word.strip(), "sé")
        self.assertTrue(aligned[2].is_line_end)

    def test_find_song_folders_in_dir(self):
        import tempfile
        import os
        from modules.Audio.ultrastar_autocorrect import find_song_folders_in_dir

        with tempfile.TemporaryDirectory() as temp_dir:
            # Subfolder 1 with .txt (Valid song)
            s1 = os.path.join(temp_dir, "Artist - Song 1")
            os.makedirs(s1)
            with open(os.path.join(s1, "Artist - Song 1.txt"), "w") as f:
                f.write("#TITLE:Song 1\n")

            # Subfolder 2 with .txt (Valid song)
            s2 = os.path.join(temp_dir, "Artist - Song 2")
            os.makedirs(s2)
            with open(os.path.join(s2, "Artist - Song 2.txt"), "w") as f:
                f.write("#TITLE:Song 2\n")

            # Subfolder 3 without .txt (Not a song)
            s3 = os.path.join(temp_dir, "Other Folder")
            os.makedirs(s3)

            # Test Batch discovery: passing parent directory (e.g. /output/)
            found_batch = find_song_folders_in_dir(temp_dir)
            self.assertEqual(len(found_batch), 2)
            self.assertIn(s1, found_batch)
            self.assertIn(s2, found_batch)
            self.assertNotIn(s3, found_batch)

            # Test Single song folder discovery
            found_single = find_song_folders_in_dir(s1)
            self.assertEqual(found_single, [s1])


if __name__ == "__main__":
    unittest.main()
