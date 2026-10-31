import unittest

from modules.Speech_Recognition.genre_profiles import (
    resolve_genre_profile,
    split_genre_tokens,
    SALSA,
    HIP_HOP_RNB,
    EDM_HOUSE_TECHNO,
    ROCK_ALTERNATIVE,
    POP_LATIN_POP,
)


class TestGenreProfiles(unittest.TestCase):
    def test_split_genre_tokens(self):
        self.assertEqual(
            split_genre_tokens("Merengue/House/Hip Hop"),
            ["Merengue", "House", "Hip Hop"],
        )
        self.assertEqual(
            split_genre_tokens("Rock; Pop, EDM + Trap & R&B"),
            ["Rock", "Pop", "EDM", "Trap", "R&B"],
        )
        self.assertEqual(split_genre_tokens("Salsa y Bachata"), ["Salsa", "Bachata"])
        self.assertEqual(split_genre_tokens(""), [])
        self.assertEqual(split_genre_tokens(None), [])

    def test_single_known_genres(self):
        self.assertEqual(resolve_genre_profile("Rock").name, ROCK_ALTERNATIVE.name)
        self.assertEqual(resolve_genre_profile("Merengue").name, SALSA.name)
        self.assertEqual(resolve_genre_profile("Hip Hop").name, HIP_HOP_RNB.name)
        self.assertEqual(resolve_genre_profile("House").name, EDM_HOUSE_TECHNO.name)
        self.assertEqual(resolve_genre_profile("").name, POP_LATIN_POP.name)
        self.assertEqual(resolve_genre_profile(None).name, POP_LATIN_POP.name)

    def test_multi_genre_merengue_house_hiphop(self):
        profile = resolve_genre_profile("Merengue/House/Hip Hop")

        # 1. Primary dominant family must be Salsa/Merengue, NOT Hip Hop
        self.assertIn("Latinos, Tropicales y Regionales", profile.family)
        self.assertIn("Merengue / House / Hip Hop", profile.name)

        # 2. VAD and DSP parameters must derive from primary (Salsa/Merengue)
        self.assertEqual(profile.vad_onset, SALSA.vad_onset)
        self.assertEqual(profile.vad_offset, SALSA.vad_offset)
        self.assertEqual(profile.vibrato_tolerance_semitones, SALSA.vibrato_tolerance_semitones)

        # 3. Bilingual prompts must combine primary and secondary elements
        prompt_en = profile.get_initial_prompt("en")
        self.assertIn("Merengue", prompt_en)
        self.assertIn("House", prompt_en)
        self.assertIn("Hip Hop", prompt_en)

        prompt_es = profile.get_initial_prompt("es")
        self.assertIn("Merengue", prompt_es)
        self.assertIn("House", prompt_es)
        self.assertIn("Hip Hop", prompt_es)

    def test_custom_unknown_single_genre(self):
        profile = resolve_genre_profile("Chiptune")
        self.assertEqual(profile.name, "Chiptune")
        self.assertIn("Personalizado", profile.family)
        self.assertIn("Chiptune", profile.get_initial_prompt("en"))
        self.assertIn("Chiptune", profile.get_initial_prompt("es"))

    def test_custom_unknown_fusion(self):
        profile = resolve_genre_profile("Hyperpop / Glitchcore")
        self.assertIn("Hyperpop / Glitchcore", profile.name)
        self.assertIn("Fusión", profile.family)
        self.assertIn("Hyperpop", profile.get_initial_prompt("es"))


if __name__ == "__main__":
    unittest.main()
