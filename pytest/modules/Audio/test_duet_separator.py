"""Tests for duet_separator.py and duet writing in ultrastar_writer.py"""

import os
import tempfile
import unittest
import numpy as np

from modules.Audio.duet_separator import (
    group_midi_segments_into_phrases,
    extract_phrase_features,
    separate_duet_phrases,
)
from modules.Midi.MidiSegment import MidiSegment
from modules.Ultrastar.ultrastar_writer import create_ultrastar_txt
from modules.Ultrastar.ultrastar_txt import UltrastarTxtValue


class DuetSeparatorTest(unittest.TestCase):
    def test_group_midi_segments_into_phrases(self):
        # 4 notes: 2 in phrase 1, pause of 1.0s, 2 in phrase 2
        segments = [
            MidiSegment(note="C4", start=0.0, end=0.4, word="la "),
            MidiSegment(note="D4", start=0.4, end=0.8, word="la "),
            MidiSegment(note="E4", start=1.8, end=2.2, word="le "),
            MidiSegment(note="F4", start=2.2, end=2.6, word="le "),
        ]
        phrases = group_midi_segments_into_phrases(segments, silence_split_duration=0.5)
        self.assertEqual(len(phrases), 2)
        self.assertEqual(len(phrases[0]), 2)
        self.assertEqual(len(phrases[1]), 2)

    def test_extract_phrase_features(self):
        phrase = [
            MidiSegment(note="A4", start=0.0, end=0.5, word="hi "),
            MidiSegment(note="B4", start=0.5, end=1.0, word="there "),
        ]
        feat = extract_phrase_features("/nonexistent/vocals.wav", phrase)
        self.assertIsInstance(feat, np.ndarray)
        self.assertEqual(len(feat), 19)
        # Median pitch of A4 (69) and B4 (71) should be 70
        self.assertAlmostEqual(feat[0], 70.0, delta=1.0)

    def test_separate_duet_phrases_pitch_distinction(self):
        # 4 phrases: 2 deep male voice (A2/C3 ~ 45-48), 2 high female voice (A4/C5 ~ 69-72)
        male_phrase1 = [MidiSegment(note="A2", start=0.0, end=1.0, word="yo ")]
        female_phrase1 = [MidiSegment(note="A4", start=2.0, end=3.0, word="tu ")]
        male_phrase2 = [MidiSegment(note="B2", start=4.0, end=5.0, word="el ")]
        female_phrase2 = [MidiSegment(note="B4", start=6.0, end=7.0, word="ella ")]

        all_segments = male_phrase1 + female_phrase1 + male_phrase2 + female_phrase2
        player_ids, phrases, p1, p2 = separate_duet_phrases(
            "/nonexistent/vocals.wav",
            all_segments,
            silence_split_duration=0.5,
            p1_name="Romeo",
            p2_name="Julieta",
        )

        self.assertEqual(len(phrases), 4)
        self.assertEqual(len(player_ids), 4)
        self.assertEqual(p1, "Romeo")
        self.assertEqual(p2, "Julieta")
        # P1 starts the song (male_phrase1 is P1)
        self.assertEqual(player_ids[0], 1)
        # female_phrase1 should be clustered into P2
        self.assertEqual(player_ids[1], 2)
        # male_phrase2 should be P1
        self.assertEqual(player_ids[2], 1)
        # female_phrase2 should be P2
        self.assertEqual(player_ids[3], 2)

    def test_create_ultrastar_duet_file(self):
        segments = [
            MidiSegment(note="C4", start=1.0, end=1.4, word="hola "),
            MidiSegment(note="D4", start=1.5, end=1.9, word="amigo "),
            MidiSegment(note="G4", start=3.5, end=3.9, word="como "),
            MidiSegment(note="A4", start=4.0, end=4.4, word="estas "),
        ]
        ultrastar_class = UltrastarTxtValue()
        ultrastar_class.artist = "Dueto Artist"
        ultrastar_class.title = "Dueto Song"
        ultrastar_class.version = "1.2.0"
        ultrastar_class.mp3 = "song.mp3"

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            create_ultrastar_txt(
                midi_segments=segments,
                ultrastar_file_output=tmp_path,
                ultrastar_class=ultrastar_class,
                real_bpm=120.0,
                is_duet=True,
                p1_name="Cantante 1",
                p2_name="Cantante 2",
                phrase_player_ids=[1, 2],
            )

            with open(tmp_path, "r", encoding="utf-8") as f:
                content = f.read()

            self.assertIn("#P1:Cantante 1\n", content)
            self.assertIn("#P2:Cantante 2\n", content)
            self.assertIn("P 1\n", content)
            self.assertIn("P 2\n", content)

            idx_p1 = content.index("P 1\n")
            idx_p2 = content.index("P 2\n")
            self.assertLess(idx_p1, idx_p2, "Block P 1 must precede Block P 2")

            p1_block = content[idx_p1:idx_p2]
            p2_block = content[idx_p2:]

            self.assertIn("hola ", p1_block)
            self.assertIn("amigo ", p1_block)
            self.assertNotIn("como ", p1_block)

            self.assertIn("como ", p2_block)
            self.assertIn("estas ", p2_block)
            self.assertNotIn("hola ", p2_block)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_autocorrect_duet_parsing(self):
        from modules.Audio.ultrastar_autocorrect import parse_ultrastar_txt
        duet_content = (
            "#TITLE:Duet Song\n"
            "#ARTIST:Duet Artist\n"
            "#BPM:120\n"
            "#GAP:1000\n"
            "#P1:Romeo\n"
            "#P2:Julieta\n"
            "P 1\n"
            ": 0 4 10 Hola \n"
            "- 10\n"
            "P 2\n"
            ": 12 4 15 Adios \n"
            "- 20\n"
            "E\n"
        )
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", encoding="utf-8", delete=False) as tmp:
            tmp.write(duet_content)
            tmp_path = tmp.name

        try:
            headers, lines = parse_ultrastar_txt(tmp_path)
            self.assertEqual(headers.get("P1"), "Romeo")
            self.assertEqual(headers.get("P2"), "Julieta")

            p_lines = [l for l in lines if l.line_type == "P"]
            self.assertEqual(len(p_lines), 2)
            self.assertEqual(p_lines[0].raw_line, "P 1")
            self.assertEqual(p_lines[1].raw_line, "P 2")

            note_words = [l.word for l in lines if l.is_note]
            self.assertEqual(note_words, ["Hola ", "Adios "])
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
