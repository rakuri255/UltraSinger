"""Tests for lyrics_aligner module and song folder detection."""

import os
import tempfile
import unittest

from modules.Speech_Recognition.TranscribedData import TranscribedData
from modules.Speech_Recognition.lyrics_aligner import (
    load_lyrics_file,
    align_transcribed_data_with_lyrics,
    clean_word_for_matching,
)
from modules.Midi.MidiSegment import MidiSegment
from modules.Ultrastar.ultrastar_writer import create_ultrastar_txt
from modules.Ultrastar.ultrastar_txt import UltrastarTxtValue
from UltraSinger import find_song_audio_in_dir, find_lyrics_in_dir


class TestLyricsAligner(unittest.TestCase):
    """Test suite for reference lyrics alignment."""

    def test_load_lyrics_file(self):
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".txt") as f:
            f.write("Uno dos tres cuatro cinco seis siete ocho aquí\n")
            f.write("Uno dos tres cuatro ido\n")
            f.write("\n")  # Empty line (stanza break)
            f.write("Uno dos tres cuatro cinco prueba\n")
            temp_path = f.name

        try:
            words = load_lyrics_file(temp_path)
            self.assertEqual(len(words), 9 + 5 + 6)
            # Last word of first line
            self.assertEqual(words[8][0], "aquí")
            self.assertTrue(words[8][1])  # is_line_end == True
            # Words before it are not line end
            self.assertFalse(words[7][1])

            # Last word of second line
            self.assertEqual(words[13][0], "ido")
            self.assertTrue(words[13][1])

            # Last word of third line
            self.assertEqual(words[19][0], "prueba")
            self.assertTrue(words[19][1])
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_align_with_whisper_and_drop_hallucinations(self):
        # Simulated Whisper data with an extra hallucinated word ("la la la")
        transcribed = [
            TranscribedData(word="Canto ", start=1.0, end=1.5, confidence=0.9),
            TranscribedData(word="esta ", start=1.6, end=1.8, confidence=0.9),
            TranscribedData(word="cancion ", start=1.9, end=2.4, confidence=0.9),
            TranscribedData(word="extra ", start=2.5, end=2.7, confidence=0.3),  # Hallucination!
            TranscribedData(word="hoy ", start=2.8, end=3.2, confidence=0.9),
        ]

        lyrics_words = [
            ("Canto", False),
            ("esta", False),
            ("cancion", False),
            ("hoy", True),
        ]

        aligned = align_transcribed_data_with_lyrics(transcribed, lyrics_words)

        self.assertEqual(len(aligned), 4)
        words_out = [td.word.strip() for td in aligned]
        self.assertEqual(words_out, ["Canto", "esta", "cancion", "hoy"])
        self.assertNotIn("extra", words_out)
        self.assertTrue(aligned[-1].is_line_end)
        self.assertFalse(aligned[0].is_line_end)

    def test_folder_detection_helpers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            song_dir = os.path.join(temp_dir, "Test Artist - Test Title")
            os.makedirs(song_dir)

            audio_file = os.path.join(song_dir, "Test Artist - Test Title.mp3")
            with open(audio_file, "w") as f:
                f.write("audio")

            lyrics_file = os.path.join(song_dir, "lyrics.txt")
            with open(lyrics_file, "w") as f:
                f.write("Canto de prueba\n")

            found_audio = find_song_audio_in_dir(song_dir)
            found_lyrics = find_lyrics_in_dir(song_dir)

            self.assertEqual(found_audio, audio_file)
            self.assertEqual(found_lyrics, lyrics_file)

    def test_line_break_with_is_line_end(self):
        midi_segments = [
            MidiSegment("C4", 1.0, 1.4, "Can-", is_word_end=False, is_line_end=False),
            MidiSegment("D4", 1.5, 1.8, "to ", is_word_end=True, is_line_end=False),
            MidiSegment("E4", 1.9, 2.3, "hoy ", is_word_end=True, is_line_end=True),  # Line 1 ends -> linebreak 1
            MidiSegment("F4", 3.0, 3.4, "Que ", is_word_end=True, is_line_end=False),
            MidiSegment("G4", 3.5, 3.8, "bien ", is_word_end=True, is_line_end=True),  # Line 2 ends -> linebreak 2
            MidiSegment("A4", 5.0, 5.4, "se ", is_word_end=True, is_line_end=False),
            MidiSegment("B4", 5.5, 5.8, "ve ", is_word_end=True, is_line_end=True),  # Line 3 ends -> Song end (E)
        ]

        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".txt") as f:
            out_path = f.name

        try:
            u_class = UltrastarTxtValue()
            u_class.artist = "Artist"
            u_class.title = "Song"
            u_class.version = "1.2.0"
            u_class.mp3 = "song.mp3"

            create_ultrastar_txt(
                midi_segments,
                out_path,
                u_class,
                real_bpm=120.0,
                auto_octave=False,
            )

            with open(out_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Find all linebreaks (-)
            linebreaks = [l.strip() for l in lines if l.startswith("-")]
            self.assertEqual(len(linebreaks), 2)
            self.assertTrue(lines[-1].strip().startswith("E"))
        finally:
            if os.path.exists(out_path):
                os.remove(out_path)

    def test_accent_normalization_matching(self):
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as f:
            f.write("Yo sólo pienso en ti y en mí\n")
            lyrics_path = f.name

        try:
            # Whisper transcribed without accents: "solo", "mi"
            whisper_data = [
                TranscribedData(word="yo", confidence=0.9, start=1.0, end=1.4),
                TranscribedData(word="solo", confidence=0.9, start=1.5, end=1.9),
                TranscribedData(word="pienso", confidence=0.9, start=2.0, end=2.4),
                TranscribedData(word="en", confidence=0.9, start=2.5, end=2.7),
                TranscribedData(word="ti", confidence=0.9, start=2.8, end=3.1),
                TranscribedData(word="y", confidence=0.9, start=3.2, end=3.3),
                TranscribedData(word="en", confidence=0.9, start=3.4, end=3.6),
                TranscribedData(word="mi", confidence=0.9, start=3.7, end=4.0),
            ]

            aligned = align_transcribed_data_with_lyrics(whisper_data, lyrics_path)
            self.assertEqual(len(aligned), 8)
            # Accented words from lyrics.txt are preserved in output
            self.assertEqual(aligned[1].word.strip(), "sólo")
            self.assertEqual(aligned[-1].word.strip(), "mí")
            # All timings matched 1-to-1 without interpolation
            for orig, res in zip(whisper_data, aligned):
                self.assertAlmostEqual(res.start, orig.start)
        finally:
            if os.path.exists(lyrics_path):
                os.remove(lyrics_path)

    def test_phantom_duplicate_chorus_dropped_over_scat(self):
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as f:
            # Duplicate chorus in lyrics.txt
            f.write("Canta junto a mi\nCanta junto a mi\nQuiero que sepas\n")
            lyrics_path = f.name

        try:
            whisper_data = [
                # First chorus
                TranscribedData(word="Canta", confidence=0.9, start=1.0, end=1.5),
                TranscribedData(word="junto", confidence=0.9, start=1.6, end=2.0),
                TranscribedData(word="a", confidence=0.9, start=2.1, end=2.3),
                TranscribedData(word="mí", confidence=0.9, start=2.4, end=2.9),
                # Instrumental solo with scat adlibs
                TranscribedData(word="Tararara", confidence=0.8, start=5.0, end=7.0),
                TranscribedData(word="tararara", confidence=0.8, start=7.5, end=9.5),
                TranscribedData(word="tararara", confidence=0.8, start=10.0, end=12.0),
                # Next verse
                TranscribedData(word="Quiero", confidence=0.9, start=15.0, end=15.5),
                TranscribedData(word="que", confidence=0.9, start=15.6, end=15.8),
                TranscribedData(word="sepas", confidence=0.9, start=15.9, end=16.4),
            ]

            aligned = align_transcribed_data_with_lyrics(whisper_data, lyrics_path)
            words = [td.word.strip() for td in aligned]
            # Phantom duplicate chorus was dropped over scat; verse was matched cleanly
            self.assertEqual(words, ["Canta", "junto", "a", "mi", "Quiero", "que", "sepas"])
            self.assertAlmostEqual(aligned[-1].start, 15.9)
        finally:
            if os.path.exists(lyrics_path):
                os.remove(lyrics_path)


if __name__ == "__main__":
    unittest.main()
