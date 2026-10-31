import os
import tempfile
import unittest

from modules.Ultrastar.ultrastar_writer import format_separated_string
from modules.Ultrastar.ultrastar_parser import parse


class TestGenreFormatAndParse(unittest.TestCase):
    def test_format_separated_string_fusions(self):
        # Merengue / House / Hip Hop
        self.assertEqual(
            format_separated_string("Merengue/House/Hip Hop"),
            "Merengue, House, Hip Hop",
        )
        # Semicolons and mixed
        self.assertEqual(
            format_separated_string("Salsa; Bachata, Merengue"),
            "Salsa, Bachata, Merengue",
        )
        # Acronyms preserved in uppercase
        self.assertEqual(
            format_separated_string("edm/r&b/ost/dj"),
            "EDM, R&B, OST, DJ",
        )
        # Special hyphenated acronyms
        self.assertEqual(
            format_separated_string("k-pop, lo-fi; hip-hop"),
            "K-Pop, Lo-Fi, Hip-Hop",
        )
        # Standard legacy test cases
        self.assertEqual(
            format_separated_string("rock,pop/rock-pop;80s,"),
            "Rock, Pop, Rock-Pop, 80s",
        )

    def test_ultrastar_parser_reads_genre(self):
        content = (
            "#ARTIST:Test Artist\n"
            "#TITLE:Test Title\n"
            "#GENRE:Merengue/House/Hip Hop\n"
            "#YEAR:1993\n"
            "#LANGUAGE:Spanish\n"
            "#BPM:480\n"
            "#GAP:1000\n"
            "#MP3:music.mp3\n"
            ": 0 4 1 Can\n"
            ": 4 4 1 cion\n"
            ": 8 4 1 hoy\n"
            "E\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as tf:
            tf.write(content)
            temp_path = tf.name

        try:
            ultrastar_obj = parse(temp_path)
            self.assertEqual(ultrastar_obj.artist, "Test Artist")
            self.assertEqual(ultrastar_obj.title, "Test Title")
            self.assertEqual(ultrastar_obj.genre, "Merengue/House/Hip Hop")
            self.assertEqual(ultrastar_obj.year, "1993")
            self.assertEqual(ultrastar_obj.language, "Spanish")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


if __name__ == "__main__":
    unittest.main()

