"""Tests for UltraSinger.py"""

import unittest
from unittest.mock import patch, MagicMock
from src.UltraSinger import format_separated_string
from src.UltraSinger import extract_year
from src.UltraSinger import parse_ultrastar_txt

class TestUltraSinger(unittest.TestCase):
    def test_format_separated_string(self):

        self.assertEqual(format_separated_string('rock,pop,rock-pop,'), 'Rock, Pop, Rock-Pop')
        self.assertEqual(format_separated_string('rock;pop;rock-pop,'), 'Rock, Pop, Rock-Pop')
        self.assertEqual(format_separated_string('rock/pop/rock-pop,'), 'Rock, Pop, Rock-Pop')
        self.assertEqual(format_separated_string('rock,pop/rock-pop;80s,'), 'Rock, Pop, Rock-Pop, 80s')
        self.assertEqual(format_separated_string('rock, pop, rock-pop, '), 'Rock, Pop, Rock-Pop')

    def test_extract_year(self):
        years = {extract_year("2023-12-31"), extract_year("2023-12-31 23:59:59"), extract_year("2023/12/31"),
                 extract_year("2023\\12\\31"), extract_year("2023.12.31"), extract_year("2023 12 31"),
                 extract_year("12-31-2023"), extract_year("12/31/2023"), extract_year("12\\31\\2023"),
                 extract_year("12.31.2023"), extract_year("12 31 2023"), extract_year("12-2023"),
                 extract_year("12/2023"), extract_year("2023")}

        for year in years:
            self.assertEqual(year, "2023")

    @patch("src.UltraSinger.ultrastar_parser.parse_ultrastar_txt")
    @patch("src.UltraSinger.ultrastar_converter.ultrastar_bpm_to_real_bpm")
    @patch("src.UltraSinger.os.path.dirname")
    @patch("src.UltraSinger.get_unused_song_output_dir")
    @patch("src.UltraSinger.os_helper.create_folder")
    def test_parse_ultrastar_txt(self, mock_create_folder, mock_get_unused_song_output_dir,
                                 mock_dirname, mock_ultrastar_bpm_to_real_bpm,
                                 mock_parse_ultrastar_txt):
        # Arrange
        mock_parse_ultrastar_txt.return_value = MagicMock(mp3="test.mp3",
                                                          artist="  Test Artist  ",  # Also test leading and trailing whitespaces
                                                          title="  Test Title  ")  # Also test leading and trailing whitespaces
        mock_ultrastar_bpm_to_real_bpm.return_value = 120.0
        mock_dirname.return_value = "\\path\\to\\input"
        mock_get_unused_song_output_dir.return_value = "\\path\\to\\output\\Test Artist - Test Title"
        mock_create_folder.return_value = None

        # Act
        result = parse_ultrastar_txt()

        # Assert
        self.assertEqual(result, ("test",
                                  120.0,
                                  "\\path\\to\\output\\Test Artist - Test Title",
                                  "\\path\\to\\input\\test.mp3",
                                  mock_parse_ultrastar_txt.return_value))

        mock_parse_ultrastar_txt.assert_called_once()
        mock_ultrastar_bpm_to_real_bpm.assert_called_once()
        mock_dirname.assert_called_once()
        mock_get_unused_song_output_dir.assert_called_once()
        mock_create_folder.assert_called_once()
