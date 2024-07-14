"""Tests for UltraSinger.py"""

import unittest
import os
from unittest.mock import patch, MagicMock
from modules.Ultrastar.ultrastar_writer import format_separated_string
from src.UltraSinger import extract_year
from src.UltraSinger import parse_ultrastar_txt
from src import UltraSinger

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
    @patch("src.UltraSinger.os_helper.create_folder")
    def test_parse_ultrastar_txt(self, mock_create_folder,
                                 mock_dirname, mock_ultrastar_bpm_to_real_bpm,
                                 mock_parse_ultrastar_txt):
        # Arrange
        mock_parse_ultrastar_txt.return_value = MagicMock(mp3="test.mp3",
                                                          artist="  Test Artist  ",  # Also test leading and trailing whitespaces
                                                          title="  Test Title  ")  # Also test leading and trailing whitespaces
        mock_ultrastar_bpm_to_real_bpm.return_value = 120.0
        mock_dirname.return_value = os.path.join("path", "to", "input")
        UltraSinger.Settings.output_file_path = os.path.join("path", "to", "output")
        mock_create_folder.return_value = None

        # Act
        result = parse_ultrastar_txt()

        # Assert
        self.assertEqual(result, ("Test Artist - Test Title",
                                  120.0,
                                  os.path.join("path", "to", "output", "Test Artist - Test Title"),
                                  os.path.join("path", "to", "input", "test.mp3"),
                                  mock_parse_ultrastar_txt.return_value))

        mock_parse_ultrastar_txt.assert_called_once()
        mock_ultrastar_bpm_to_real_bpm.assert_called_once()
        mock_dirname.assert_called_once()
        mock_create_folder.assert_called_once()
