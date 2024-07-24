"""Tests for ultrastar_parser.py"""

import unittest
import os
from unittest.mock import patch, MagicMock
from modules.Ultrastar.ultrastar_parser import parse_ultrastar_txt


class TestUltraStarParser(unittest.TestCase):
    @patch("modules.Ultrastar.ultrastar_parser.parse")
    @patch("modules.Ultrastar.ultrastar_parser.os.path.dirname")
    @patch("modules.os_helper.create_folder")
    def test_parse_ultrastar_txt(self, mock_create_folder, mock_dirname, mock_parse):
        # Arrange
        mock_parse.return_value = MagicMock(mp3="test.mp3",
                                            artist="  Test Artist  ",
                                            # Also test leading and trailing whitespaces
                                            title="  Test Title  ")  # Also test leading and trailing whitespaces

        mock_dirname.return_value = os.path.join("path", "to", "input")
        output_file_path = os.path.join("path", "to", "output")

        mock_create_folder.return_value = None

        # Act
        result = parse_ultrastar_txt(mock_dirname.return_value, output_file_path)

        # Assert
        self.assertEqual(result, ("Test Artist - Test Title",
                                  os.path.join("path", "to", "output", "Test Artist - Test Title"),
                                  os.path.join("path", "to", "input", "test.mp3"),
                                  mock_parse.return_value))
        #
        mock_parse.assert_called_once()
        mock_dirname.assert_called_once()
        mock_create_folder.assert_called_once()
