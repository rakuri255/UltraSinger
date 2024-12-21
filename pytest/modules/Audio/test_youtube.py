"""Tests for youtube.py"""

import unittest
from unittest.mock import patch
from src.modules.Audio.youtube import get_youtube_title
from src.modules.Audio.youtube import download_and_convert_thumbnail

class TestGetYoutubeTitle(unittest.TestCase):
    @patch("yt_dlp.YoutubeDL")
    def test_get_youtube_title(self, mock_youtube_dl):
        # Arrange
        # Also test leading and trailing whitespaces
        mock_youtube_dl.return_value.__enter__.return_value.extract_info.return_value = {
            "artist": "   Test Artist   ",
            "track": "   Test Track   ",
            "title": "   Test Artist - Test Track   ",
            "channel": "   Test Channel   "
        }
        url = "   https://fakeUrl   "

        # Act
        result = get_youtube_title(url)

        # Assert
        self.assertEqual(result, ("Test Artist", "Test Track"))
        mock_youtube_dl.assert_called_once()

    @patch("src.modules.Audio.youtube.yt_dlp.YoutubeDL")
    @patch("src.modules.Audio.youtube.save_image")
    def test_download_and_convert_thumbnail(self, mock_save_image, mock_youtube_dl):
        # Arrange
        mock_youtube_dl.return_value.__enter__.return_value.extract_info.return_value = {"thumbnail": "test_thumbnail_url"}
        mock_youtube_dl.return_value.__enter__.return_value.urlopen.return_value.read.return_value = b"test_image_data"

        mock_save_image.return_value = None

        ydl_opts = {}
        url = "https://fakeUrl"
        clear_filename = "test"
        output_path = "/path/to/output"

        # Act
        download_and_convert_thumbnail(ydl_opts, url, clear_filename, output_path)

        # Assert
        mock_youtube_dl.assert_called_once_with(ydl_opts)
        mock_youtube_dl.return_value.__enter__.return_value.extract_info.assert_called_once_with(url, download=False)
        mock_youtube_dl.return_value.__enter__.return_value.urlopen.assert_called_once_with("test_thumbnail_url")

if __name__ == "__main__":
    unittest.main()