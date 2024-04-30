import unittest
from unittest.mock import patch, MagicMock
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
        url = "   https://www.youtube.com/watch?v=dQw4w9WgXcQ   "

        # Act
        result = get_youtube_title(url)

        # Assert
        self.assertEqual(result, ("Test Artist", "Test Track"))
        mock_youtube_dl.assert_called_once()

    @patch("src.modules.Audio.youtube.yt_dlp.YoutubeDL")
    @patch("src.modules.Audio.youtube.Image.open")
    @patch("src.modules.Audio.youtube.os.path.join")
    @patch("src.modules.Audio.youtube.crop_image_to_square")
    def test_download_and_convert_thumbnail(self, mock_crop_image_to_square, mock_os_path_join, mock_image_open, mock_youtube_dl):
        # Arrange
        mock_youtube_dl.return_value.__enter__.return_value.extract_info.return_value = {"thumbnail": "test_thumbnail_url"}
        mock_youtube_dl.return_value.__enter__.return_value.urlopen.return_value.read.return_value = b"test_image_data"
        mock_image = MagicMock()
        mock_image.convert.return_value = mock_image
        mock_image_open.return_value = mock_image
        mock_os_path_join.return_value = "/path/to/output/test.jpg"

        ydl_opts = {}
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        clear_filename = "test"
        output_path = "/path/to/output"

        # Act
        download_and_convert_thumbnail(ydl_opts, url, clear_filename, output_path)

        # Assert
        mock_youtube_dl.assert_called_once_with(ydl_opts)
        mock_youtube_dl.return_value.__enter__.return_value.extract_info.assert_called_once_with(url, download=False)
        mock_youtube_dl.return_value.__enter__.return_value.urlopen.assert_called_once_with("test_thumbnail_url")
        mock_image.convert.assert_called_once_with('RGB')
        mock_os_path_join.assert_called_once_with(output_path, clear_filename + " [CO].jpg")
        mock_image.save.assert_called_once_with("/path/to/output/test.jpg", "JPEG")
        mock_crop_image_to_square.assert_called_once_with("/path/to/output/test.jpg")

if __name__ == "__main__":
    unittest.main()