"""Tests for image_helper.py"""

import unittest
from unittest.mock import patch, MagicMock
from src.modules.Image.image_helper import save_image

class TestImageHelper(unittest.TestCase):

    @patch("src.modules.Image.image_helper.Image.open")
    @patch("src.modules.Image.image_helper.os.path.join")
    @patch("src.modules.Image.image_helper.crop_image_to_square")
    def test_save_image(self, mock_crop_image_to_square, mock_os_path_join, mock_image_open):
        # Arrange
        mock_image = MagicMock()
        mock_image.convert.return_value = mock_image
        mock_image_open.return_value = mock_image
        mock_os_path_join.return_value = "/path/to/output/test.jpg"

        clear_filename = "test"
        output_path = "/path/to/output"

        # Act
        save_image(b'fake_image_bytes', clear_filename, output_path)

        # Assert
        mock_image.convert.assert_called_once_with('RGB')
        mock_os_path_join.assert_called_once_with(output_path, clear_filename + " [CO].jpg")
        mock_image.save.assert_called_once_with("/path/to/output/test.jpg", "JPEG")
        mock_crop_image_to_square.assert_called_once_with("/path/to/output/test.jpg")

if __name__ == "__main__":
    unittest.main()