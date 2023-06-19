"""Tests for the ultrastar_writer.py module."""

import unittest
from unittest.mock import patch, mock_open
from src.modules.Ultrastar.ultrastar_writer import create_ultrastar_txt_from_automation
from src.modules.Speech_Recognition.TranscribedData import TranscribedData
from src.modules.Ultrastar.ultrastar_txt import UltrastarTxtValue, UltrastarTxtTag


class TestCreateUltrastarTxt(unittest.TestCase):
    def test_create_ultrastar_txt_from_automation_default_values(self):
        # Arrange
        bpm, note_numbers, transcribed_data, ultrastar_file_output = self.arrange()

        # Assert default values
        class_under_test = UltrastarTxtValue()
        expected_calls = self.default_values(class_under_test)

        self.act_and_assert(bpm, class_under_test, expected_calls, note_numbers, transcribed_data,
                            ultrastar_file_output)

    def test_create_ultrastar_txt_from_automation_full_values(self):
        # Arrange
        bpm, note_numbers, transcribed_data, ultrastar_file_output = self.arrange()

        # Assert full values
        class_under_test = UltrastarTxtValue()
        class_under_test.artist = "artist"
        class_under_test.title = "title"
        class_under_test.year = "2023"
        class_under_test.language = "de"
        class_under_test.genre = "pop, rock"
        class_under_test.cover = "cover [CO].jpg"
        class_under_test.video = "video.mp4"
        class_under_test.mp3 = "music.mp3"

        expected_calls = self.full_values(class_under_test)

        self.act_and_assert(bpm, class_under_test, expected_calls, note_numbers, transcribed_data,
                            ultrastar_file_output)

    def arrange(self):
        # Arrange
        transcribed_data = [
            TranscribedData({
                "conf": 0.95,
                "word": "UltraSinger ",
                "end": 2.5,
                "start": 0.5
            }),
            TranscribedData({
                "conf": 0.9,
                "word": "is ",
                "end": 4.5,
                "start": 3.0
            }),
            TranscribedData({
                "conf": 0.85,
                "word": "cool! ",
                "end": 7.5,
                "start": 5.5
            }),
        ]
        note_numbers = [1, 2, 3]
        ultrastar_file_output = "output.txt"
        bpm = 120
        return bpm, note_numbers, transcribed_data, ultrastar_file_output

    def act_and_assert(self, bpm, default_ultrastar_class, expected_calls_default_values, note_numbers,
                       transcribed_data, ultrastar_file_output):
        # Act
        mock_file = self.act(bpm, default_ultrastar_class, note_numbers, transcribed_data, ultrastar_file_output)

        # Assert the file was opened and is utf-8
        mock_file.assert_called_once_with(ultrastar_file_output, "w", encoding='utf-8')

        # Assert that expected_calls_default_values were written to the file
        mock_file_handle = mock_file.return_value.__enter__.return_value.write
        write_calls = [args[0][0] for args in mock_file_handle.call_args_list]
        self.assertEqual(write_calls, expected_calls_default_values)

    @staticmethod
    def default_values(default_ultrastar_class):
        expected_calls = [
            f"#{UltrastarTxtTag.ARTIST}:{default_ultrastar_class.artist}\n",
            f"#{UltrastarTxtTag.TITLE}:{default_ultrastar_class.title}\n",
            f"#{UltrastarTxtTag.MP3}:{default_ultrastar_class.mp3}\n",
            f"#{UltrastarTxtTag.VIDEO}:{default_ultrastar_class.video}\n",
            f"#{UltrastarTxtTag.BPM}:390.0\n",
            f"#{UltrastarTxtTag.GAP}:500\n",
            f"#{UltrastarTxtTag.CREATOR}:{default_ultrastar_class.creator}\n",
            f"#{UltrastarTxtTag.FIXER}:{default_ultrastar_class.fixer}\n",
            f"#{UltrastarTxtTag.COMMENT}:{default_ultrastar_class.comment}\n",
            ": 0 52 1 UltraSinger \n",
            "- 52\n",
            ": 65 39 2 is \n",
            "- 104\n",
            ": 130 52 3 cool! \n",
            "E"
        ]
        return expected_calls

    @staticmethod
    def full_values(default_ultrastar_class):
        expected_calls = [
            f"#{UltrastarTxtTag.ARTIST}:{default_ultrastar_class.artist}\n",
            f"#{UltrastarTxtTag.TITLE}:{default_ultrastar_class.title}\n",
            f"#{UltrastarTxtTag.YEAR}:{default_ultrastar_class.year}\n",
            f"#{UltrastarTxtTag.LANGUAGE}:German\n",
            f"#{UltrastarTxtTag.GENRE}:{default_ultrastar_class.genre}\n",
            f"#{UltrastarTxtTag.COVER}:{default_ultrastar_class.cover}\n",
            f"#{UltrastarTxtTag.MP3}:{default_ultrastar_class.mp3}\n",
            f"#{UltrastarTxtTag.VIDEO}:{default_ultrastar_class.video}\n",
            f"#{UltrastarTxtTag.BPM}:390.0\n",
            f"#{UltrastarTxtTag.GAP}:500\n",
            f"#{UltrastarTxtTag.CREATOR}:{default_ultrastar_class.creator}\n",
            f"#{UltrastarTxtTag.FIXER}:{default_ultrastar_class.fixer}\n",
            f"#{UltrastarTxtTag.COMMENT}:{default_ultrastar_class.comment}\n",
            ": 0 52 1 UltraSinger \n",
            "- 52\n",
            ": 65 39 2 is \n",
            "- 104\n",
            ": 130 52 3 cool! \n",
            "E"
        ]
        return expected_calls

    @staticmethod
    def act(bpm, default_ultrastar_class, note_numbers, transcribed_data, ultrastar_file_output):
        with patch("builtins.open", mock_open()) as mock_file:
            create_ultrastar_txt_from_automation(
                transcribed_data, note_numbers, ultrastar_file_output,
                default_ultrastar_class, bpm
            )
        return mock_file

if __name__ == "__main__":
    unittest.main()