import os
import unittest
from src.modules.Audio.silence_processing import remove_silence2
import pytest
from modules.Speech_Recognition.TranscribedData import TranscribedData


class SilenceProcessingTest(unittest.TestCase):
    def test_remove_silence2(self):
        #
        # |    ***************         | silence_parts_list
        # |  00                        | before silence
        # |  00*******                 | end is in silence
        # |      ***                   | is in silence
        # |          *********00       | start is in silence
        # |                   00       | after silence
        # |  00***************00000000 | silence is inside
        # |0 1 2 3 4 5 6 7 8 9 10 11 12| time

        # Arrange

        silence_parts_list = [(2.0, 9.0)]
        transcribed_data = [
            TranscribedData({
                "conf": 0.95,
                "word": "Before silence",
                "end": 2.0,
                "start": 1.0
            }),
            TranscribedData({
                "conf": 0.95,
                "word": "End is in silence",
                "end": 5.0,
                "start": 1.0
            }),
            TranscribedData({
                "conf": 0.95,
                "word": "Is in silence",
                "end": 4.0,
                "start": 3.0
            }),
            TranscribedData({
                "conf": 0.95,
                "word": "Start is in silence",
                "end": 10.0,
                "start": 5.0
            }),
            TranscribedData({
                "conf": 0.95,
                "word": "After silence",
                "end": 10.0,
                "start": 9.0
            }),
            TranscribedData({
                "conf": 0.95,
                "word": "Silence is inside",
                "end": 12.0,
                "start": 1.0
            }),
        ]

        # Act
        result = remove_silence2(silence_parts_list, transcribed_data)

        # Assert
        self.assertEqual(result[0].word, "Before silence")
        self.assertEqual(result[0].start, 1.0)
        self.assertEqual(result[0].end, 2.0)

        self.assertEqual(result[1].word, "End is in silence")
        self.assertEqual(result[1].start, 1.0)
        self.assertEqual(result[1].end, 2.0)

        self.assertEqual(len(result), 6) # "in silence" is removed

        self.assertEqual(result[2].word, "Start is in silence")
        self.assertEqual(result[2].start, 9.0)
        self.assertEqual(result[2].end, 10.0)

        self.assertEqual(result[3].word, "After silence")
        self.assertEqual(result[3].start, 9.0)
        self.assertEqual(result[3].end, 10.0)

        # Split to "Silence is inside~"
        self.assertEqual(result[4].word, "Silence is inside")
        self.assertEqual(result[4].start, 1.0)
        self.assertEqual(result[4].end, 2.0)

        self.assertEqual(result[5].word, "~")
        self.assertEqual(result[5].start, 9.0)
        self.assertEqual(result[5].end, 12.0)

    def test_remove_multi_silence2(self):
        #
        # |    **  **  **  **          | silence_parts_list
        # |  00**00**00**00**00000000  | multi silence is inside
        # |0 1 2 3 4 5 6 7 8 9 10 11 12| time

        # Arrange

        silence_parts_list = [(2.0, 3.0), (4.0, 5.0), (6.0, 7.0), (8.0, 9.0)]
        transcribed_data = [
            TranscribedData({
                "conf": 0.95,
                "word": "Silence is inside",
                "end": 12.0,
                "start": 1.0
            }),
        ]

        # Act
        result = remove_silence2(silence_parts_list, transcribed_data)

        # Assert
        self.assertEqual(result[0].word, "Before silence")
        self.assertEqual(result[0].start, 1.0)
        self.assertEqual(result[0].end, 2.0)

        self.assertEqual(result[1].word, "~")
        self.assertEqual(result[1].start, 3.0)
        self.assertEqual(result[1].end, 4.0)

        self.assertEqual(result[2].word, "~")
        self.assertEqual(result[2].start, 5.0)
        self.assertEqual(result[2].end, 6.0)

        self.assertEqual(result[3].word, "~")
        self.assertEqual(result[3].start, 7.0)
        self.assertEqual(result[3].end, 8.0)

        self.assertEqual(result[4].word, "~")
        self.assertEqual(result[4].start, 9.0)
        self.assertEqual(result[4].end, 12.0)

        self.assertEqual(result[5].word, "~")
        self.assertEqual(result[5].start, 1.0)
        self.assertEqual(result[5].end, 2.0)

if __name__ == "__main__":
    unittest.main()
