import os
import unittest
from src.modules.Audio.silence_processing import remove_silence2
import pytest
from modules.Speech_Recognition.TranscribedData import TranscribedData


class SilenceProcessingTest(unittest.TestCase):
    def test_remove_silence2(self):
        # Arrange
        timeList = [(2.0, 9.0)]
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
        result = remove_silence2(timeList, transcribed_data)

        # Assert
        self.assertEqual(result[0].word, "Before silence")
        self.assertEqual(result[0].start, 1.0)
        self.assertEqual(result[0].end, 2.0)

        self.assertEqual(result[1].word, "End is in silence")
        self.assertEqual(result[1].start, 1.0)
        self.assertEqual(result[1].end, 2.0)

        self.assertEqual(len(result), 6) # "in silence" was removed

        self.assertEqual(result[2].word, "Start is in silence")
        self.assertEqual(result[2].start, 9.0)
        self.assertEqual(result[2].end, 10.0)

        self.assertEqual(result[3].word, "After silence")
        self.assertEqual(result[3].start, 9.0)
        self.assertEqual(result[3].end, 10.0)

        #self.assertEqual(transcribed_data[3].word, "Silence is inside")
        self.assertEqual(result[4].start, 1.0)
        self.assertEqual(result[4].end, 2.0)

        #self.assertEqual(transcribed_data[3].word, "Silence is inside")
        self.assertEqual(result[5].start, 9.0)
        self.assertEqual(result[5].end, 12.0)

if __name__ == "__main__":
    unittest.main()
