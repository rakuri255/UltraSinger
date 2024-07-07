"""Tests for silence_processing.py"""

import unittest
from src.modules.Audio.silence_processing import remove_silence
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
            TranscribedData(word="Before silence ", confidence=0.95, start=1.0, end=2.0),
            TranscribedData(word="End is in silence ", confidence=0.95, start=1.0, end=5.0),
            TranscribedData(word="Is in silence ", confidence=0.95, start=3.0, end=4.0),
            TranscribedData(word="Start is in silence ", confidence=0.95, start=5.0, end=10.0),
            TranscribedData(word="After silence ", confidence=0.95, start=9.0, end=10.0),
            TranscribedData(word="Silence is inside ", confidence=0.95, start=1.0, end=12.0),
        ]

        # Act
        result = remove_silence(silence_parts_list, transcribed_data)

        # Assert
        self.assertEqual(result[0].word, "Before silence ")
        self.assertEqual(result[0].start, 1.0)
        self.assertEqual(result[0].end, 2.0)
        self.assertEqual(result[0].is_word_end, True)

        self.assertEqual(result[1].word, "End is in silence ")
        self.assertEqual(result[1].start, 1.0)
        self.assertEqual(result[1].end, 2.0)
        self.assertEqual(result[1].is_word_end, True)

        self.assertEqual(len(result), 6) # "in silence" is removed

        self.assertEqual(result[2].word, "Start is in silence ")
        self.assertEqual(result[2].start, 9.0)
        self.assertEqual(result[2].end, 10.0)
        self.assertEqual(result[2].is_word_end, True)

        self.assertEqual(result[3].word, "After silence ")
        self.assertEqual(result[3].start, 9.0)
        self.assertEqual(result[3].end, 10.0)
        self.assertEqual(result[3].is_word_end, True)

        # Split to "Silence is inside~ "
        self.assertEqual(result[4].word, "Silence is inside")
        self.assertEqual(result[4].start, 1.0)
        self.assertEqual(result[4].end, 2.0)
        self.assertEqual(result[4].is_word_end, False)

        self.assertEqual(result[5].word, "~ ")
        self.assertEqual(result[5].start, 9.0)
        self.assertEqual(result[5].end, 12.0)
        self.assertEqual(result[5].is_word_end, True)

    def test_remove_multiple_silence_in_between_silence2(self):
        #
        # |    **  **  **  **          | silence_parts_list
        # |  00**00**00**00**00000000  | multi silence is inside
        # |0 1 2 3 4 5 6 7 8 9 10 11 12| time

        # Arrange

        silence_parts_list = [(2.0, 3.0), (4.0, 5.0), (6.0, 7.0), (8.0, 9.0)]
        transcribed_data = [
            TranscribedData(word="Silence is inside ", confidence=0.95, start=1.0, end=12.0),
        ]

        # Act
        result = remove_silence(silence_parts_list, transcribed_data)

        # Assert
        self.assertEqual(result[0].word, "Silence is inside")
        self.assertEqual(result[0].start, 1.0)
        self.assertEqual(result[0].end, 2.0)
        self.assertEqual(result[0].is_word_end, False)

        self.assertEqual(result[1].word, "~")
        self.assertEqual(result[1].start, 3.0)
        self.assertEqual(result[1].end, 4.0)
        self.assertEqual(result[1].is_word_end, False)

        self.assertEqual(result[2].word, "~")
        self.assertEqual(result[2].start, 5.0)
        self.assertEqual(result[2].end, 6.0)
        self.assertEqual(result[2].is_word_end, False)

        self.assertEqual(result[3].word, "~")
        self.assertEqual(result[3].start, 7.0)
        self.assertEqual(result[3].end, 8.0)
        self.assertEqual(result[3].is_word_end, False)

        self.assertEqual(result[4].word, "~ ")
        self.assertEqual(result[4].start, 9.0)
        self.assertEqual(result[4].end, 12.0)
        self.assertEqual(result[4].is_word_end, True)


    def test_remove_multiple_silence_till_end_silence2(self):
        #
        # |    **  **  **  **          | silence_parts_list
        # |  00**00**00**00**          | multi silence is inside
        # |0 1 2 3 4 5 6 7 8 9 10 11 12| time

        # Arrange

        silence_parts_list = [(2.0, 3.0), (4.0, 5.0), (6.0, 7.0), (8.0, 9.0)]
        transcribed_data = [
            TranscribedData(word="Silence is inside ", confidence=0.95, start=1.0, end=9.0),
        ]

        # Act
        result = remove_silence(silence_parts_list, transcribed_data)

        # Assert
        self.assertEqual(result[0].word, "Silence is inside")
        self.assertEqual(result[0].start, 1.0)
        self.assertEqual(result[0].end, 2.0)
        self.assertEqual(result[0].is_word_end, False)

        self.assertEqual(result[1].word, "~")
        self.assertEqual(result[1].start, 3.0)
        self.assertEqual(result[1].end, 4.0)
        self.assertEqual(result[1].is_word_end, False)

        self.assertEqual(result[2].word, "~")
        self.assertEqual(result[2].start, 5.0)
        self.assertEqual(result[2].end, 6.0)
        self.assertEqual(result[2].is_word_end, False)

        self.assertEqual(result[3].word, "~ ")
        self.assertEqual(result[3].start, 7.0)
        self.assertEqual(result[3].end, 8.0)
        self.assertEqual(result[3].is_word_end, True)

    def test_remove_no_duration_silence2(self):
        #
        # |    * * * * * *   | silence_parts_list
        # |  000 000 00000   | multi silence is inside
        # |0 1 2 3 4 5 6 7 8 | time

        # Arrange

        silence_parts_list = [(2.0, 2.1), (3.09, 3.1), (4.0, 4.09), (5.0, 5.09), (6.0, 6.09), (7.0, 7.09)]
        transcribed_data = [
            TranscribedData(word="Remove split ", confidence=0.95, start=1.0, end=2.19),
            TranscribedData(word="Is split ", confidence=0.95, start=3.0, end=4.1),
            TranscribedData(word="Is split2 ", confidence=0.95, start=5.0, end=7.1),
        ]

        # Act
        result = remove_silence(silence_parts_list, transcribed_data)

        # Assert
        self.assertEqual(result[0].word, "Remove split ")
        self.assertEqual(result[0].start, 1.0)
        self.assertEqual(result[0].end, 2.0)
        self.assertEqual(result[0].is_word_end, True)

        self.assertEqual(result[1].word, "Is split ")
        self.assertEqual(result[1].start, 3.1)
        self.assertEqual(result[1].end, 4.0)
        self.assertEqual(result[1].is_word_end, True)

        self.assertEqual(len(result), 4) # "removed " is removed

        self.assertEqual(result[2].word, "Is split2")
        self.assertEqual(result[2].start, 5.09)
        self.assertEqual(result[2].end, 6.0)
        self.assertEqual(result[2].is_word_end, False)

        self.assertEqual(result[3].word, "~ ")
        self.assertEqual(result[3].start, 6.09)
        self.assertEqual(result[3].end, 7.0)
        self.assertEqual(result[3].is_word_end, True)

if __name__ == "__main__":
    unittest.main()
