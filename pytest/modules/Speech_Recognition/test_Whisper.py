"""Tests for whisper.py"""

import unittest
from src.modules.Speech_Recognition.TranscribedData import TranscribedData
from src.modules.Speech_Recognition.Whisper import convert_to_transcribed_data, number_to_words

class ConvertToTranscribedDataTest(unittest.TestCase):
    def test_convert_to_transcribed_data(self):
        # Arrange
        result_aligned = {
            "segments": [
                {
                    "words": [
                        {"word": "UltraSinger", "start": 1.23, "end": 2.34, "confidence": 0.95},
                        {"word": "is", "start": 2.34, "end": 3.45, "confidence": 0.9},
                        {"word": "cool!", "start": 3.45, "end": 4.56, "confidence": 0.85},
                    ]
                },
                {
                    "words": [
                        {"word": "And", "start": 4.56, "end": 5.67, "confidence": 0.95},
                        {"word": "will", "start": 5.67, "end": 6.78, "confidence": 0.9},
                        {"word": "be", "start": 6.78, "end": 7.89, "confidence": 0.85},
                        {"word": "better!", "start": 7.89, "end": 9.01, "confidence": 0.8},
                    ]
                },
            ]
        }

        # Words should have space at the end
        expected_output = [
            TranscribedData(word="UltraSinger ", start=1.23, end=2.34, is_hyphen=False, confidence=0.95),
            TranscribedData(word="is ", start=2.34, end=3.45, is_hyphen=False, confidence=0.9),
            TranscribedData(word="cool! ", start=3.45, end=4.56, is_hyphen=False, confidence=0.85),
            TranscribedData(word="And ", start=4.56, end=5.67, is_hyphen=False, confidence=0.95),
            TranscribedData(word="will ", start=5.67, end=6.78, is_hyphen=False, confidence=0.9),
            TranscribedData(word="be ", start=6.78, end=7.89, is_hyphen=False, confidence=0.85),
            TranscribedData(word="better! ", start=7.89, end=9.01, is_hyphen=False, confidence=0.8),
        ]

        # Act
        transcribed_data = convert_to_transcribed_data(result_aligned)

        # Assert
        self.assertEqual(len(transcribed_data), len(expected_output))
        for i in range(len(transcribed_data)):
            self.assertEqual(transcribed_data[i].word, expected_output[i].word)
            self.assertEqual(transcribed_data[i].end, expected_output[i].end)
            self.assertEqual(transcribed_data[i].start, expected_output[i].start)
            self.assertEqual(transcribed_data[i].is_hyphen, expected_output[i].is_hyphen)

    def test_number_to_words_converts(self):
        #Original, test with no language passed
        self.act_and_assert("I have 1 million dollars and 2 cents.", "I have one million dollars and two cents.")
        self.act_and_assert("1 2 3 4 5", "one two three four five")
        self.act_and_assert("1, 2, 3, 4, 5,", "one, two, three, four, five,")
        self.act_and_assert("Hello world 1, 2!. 3. 4? Test 100#",
                            "Hello world one, two!. three. four? Test one hundred#")
        #Test English
        self.act_and_assert("I have 1 million dollars and 2 cents.", "I have one million dollars and two cents.", "en")
        self.act_and_assert("1 2 3 4 5", "one two three four five", "en")
        self.act_and_assert("1, 2, 3, 4, 5,", "one, two, three, four, five,", "en")
        self.act_and_assert("Hello world 1, 2!. 3. 4? Test 100#",
                            "Hello world one, two!. three. four? Test one hundred#", "en")
        #Test German
        self.act_and_assert("1 2 3 4 5", "eins zwei drei vier fünf" ,"de")
        self.act_and_assert("1, 2, 3, 4, 5","eins, zwei, drei, vier, fünf","de")
        self.act_and_assert("Ich habe 1 Million Dollar und 2 Cent.","Ich habe eins Million Dollar und zwei Cent.","de")
        self.act_and_assert("Hallo Welt 1, 2!. 3. 4? Test 100#","Hallo Welt eins, zwei!. drei. vier? Test einhundert#","de")
        #Test Spanish
        self.act_and_assert("1 2 3 4 5","uno dos tres cuatro cinco","es")
        self.act_and_assert("1, 2, 3, 4, 5","uno, dos, tres, cuatro, cinco","es")
        self.act_and_assert("Tengo un millón de dólares y 2 centavos","Tengo un millón de dólares y dos centavos","es")
        self.act_and_assert("Hola mundo 1, 2!. 3. 4? Prueba 100#","Hola mundo uno, dos!. tres. cuatro? Prueba cien#","es")
        

    def act_and_assert(self, text, expected_output, language="en"):
        # Act
        result = number_to_words(text, language)

        # Assert
        self.assertEqual(result, expected_output)
if __name__ == "__main__":
    unittest.main()
