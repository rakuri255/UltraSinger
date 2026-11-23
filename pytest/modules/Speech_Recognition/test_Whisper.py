"""Tests for whisper.py"""

import unittest
from src.modules.Speech_Recognition.TranscribedData import TranscribedData
import numpy as np
from src.modules.Speech_Recognition.Whisper import (
    convert_to_transcribed_data,
    number_to_words,
    detect_language_on_vocals,
)

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


class _FakeCTranslateModel:
    def __init__(self, results):
        self.results = list(results)
        self.calls = 0

    def detect_language(self, encoder_output):
        result = self.results[self.calls]
        self.calls += 1
        return [result]


class _FakeWhisperModel:
    """Mimics the parts of whisperx's FasterWhisperPipeline used for language detection"""

    def __init__(self, results):
        self.feat_kwargs = {"feature_size": 80}
        self.model = _FakeCTranslateModel(results)

    def encode(self, segment):
        return segment


class _FakePipeline:
    def __init__(self, results):
        self.model = _FakeWhisperModel(results)


class DetectLanguageOnVocalsTest(unittest.TestCase):
    sample_rate = 16000

    def _audio(self, silence_seconds, voiced_seconds):
        rng = np.random.default_rng(0)
        silence = np.zeros(silence_seconds * self.sample_rate, dtype=np.float32)
        voiced = (rng.standard_normal(voiced_seconds * self.sample_rate) * 0.1).astype(np.float32)
        return np.concatenate([silence, voiced])

    def test_sums_language_probabilities_of_vocal_windows(self):
        # 20 s muted intro + 70 s of "singing" -> 2 windows of 30 s singing
        pipeline = _FakePipeline([
            [("<|nn|>", 0.6), ("<|en|>", 0.4)],
            [("<|en|>", 0.9), ("<|nn|>", 0.1)],
        ])

        language = detect_language_on_vocals(pipeline, self._audio(20, 70))

        self.assertEqual("en", language)
        self.assertEqual(2, pipeline.model.model.calls)

    def test_returns_none_with_too_little_singing(self):
        pipeline = _FakePipeline([])

        self.assertIsNone(detect_language_on_vocals(pipeline, self._audio(30, 2)))
        self.assertEqual(0, pipeline.model.model.calls)


if __name__ == "__main__":
    unittest.main()
