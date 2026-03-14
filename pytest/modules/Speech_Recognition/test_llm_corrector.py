"""Tests for LLM-based lyric correction."""

import json
import unittest
from unittest.mock import patch, MagicMock

from src.modules.Speech_Recognition.TranscribedData import TranscribedData
from src.modules.Speech_Recognition.llm_corrector import (
    LLMConfig,
    _build_chunks,
    _build_user_prompt,
    _parse_response,
    _apply_corrections,
    correct_lyrics_with_llm,
)


def _td(word: str, start: float, end: float, confidence: float = 0.9) -> TranscribedData:
    """Helper to create TranscribedData with minimal args."""
    return TranscribedData(word=word, start=start, end=end, confidence=confidence)


def _config(**overrides) -> LLMConfig:
    """Helper to create an LLMConfig with defaults."""
    defaults = {
        "api_base_url": "https://api.example.com/v1",
        "api_key": "test-key",
        "model": "test-model",
        "language": "en",
        "artist": "Test Artist",
        "title": "Test Song",
    }
    defaults.update(overrides)
    return LLMConfig(**defaults)


class TestBuildChunks(unittest.TestCase):
    def test_empty_input(self):
        self.assertEqual(_build_chunks([]), [])

    def test_single_word(self):
        data = [_td("hello", 0.0, 0.5)]
        self.assertEqual(_build_chunks(data), [(0, 1)])

    def test_contiguous_words_single_chunk(self):
        data = [_td("hello", 0.0, 0.5), _td("world", 0.6, 1.0)]
        self.assertEqual(_build_chunks(data), [(0, 2)])

    def test_gap_splits_chunks(self):
        data = [
            _td("hello", 0.0, 0.5),
            _td("world", 0.6, 1.0),
            _td("foo", 3.0, 3.5),  # 2.0s gap > 1.5s threshold
        ]
        chunks = _build_chunks(data)
        self.assertEqual(chunks, [(0, 2), (2, 3)])

    def test_max_words_per_chunk(self):
        """Chunks exceeding _MAX_WORDS_PER_CHUNK are split."""
        data = [_td(f"w{i}", float(i), float(i) + 0.1) for i in range(50)]
        chunks = _build_chunks(data)
        # First chunk should be 40 words, second 10
        self.assertEqual(chunks[0], (0, 40))
        self.assertEqual(chunks[1], (40, 50))


class TestBuildUserPrompt(unittest.TestCase):
    def test_includes_metadata(self):
        prompt = _build_user_prompt(["hello", "world"], "en", "Artist", "Song")
        self.assertIn("Language: en", prompt)
        self.assertIn("Artist: Artist", prompt)
        self.assertIn("Title: Song", prompt)
        self.assertIn("Word count: 2", prompt)
        self.assertIn("hello", prompt)
        self.assertIn("world", prompt)

    def test_no_metadata(self):
        prompt = _build_user_prompt(["hello"], None, None, None)
        self.assertNotIn("Language:", prompt)
        self.assertNotIn("Artist:", prompt)
        self.assertNotIn("Title:", prompt)
        self.assertIn("Word count: 1", prompt)
        self.assertIn("hello", prompt)


class TestParseResponse(unittest.TestCase):
    def test_correct_count(self):
        response = "hello\nworld\n"
        result = _parse_response(response, 2)
        self.assertEqual(result, ["hello", "world"])

    def test_count_mismatch_returns_none(self):
        response = "hello\nworld\nextra\n"
        result = _parse_response(response, 2)
        self.assertIsNone(result)

    def test_empty_lines_ignored(self):
        response = "\nhello\n\nworld\n\n"
        result = _parse_response(response, 2)
        self.assertEqual(result, ["hello", "world"])

    def test_whitespace_stripped(self):
        response = "  hello  \n  world  \n"
        result = _parse_response(response, 2)
        self.assertEqual(result, ["hello", "world"])


class TestApplyCorrections(unittest.TestCase):
    def test_replaces_changed_words(self):
        data = [_td("helo ", 0.0, 0.5), _td("wrld ", 0.6, 1.0)]
        changed = _apply_corrections(data, 0, 2, ["hello", "world"])
        self.assertEqual(changed, 2)
        self.assertEqual(data[0].word, "hello ")
        self.assertEqual(data[1].word, "world ")

    def test_preserves_trailing_whitespace(self):
        data = [_td("helo ", 0.0, 0.5)]
        _apply_corrections(data, 0, 1, ["hello"])
        self.assertEqual(data[0].word, "hello ")

    def test_no_trailing_whitespace_preserved(self):
        data = [_td("helo", 0.0, 0.5)]
        _apply_corrections(data, 0, 1, ["hello"])
        self.assertEqual(data[0].word, "hello")

    def test_same_word_not_counted_as_change(self):
        data = [_td("hello ", 0.0, 0.5)]
        changed = _apply_corrections(data, 0, 1, ["hello"])
        self.assertEqual(changed, 0)

    def test_case_insensitive_comparison(self):
        data = [_td("Hello ", 0.0, 0.5)]
        changed = _apply_corrections(data, 0, 1, ["hello"])
        self.assertEqual(changed, 0)

    def test_partial_range(self):
        data = [
            _td("keep", 0.0, 0.5),
            _td("helo", 0.6, 1.0),
            _td("wrld", 1.1, 1.5),
            _td("keep", 1.6, 2.0),
        ]
        changed = _apply_corrections(data, 1, 3, ["hello", "world"])
        self.assertEqual(changed, 2)
        self.assertEqual(data[0].word, "keep")  # untouched
        self.assertEqual(data[1].word, "hello")
        self.assertEqual(data[2].word, "world")
        self.assertEqual(data[3].word, "keep")  # untouched

    def test_timing_preserved(self):
        data = [_td("helo ", 1.23, 4.56)]
        _apply_corrections(data, 0, 1, ["hello"])
        self.assertAlmostEqual(data[0].start, 1.23)
        self.assertAlmostEqual(data[0].end, 4.56)


class TestCorrectLyricsWithLLM(unittest.TestCase):
    def test_empty_data_returns_unchanged(self):
        result = correct_lyrics_with_llm([], _config())
        self.assertEqual(result, [])

    def test_no_api_key_returns_unchanged(self):
        data = [_td("hello", 0.0, 0.5)]
        result = correct_lyrics_with_llm(data, _config(api_key=""))
        self.assertEqual(result[0].word, "hello")

    @patch("src.modules.Speech_Recognition.llm_corrector._call_llm_api")
    def test_successful_correction(self, mock_api):
        mock_api.return_value = "hello\nworld\n"
        data = [_td("helo ", 0.0, 0.5), _td("wrld ", 0.6, 1.0)]
        result = correct_lyrics_with_llm(data, _config())
        self.assertEqual(result[0].word, "hello ")
        self.assertEqual(result[1].word, "world ")

    @patch("src.modules.Speech_Recognition.llm_corrector._call_llm_api")
    def test_api_error_returns_unchanged(self, mock_api):
        mock_api.side_effect = Exception("Connection refused")
        data = [_td("helo ", 0.0, 0.5)]
        result = correct_lyrics_with_llm(data, _config())
        self.assertEqual(result[0].word, "helo ")

    @patch("src.modules.Speech_Recognition.llm_corrector._call_llm_api")
    def test_word_count_mismatch_skips_chunk(self, mock_api):
        mock_api.return_value = "hello\nworld\nextra\n"
        data = [_td("helo ", 0.0, 0.5), _td("wrld ", 0.6, 1.0)]
        result = correct_lyrics_with_llm(data, _config())
        # Should be unchanged because word count mismatched
        self.assertEqual(result[0].word, "helo ")
        self.assertEqual(result[1].word, "wrld ")

    @patch("src.modules.Speech_Recognition.llm_corrector._call_llm_api")
    def test_no_changes_needed(self, mock_api):
        mock_api.return_value = "hello\nworld\n"
        data = [_td("hello ", 0.0, 0.5), _td("world ", 0.6, 1.0)]
        result = correct_lyrics_with_llm(data, _config())
        self.assertEqual(result[0].word, "hello ")
        self.assertEqual(result[1].word, "world ")


if __name__ == "__main__":
    unittest.main()
