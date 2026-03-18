"""LLM-based lyric correction for WhisperX transcriptions.

Uses an OpenAI-compatible chat completions API to post-correct
speech-to-text output. The LLM sees full sentences for context
but must return exactly the same number of words — only fixing
misheard or misspelled words while preserving all timing data.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass

from modules.console_colors import ULTRASINGER_HEAD, red_highlighted, blue_highlighted
from modules.Speech_Recognition.TranscribedData import TranscribedData

# Maximum gap in seconds between words before starting a new sentence group
_SENTENCE_GAP_S = 1.5
# Maximum words per LLM request
_MAX_WORDS_PER_CHUNK = 40
# Request timeout in seconds
_REQUEST_TIMEOUT_S = 30

_SYSTEM_PROMPT = """\
You are a lyrics correction assistant. You receive speech-to-text output \
from a song and must fix misheard or misspelled words.

RULES:
1. Return EXACTLY the same number of words as the input
2. Only correct words that are clearly wrong — do not change correct words
3. Return one word per line, nothing else
4. Do not add punctuation, explanations, or numbering
5. Preserve the original language"""


@dataclass
class LLMConfig:
    """Configuration for LLM lyric correction."""
    api_base_url: str
    api_key: str
    model: str
    language: str | None = None
    artist: str | None = None
    title: str | None = None


def correct_lyrics_with_llm(
    transcribed_data: list[TranscribedData],
    config: LLMConfig,
) -> list[TranscribedData]:
    """Post-correct WhisperX transcription using an LLM.

    Groups words into sentence-like chunks, sends each to the LLM with a
    strict "return same number of words" constraint, and replaces ``.word``
    fields while preserving all timing data.

    Returns the (possibly modified) *transcribed_data* list.
    """
    if not transcribed_data:
        return transcribed_data

    if not config.api_key:
        print(f"{ULTRASINGER_HEAD} {red_highlighted('LLM correction skipped: no API key configured')}")
        return transcribed_data

    chunks = _build_chunks(transcribed_data)
    total_corrected = 0

    for start_idx, end_idx in chunks:
        words = [td.word.strip() for td in transcribed_data[start_idx:end_idx]]
        if not words:
            continue

        prompt = _build_user_prompt(words, config.language, config.artist, config.title)

        try:
            response_text = _call_llm_api(prompt, config)
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"{ULTRASINGER_HEAD} {red_highlighted(f'LLM network error: {e}')}")
            continue
        except (json.JSONDecodeError, KeyError) as e:
            print(f"{ULTRASINGER_HEAD} {red_highlighted(f'LLM response parse error: {e}')}")
            continue
        except Exception as e:
            print(f"{ULTRASINGER_HEAD} {red_highlighted(f'LLM API error: {e}')}")
            continue

        corrected = _parse_response(response_text, len(words))
        if corrected is None:
            continue

        n_changed = _apply_corrections(transcribed_data, start_idx, end_idx, corrected)
        total_corrected += n_changed

    if total_corrected > 0:
        print(f"{ULTRASINGER_HEAD} LLM corrected {blue_highlighted(str(total_corrected))} word(s)")
    else:
        print(f"{ULTRASINGER_HEAD} LLM found no corrections needed")

    return transcribed_data


def _build_chunks(
    transcribed_data: list[TranscribedData],
) -> list[tuple[int, int]]:
    """Group transcribed words into sentence-like chunks.

    Splits on timing gaps > ``_SENTENCE_GAP_S`` and limits chunk size
    to ``_MAX_WORDS_PER_CHUNK``.

    Returns a list of ``(start_idx, end_idx)`` pairs (half-open).
    """
    if not transcribed_data:
        return []

    chunks: list[tuple[int, int]] = []
    chunk_start = 0

    for i in range(1, len(transcribed_data)):
        gap = transcribed_data[i].start - transcribed_data[i - 1].end
        chunk_len = i - chunk_start

        if gap > _SENTENCE_GAP_S or chunk_len >= _MAX_WORDS_PER_CHUNK:
            chunks.append((chunk_start, i))
            chunk_start = i

    # Last chunk
    if chunk_start < len(transcribed_data):
        chunks.append((chunk_start, len(transcribed_data)))

    return chunks


def _build_user_prompt(
    words: list[str],
    language: str | None,
    artist: str | None,
    title: str | None,
) -> str:
    """Build the user prompt for the LLM."""
    parts: list[str] = []
    if language:
        parts.append(f"Language: {language}")
    if artist:
        parts.append(f"Artist: {artist}")
    if title:
        parts.append(f"Title: {title}")
    parts.append(f"Word count: {len(words)}")
    parts.append("")
    parts.append("Correct these lyrics (return one word per line):")
    parts.extend(words)
    return "\n".join(parts)


def _call_llm_api(user_prompt: str, config: LLMConfig) -> str:
    """Call an OpenAI-compatible chat completions endpoint.

    Uses ``urllib.request`` (stdlib) to avoid adding dependencies.
    """
    url = config.api_base_url.rstrip("/") + "/chat/completions"

    # Validate URL scheme to prevent SSRF
    if not url.startswith(("https://", "http://")):
        raise ValueError(f"Invalid URL scheme: {url} (only http/https allowed)")

    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.api_key}",
        "User-Agent": "Mozilla/5.0 (compatible; UltraSinger/1.0)",
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")

    with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_S) as resp:
        body = json.loads(resp.read().decode("utf-8"))

    return body["choices"][0]["message"]["content"]


def _parse_response(response_text: str, expected_count: int) -> list[str] | None:
    """Parse the LLM response into a list of words.

    Returns ``None`` if the word count does not match *expected_count*.
    """
    # Strip empty lines and whitespace
    lines = [line.strip() for line in response_text.strip().splitlines() if line.strip()]

    if len(lines) != expected_count:
        print(
            f"{ULTRASINGER_HEAD} {red_highlighted(f'LLM word count mismatch: expected {expected_count}, got {len(lines)} — skipping chunk')}"
        )
        return None

    return lines


def _apply_corrections(
    transcribed_data: list[TranscribedData],
    start_idx: int,
    end_idx: int,
    corrected_words: list[str],
) -> int:
    """Replace word fields in-place, preserving trailing whitespace.

    Returns the number of words actually changed.
    """
    changed = 0
    for i, new_word in enumerate(corrected_words):
        td = transcribed_data[start_idx + i]
        old_stripped = td.word.strip()

        if new_word.lower() != old_stripped.lower():
            # Preserve trailing whitespace convention
            trailing = td.word[len(td.word.rstrip()):]
            td.word = new_word + trailing
            changed += 1

    return changed
