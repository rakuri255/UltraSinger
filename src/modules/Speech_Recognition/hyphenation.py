"""Hyphenation module"""

import string

from hyphen import Hyphenator, dictools

from src.modules.console_colors import (
    ULTRASINGER_HEAD,
    blue_highlighted,
)


def language_check(language="en") -> str | None:
    """Check if language is supported"""

    dict_langs = dictools.LANGUAGES

    lang_region = None
    for dict_lang in dict_langs:
        if dict_lang.startswith(language):
            lang_region = dict_lang
            break

    if lang_region is None:
        return None

    print(
        f"{ULTRASINGER_HEAD} Hyphenate using language code: {blue_highlighted(lang_region)}"
    )
    return lang_region


def contains_punctuation(word: str) -> bool:
    """Check if word contains punctuation"""

    return any(elem in word for elem in string.punctuation)


def clean_word(word: str):
    """Remove punctuation from word"""
    cleaned_string = ""
    removed_indices = []
    removed_symbols = []
    for i, char in enumerate(word):
        if char not in string.punctuation and char not in " ":
            cleaned_string += char
        else:
            removed_indices.append(i)
            removed_symbols.append(char)
    return cleaned_string, removed_indices, removed_symbols


def insert_removed_symbols(separated_array, removed_indices, symbols):
    """Insert symbols into the syllables"""
    result = []
    symbol_index = 0
    i = 0

    # Add removed symbols to the syllables
    for syllable in separated_array:
        tmp = ""
        for char in syllable:
            if i in removed_indices:
                tmp += symbols[symbol_index]
                symbol_index += 1
                i += 1
            tmp += char
            i += 1
        result.append(tmp)

    # Add remaining symbols to the last syllable
    if symbol_index < len(symbols):
        tmp = result[-1]
        for i in range(symbol_index, len(symbols)):
            tmp += symbols[i]
        result[-1] = tmp

    return result


def create_hyphenator(lang_region: str) -> Hyphenator:
    """Create hyphenator"""
    hyphenator = Hyphenator(lang_region)
    return hyphenator


def hyphenation(word: str, hyphenator: Hyphenator) -> list[str] | None:
    """Hyphenate word"""

    cleaned_string, removed_indices, removed_symbols = clean_word(word)
    syllabus = hyphenator.syllables(cleaned_string)

    length = len(syllabus)
    if length > 1:
        hyphen = []
        for i in range(length):
            hyphen.append(syllabus[i])
        hyphen = insert_removed_symbols(hyphen, removed_indices, removed_symbols)
    else:
        hyphen = None

    return hyphen
