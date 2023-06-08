"""Hyphenation module"""

import string

from hyphen import Hyphenator, dictools

from modules.console_colors import (
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


def hyphenation(word: str, lang_region: str) -> list[str] | None:
    """Hyphenate word"""

    if contains_punctuation(word):
        return None

    hyphenator = Hyphenator(lang_region)
    syllabus = hyphenator.syllables(word)

    length = len(syllabus)
    if length > 1:
        hyphen = []
        for i in range(length):
            hyphen.append(syllabus[i])
    else:
        hyphen = None

    return hyphen
