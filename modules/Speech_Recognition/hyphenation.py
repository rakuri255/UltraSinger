"""Docstring"""

import string

from hyphen import Hyphenator, dictools

from modules.Log import (
    PRINT_ULTRASTAR,
    print_blue_highlighted_text,
)


def language_check(language="en"):
    """Docstring"""

    dict_langs = dictools.LANGUAGES

    lang_region = None
    for dict_lang in dict_langs:
        if dict_lang.startswith(language):
            lang_region = dict_lang
            break

    if lang_region is None:
        return None

    print(
        f"{PRINT_ULTRASTAR} Hyphenate using language code: {print_blue_highlighted_text(lang_region)}"
    )
    return lang_region


def contains_punctuation(word):
    """Docstring"""

    return any(elem in word for elem in string.punctuation)


def hyphenation(word, lang_region):
    """Docstring"""

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
