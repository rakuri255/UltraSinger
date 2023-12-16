"""Hyphenation module"""

import string

from hyphen import Hyphenator, dictools

from modules.console_colors import (
    ULTRASINGER_HEAD,
    blue_highlighted,
)

# PyHyphen tries to retrieve dictionaries for download 'https://cgit.freedesktop.org/libreoffice/dictionaries/plain/'
# Updated PyHyphen dictools Languages, so they can be installed
LANGUAGES = [
"af_ZA",
"an_ES",
"ar",
"be_BY",
"bg_BG",
"bn_BD",
"bo",
"br_FR",
"bs_BA",
"ca",
"ckb",
"cs_CZ",
"da_DK",
"de",
"el_GR",
"en",
"eo",
"es",
"et_EE",
"fa_IR",
"fr_FR",
"gd_GB",
"gl",
"gu_IN",
"gug",
"he_IL",
"hi_IN",
"hr_HR",
"hu_HU",
"id",
"is",
"it_IT",
"kmr_Latn",
"ko_KR",
"lo_LA",
"lt_LT",
"lv_LV",
"mn_MN",
"ne_NP",
"nl_NL",
"no",
"oc_FR",
"pl_PL",
"pt_BR",
"pt_PT",
"ro",
"ru_RU",
"si_LK",
"sk_SK",
"sl_SI",
"sq_AL",
"sr",
"sv_SE",
"sw_TZ",
"te_IN",
"th_TH",
"tr_TR",
"uk_UA",
"vi",
"zu_ZA",
]

def language_check(language="en") -> str | None:
    """Check if language is supported"""

    lang_region = None
    installed = dictools.list_installed()
    installed_region_keys = [i for i in installed if i.startswith(language) and "_" in i]
    try:
        # Try to find installed language with region prediction
        lang_region = next(i for i in installed_region_keys if i == f"{language}_{language.upper()}")
    except StopIteration:
        if installed_region_keys:
            # Take first installed region language
            lang_region = installed_region_keys[0]
        else:
            # Take downloadable language key
            downloadable_key = [i for i in LANGUAGES if i.startswith(language)]
            downloadable_folder_key = [i for i in downloadable_key if i == language]
            if downloadable_folder_key:
                lang_region = downloadable_key[0]
            else:
                try:
                    # Try to find downloadable language with region prediction
                    lang_region = next(i for i in downloadable_key if i == f"{language}_{language.upper()}")
                except StopIteration:
                    if downloadable_key:
                        # Take first installed region language
                        lang_region = downloadable_key[0]

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

    # Hyphenation of word longer than 100 characters throws exception
    if len(cleaned_string) > 100:
        return None

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
