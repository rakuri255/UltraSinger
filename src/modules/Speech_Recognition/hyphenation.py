"""Hyphenation module"""

import string

from hyphen import Hyphenator, dictools
from tqdm import tqdm

from modules.Speech_Recognition.TranscribedData import TranscribedData
from modules.console_colors import (
    ULTRASINGER_HEAD,
    blue_highlighted, red_highlighted,
)

# Mapeo obligatorio a diccionarios regionales estrictos (prohíbe 'es', 'en', etc.)
STRICT_REGIONAL_MAP = {
    "es": "es_ES",
    "en": "en_US",
    "de": "de_DE",
    "fr": "fr_FR",
    "it": "it_IT",
    "pt": "pt_PT",
    "nl": "nl_NL",
    "pl": "pl_PL",
    "ru": "ru_RU",
}


def language_check(language="en") -> str | None:
    """Check if language is supported"""
    clean_lang = (language or "en").lower().strip().replace("-", "_")
    base_lang = clean_lang.split("_")[0]
    if base_lang == "es":
        print(f"{ULTRASINGER_HEAD} Hyphenate using language code: {blue_highlighted('es_ES')}")
        return "es_ES"
    if base_lang == "en":
        print(f"{ULTRASINGER_HEAD} Hyphenate using language code: {blue_highlighted('en_US')}")
        return "en_US"

    lang_region = None
    installed = dictools.list_installed()
    installed_region_keys = [i for i in installed if i.startswith(base_lang) and "_" in i]
    try:
        # Try to find installed language with region prediction
        lang_region = next(i for i in installed_region_keys if i == f"{base_lang}_{base_lang.upper()}")
    except StopIteration:
            if installed_region_keys:
                # Take first installed region language
                lang_region = installed_region_keys[0]
            else:
                # Take downloadable language key
                downloadable_key = [i for i in dictools.LANGUAGES if i.startswith(base_lang)]
                downloadable_folder_key = [i for i in downloadable_key if i == base_lang]
                if downloadable_folder_key:
                    lang_region = downloadable_key[0]
                else:
                    try:
                        # Try to find downloadable language with region prediction
                        lang_region = next(i for i in downloadable_key if i == f"{base_lang}_{base_lang.upper()}")
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


def __clean_word(word: str):
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


def __insert_removed_symbols(separated_array, removed_indices, symbols):
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


def __create_hyphenator(lang_region: str) -> Hyphenator:
    """Create hyphenator"""
    hyphenator = Hyphenator(lang_region)
    return hyphenator


def _postprocess_spanish_syllables(syllables: list[str], raw_clean_word: str) -> list[str]:
    """Apply Spanish RAE phonetic syllabification rules:
    1. Split 1-letter initial vowel V-CV ('Ella' -> 'E-lla', 'una' -> 'u-na', 'Iluminaba' -> 'I-lu-mi-na-ba').
    2. Split Spanish hiatuses (accented weak vowel í/ú + strong vowel a/e/o -> 'sen-tí-a', 'sa-bí-a', 'dí-a').
    """
    import re
    base_list = syllables if syllables else [raw_clean_word]

    # Step 1: Split initial V-CV on the first syllable if pyhyphen left it merged
    if base_list:
        first_syl = base_list[0]
        m_init = re.match(
            r"^([aeiouáéíóúAEIOUÁÉÍÓÚ])((?:ll|rr|ch|[pbcdfgtr][rl]|[bcdfghjklmnñpqrstvwxyz])[aeiouáéíóúáéíóúüAEIOUÁÉÍÓÚÜ].*)$",
            first_syl,
        )
        if m_init:
            base_list = [m_init.group(1), m_init.group(2)] + base_list[1:]

    # Step 2: Split hiatuses inside each syllable (e.g. 'tía' -> 'tí', 'a'; 'bía' -> 'bí', 'a'; 'caer' -> 'ca', 'er')
    hiatus_pattern = re.compile(
        r"([íúÍÚ])([aeoáéóAEOÁÉÓ])|([aeoáéóAEOÁÉÓ])([íúÍÚ])|([aeoáéóAEOÁÉÓ])([aeoáéóAEOÁÉÓ])"
    )
    refined: list[str] = []
    for syl in base_list:
        curr = syl
        while True:
            m = hiatus_pattern.search(curr)
            if not m:
                break
            split_pos = m.start() + 1
            refined.append(curr[:split_pos])
            curr = curr[split_pos:]
        if curr:
            refined.append(curr)

    return refined


def _postprocess_english_syllables(syllables: list[str], raw_clean_word: str, original_word: str) -> list[str]:
    """Prevent English silent-e splitting ('ta-ke' -> 'take', 'mo-ve' -> 'move', 'smi-le' -> 'smile',
    'Sin-ce' -> 'Since', 'em-bra-ce' -> 'em-brace') and keep contractions ('you've', 'can't') intact.
    """
    import re
    if "'" in original_word or "’" in original_word:
        # Never split English contractions across the apostrophe suffix
        stem = original_word.strip().split("'")[0].split("’")[0]
        if len(re.findall(r"[aeiouyAEIOUY]+", stem)) <= 1:
            return [raw_clean_word]

    if not syllables or len(syllables) <= 1:
        return syllables or [raw_clean_word]

    out = list(syllables)
    # Merge trailing silent-e syllable (e.g. 'ke', 've', 'ne', 'ce', 'se', 'ze', 'te', 'de', 'me', 'pe', 'fe', 'le' after vowel)
    if len(out) >= 2:
        last_s = out[-1].lower()
        prev_s = out[-2].lower()
        # Syllabic -le after a consonant (e.g. 'lit-tle', 'ta-ble') stays 2 syllables; after vowel ('smi-le') merges
        if re.fullmatch(r"l+es?", last_s):
            if prev_s and prev_s[-1] in "aeiouy":
                out[-2] = out[-2] + out[-1]
                out.pop()
        elif re.fullmatch(r"(?:[bcdfghjkmnpqrstvwxz]|th|sh|ch|ph)e[sd]?", last_s):
            out[-2] = out[-2] + out[-1]
            out.pop()

    return out


def hyphenation(word: str, hyphenator: Hyphenator, lang_region: str | None = None) -> list[str] | None:
    """Hyphenate word.

    Spanish uses the deterministic RAE syllabifier (pyhyphen leaves VCV groups such as
    'cer-cano' unsplit). English combines pyhyphen with a phonetic/singing-lexicon
    syllabifier so short words ('Ev-ery', 'on-ly', 'a-round', 'Ba-by') are split.
    """
    from modules.Speech_Recognition.syllabifier import syllabify_es, syllabify_en

    if lang_region is None:
        lang_region = str(getattr(hyphenator, "language", "") or "")

    cleaned_string, removed_indices, removed_symbols = __clean_word(word)

    # Hyphenation of word longer than 100 characters throws exception
    if len(cleaned_string) > 100 or not cleaned_string:
        return None

    lang_low = lang_region.lower()
    if lang_low.startswith("es"):
        syllabus = syllabify_es(cleaned_string)
        if "".join(syllabus) != cleaned_string:
            syllabus = _postprocess_spanish_syllables(hyphenator.syllables(cleaned_string), cleaned_string)
    elif lang_low.startswith("en"):
        typographic = _postprocess_english_syllables(hyphenator.syllables(cleaned_string), cleaned_string, word)
        if "'" in word or "’" in word:
            syllabus = typographic
        else:
            syllabus = syllabify_en(cleaned_string, typographic=typographic)
    else:
        syllabus = hyphenator.syllables(cleaned_string)

    length = len(syllabus)
    if length > 1:
        hyphen = []
        for i in range(length):
            hyphen.append(syllabus[i])
        hyphen = __insert_removed_symbols(hyphen, removed_indices, removed_symbols)
    else:
        hyphen = None

    return hyphen


def hyphenate_each_word(
        language: str, transcribed_data: list[TranscribedData]
) -> list[list[str]] | None:
    """Hyphenate each word in the transcribed data."""
    lang_region = language_check(language)
    if lang_region is None:
        print(
            f"{ULTRASINGER_HEAD} {red_highlighted('Error in hyphenation for language ')} {blue_highlighted(language)}{red_highlighted(', maybe you want to disable it?')}"
        )
        return None

    hyphenated_word = []
    try:
        hyphenator = __create_hyphenator(lang_region)
        for i in tqdm(enumerate(transcribed_data)):
            pos = i[0]
            hyphenated_word.append(
                hyphenation(transcribed_data[pos].word, hyphenator, lang_region=lang_region)
            )
    except Exception as e:
        print(
            f"{ULTRASINGER_HEAD} {red_highlighted('Error in hyphenation for language ')} {blue_highlighted(language)}{red_highlighted(', maybe you want to disable it?')}")
        print(f"\t{red_highlighted(f'->{e}')}")
        return None

    return hyphenated_word
