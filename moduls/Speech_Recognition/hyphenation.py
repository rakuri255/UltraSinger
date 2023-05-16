from hyphen import Hyphenator
from hyphen import dictools
from moduls.Log import PRINT_ULTRASTAR
from moduls.Log import print_blue_highlighted_text
import string


def language_check(language='en'):
    dict_langs = dictools.LANGUAGES

    lang_region = None
    for dict_lang in dict_langs:
        if dict_lang.startswith(language):
            lang_region = dict_lang
            break

    if lang_region is None:
        raise Exception('{} Missing language "{}" for hyphen. Install it or disable hyphen.'.format(PRINT_ULTRASTAR, language))

    print("{} Hyphenate using language code: {}".format(PRINT_ULTRASTAR, print_blue_highlighted_text(lang_region)))
    return lang_region


def contains_punctuation(word):
    return any(elem in word for elem in string.punctuation)


def hyphenation(word, lang_region):
    if contains_punctuation(word):
        return None

    hy = Hyphenator(lang_region)
    syllabus = hy.syllables(word)

    length = len(syllabus)
    if length > 1:
        hyphen = []
        for i in range(length):
            hyphen.append(syllabus[i])
    else:
        hyphen = None

    return hyphen
