import pyphen


def hyphenation(word, language='en'):
    dic = pyphen.Pyphen(lang=language)

    hyphen = None
    for pair in dic.iterate(word):
        hyphen = pair  # Todo: ignore the others?

    return hyphen
