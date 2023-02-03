import pyphen


def get_hyphenate(language, word):
    pyphen.language_fallback('en')

    # todo: do we need fail check?
    # if 'nl_NL' not in pyphen.LANGUAGES:
    #    error

    dic = pyphen.Pyphen(lang=language)

    hyphenate = tuple
    for pair in dic.iterate(word):
        hyphenate = pair  # Todo: ignore the others?

    print(hyphenate)
    return hyphenate
