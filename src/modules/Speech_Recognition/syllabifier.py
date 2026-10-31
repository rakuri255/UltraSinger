"""Deterministic phonological syllabifiers for sung lyrics (Spanish RAE / English).

Typographic hyphenation dictionaries (pyhyphen / LibreOffice) are designed for line
breaking and refuse to split short words ('every', 'baby', 'only') or leave VCV groups
unsplit ('cer-cano'). Karaoke files need *sung* syllables, so this module provides:

- ``syllabify_es``: RAE rules (diphthongs, hiatus, inseparable onset clusters, digraphs,
  silent 'u' in que/qui/gue/gui, final 'y' as vowel), maximal onset principle.
- ``syllabify_en``: singing lexicon override + vowel-nucleus rule syllabifier with silent-e
  handling, optionally validated by CMUdict syllable counts (``pronouncing`` / ``nltk``).

All functions return slices of the input string (case preserved) whose concatenation
equals the input.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import List, Optional

# ----------------------------------------------------------------------------------------
# Spanish
# ----------------------------------------------------------------------------------------

ES_STRONG = set("aeoáéóÁÉÓAEO")
ES_ACCENTED_WEAK = set("íúÍÚ")
ES_WEAK = set("iuüIUÜ")
ES_VOWELS = ES_STRONG | ES_ACCENTED_WEAK | ES_WEAK
ES_INSEPARABLE = {"pr", "pl", "br", "bl", "fr", "fl", "tr", "dr", "cr", "cl", "gr", "gl", "kr", "kl"}
ES_DIGRAPHS = ("ch", "ll", "rr")


def _es_is_hiatus(v1: str, v2: str) -> bool:
    """Two adjacent Spanish vowels form a hiatus (belong to different syllables)."""
    s1 = v1 in ES_STRONG
    s2 = v2 in ES_STRONG
    if s1 and s2:
        return True
    if v1 in ES_ACCENTED_WEAK and s2:
        return True
    if v2 in ES_ACCENTED_WEAK and s1:
        return True
    return False


def _es_vowel_mask(word: str) -> List[bool]:
    """Mark which characters act as vowels (nucleus candidates)."""
    low = word.lower()
    n = len(low)
    mask = [c in ES_VOWELS for c in word]
    for i, c in enumerate(low):
        # silent 'u' in que, qui, gue, gui (not ü)
        if c == "u" and i > 0 and i + 1 < n:
            prev, nxt = low[i - 1], low[i + 1]
            if prev == "q" or (prev == "g" and nxt in "eéií"):
                mask[i] = False
        elif c == "u" and i > 0 and low[i - 1] == "q":
            mask[i] = False
        if c == "y":
            # 'y' is vocalic when alone or word-final after a vowel ('hay', 'rey', 'muy')
            if n == 1 or (i == n - 1 and i > 0 and mask[i - 1]):
                mask[i] = True
    return mask


def _es_consonant_units(cluster: str) -> List[str]:
    """Split a consonant cluster into units, keeping digraphs ch/ll/rr and qu/gu (silent u) together."""
    units: List[str] = []
    i = 0
    low = cluster.lower()
    while i < len(cluster):
        two = low[i:i + 2]
        if two in ES_DIGRAPHS or two in ("qu", "gu"):
            units.append(cluster[i:i + 2])
            i += 2
        else:
            units.append(cluster[i])
            i += 1
    return units


@lru_cache(maxsize=20000)
def _syllabify_es_cached(word: str) -> tuple:
    if not word:
        return tuple()
    mask = _es_vowel_mask(word)
    n = len(word)
    # Group nuclei: consecutive vowel chars, splitting at hiatus
    nuclei: List[List[int]] = []  # each nucleus = [start, end) char indices
    i = 0
    while i < n:
        if mask[i]:
            start = i
            j = i + 1
            while j < n and mask[j] and not _es_is_hiatus(word[j - 1], word[j]):
                j += 1
            nuclei.append([start, j])
            i = j
        else:
            i += 1
    if len(nuclei) <= 1:
        return (word,)

    cut_points: List[int] = []
    for k in range(len(nuclei) - 1):
        a_end = nuclei[k][1]
        b_start = nuclei[k + 1][0]
        cluster = word[a_end:b_start]
        if not cluster:
            cut_points.append(a_end)  # hiatus
            continue
        units = _es_consonant_units(cluster)
        lows = [u.lower() for u in units]
        nu = len(units)
        if nu == 1:
            split_unit = 0
        elif nu == 2:
            split_unit = 0 if (lows[0] + lows[1]) in ES_INSEPARABLE else 1
        elif nu == 3:
            split_unit = 1 if (lows[1] + lows[2]) in ES_INSEPARABLE else 2
        else:
            split_unit = nu - 2 if (lows[-2] + lows[-1]) in ES_INSEPARABLE else nu - 1
        offset = sum(len(u) for u in units[:split_unit])
        cut_points.append(a_end + offset)

    out: List[str] = []
    prev = 0
    for c in cut_points:
        if c > prev:
            out.append(word[prev:c])
            prev = c
    out.append(word[prev:])
    return tuple(s for s in out if s)


def syllabify_es(word: str) -> List[str]:
    """Spanish RAE syllabification. Example: 'cercano' -> ['cer', 'ca', 'no']."""
    return list(_syllabify_es_cached(word))


# ----------------------------------------------------------------------------------------
# English
# ----------------------------------------------------------------------------------------

# Singing lexicon overrides (SingStar / UltraStar conventions)
EN_SINGING_LEXICON = {
    "every": ["Ev", "ery"], "everything": ["Ev", "ery", "thing"], "everyone": ["Ev", "ery", "one"],
    "everybody": ["Ev", "ery", "bo", "dy"], "everywhere": ["Ev", "ery", "where"],
    "only": ["on", "ly"], "around": ["a", "round"], "baby": ["ba", "by"], "babe": ["babe"],
    "ever": ["ev", "er"], "never": ["nev", "er"], "forever": ["for", "ev", "er"], "over": ["o", "ver"],
    "away": ["a", "way"], "alone": ["a", "lone"], "again": ["a", "gain"], "about": ["a", "bout"],
    "alive": ["a", "live"], "above": ["a", "bove"], "along": ["a", "long"], "awake": ["a", "wake"],
    "ago": ["a", "go"], "afraid": ["a", "fraid"], "another": ["an", "oth", "er"],
    "many": ["ma", "ny"], "any": ["a", "ny"], "very": ["ve", "ry"], "body": ["bo", "dy"],
    "lady": ["la", "dy"], "city": ["ci", "ty"], "easy": ["ea", "sy"], "busy": ["bu", "sy"],
    "tiny": ["ti", "ny"], "into": ["in", "to"], "onto": ["on", "to"], "upon": ["up", "on"],
    "even": ["e", "ven"], "open": ["o", "pen"], "heaven": ["hea", "ven"], "little": ["lit", "tle"],
    "people": ["peo", "ple"], "together": ["to", "geth", "er"], "nothing": ["noth", "ing"],
    "something": ["some", "thing"], "someone": ["some", "one"], "maybe": ["may", "be"],
    "because": ["be", "cause"], "before": ["be", "fore"], "believe": ["be", "lieve"],
    "beautiful": ["beau", "ti", "ful"], "fire": ["fire"], "hour": ["hour"], "our": ["our"],
    "lonely": ["lone", "ly"], "really": ["real", "ly"],
    "crying": ["cry", "ing"], "dying": ["dy", "ing"], "trying": ["try", "ing"],
    "flying": ["fly", "ing"], "lying": ["ly", "ing"], "eyes": ["eyes"], "aches": ["aches"],
    "watching": ["watch", "ing"], "belong": ["be", "long"], "embrace": ["em", "brace"],
    "since": ["since"], "gone": ["gone"], "cold": ["cold"], "lost": ["lost"],
}

EN_VOWELS = set("aeiouyAEIOUY")
EN_DIGRAPHS = ("th", "sh", "ch", "ph", "wh", "ck", "ng", "gh", "qu")
EN_ONSETS = {"bl", "br", "cl", "cr", "dr", "fl", "fr", "gl", "gr", "pl", "pr", "tr", "thr", "str", "scr", "spr", "shr"}


def _restore_case(template: str, parts: List[str]) -> List[str]:
    """Apply the case of ``template`` to lexicon parts (lexicon stores lowercase-ish strings)."""
    out = []
    pos = 0
    for p in parts:
        seg = template[pos:pos + len(p)]
        out.append(seg if len(seg) == len(p) else p)
        pos += len(p)
    if "".join(out) != template:
        return [template]
    return out


def _en_vowel_mask(word: str) -> List[bool]:
    low = word.lower()
    n = len(low)
    mask = [c in "aeiou" for c in low]
    for i, c in enumerate(low):
        if c == "y":
            # consonant 'y' at word start before vowel ('you', 'yes'), vocalic otherwise
            if i == 0 and n > 1 and low[1] in "aeiou":
                mask[i] = False
            elif i > 0 and mask[i - 1] and low[i - 1] in "aeo":
                # 'ay','ey','oy' glide: part of previous vowel group
                mask[i] = True
            else:
                mask[i] = True
        if c == "u" and i > 0 and low[i - 1] == "q":
            mask[i] = False
    return mask


def _en_silent_final(word: str, mask: List[bool]) -> List[bool]:
    """Mark silent final 'e' / '-es' / '-ed' as non-nuclear."""
    low = word.lower()
    n = len(low)
    m = list(mask)
    n_groups = sum(1 for i in range(n) if m[i] and (i == 0 or not m[i - 1]))
    if n_groups <= 1:
        return m
    if low.endswith("e") and n >= 3 and not m[n - 2]:
        # syllabic -le after consonant ('lit-tle', 'ta-ble') keeps the syllable
        if not (low.endswith("le") and n >= 3 and low[n - 3] not in "aeiouy"):
            m[n - 1] = False
    elif low.endswith("es") and n >= 4 and not m[n - 3]:
        if low[n - 3] not in "sxzcg" and low[n - 4:n - 2] not in ("ch", "sh"):
            m[n - 2] = False
    elif low.endswith("ed") and n >= 4 and not m[n - 3]:
        if low[n - 3] not in "td":
            m[n - 2] = False
    return m


def _en_units(cluster: str) -> List[str]:
    units: List[str] = []
    i = 0
    low = cluster.lower()
    while i < len(cluster):
        if low[i:i + 2] in EN_DIGRAPHS:
            units.append(cluster[i:i + 2])
            i += 2
        else:
            units.append(cluster[i])
            i += 1
    return units


def syllabify_en_rules(word: str) -> List[str]:
    """Rule-based English syllabification (vowel nuclei + maximal legal onset)."""
    if not word:
        return []
    low_w = word.lower()
    # Productive suffix '-ing' ('watch-ing', 'cry-ing'): split before it when the stem has a vowel
    if len(low_w) > 5 and low_w.endswith("ing") and any(c in "aeiouy" for c in low_w[:-3]):
        return syllabify_en_rules(word[:-3]) + [word[-3:]]
    mask = _en_silent_final(word, _en_vowel_mask(word))
    n = len(word)
    syllabic_le = low_w.endswith("le") and n >= 4 and low_w[n - 3] not in "aeiouy" and mask[n - 1]
    nuclei: List[List[int]] = []
    i = 0
    while i < n:
        if mask[i]:
            j = i + 1
            while j < n and mask[j]:
                j += 1
            nuclei.append([i, j])
            i = j
        else:
            i += 1
    if len(nuclei) <= 1:
        return [word]
    cuts: List[int] = []
    for k in range(len(nuclei) - 1):
        a_end, b_start = nuclei[k][1], nuclei[k + 1][0]
        cluster = word[a_end:b_start]
        if not cluster:
            cuts.append(a_end)
            continue
        units = _en_units(cluster)
        lows = [u.lower() for u in units]
        nu = len(units)
        if syllabic_le and k == len(nuclei) - 2 and nu >= 2:
            # consonant + 'le' forms the last syllable ('ta-ble', 'lit-tle', 'can-dle')
            split_unit = nu - 2
        elif nu == 1:
            split_unit = 1 if lows[0] in ("ck", "ng", "x") else 0
        elif nu == 2:
            split_unit = 0 if (lows[0] + lows[1]) in EN_ONSETS else 1
        else:
            if (lows[-2] + lows[-1]) in EN_ONSETS or "".join(lows[-3:]) in EN_ONSETS:
                split_unit = nu - 2 if "".join(lows[-3:]) not in EN_ONSETS else nu - 3
            else:
                split_unit = nu - 1
        cuts.append(a_end + sum(len(u) for u in units[:split_unit]))
    out: List[str] = []
    prev = 0
    for c in cuts:
        if c > prev:
            out.append(word[prev:c])
            prev = c
    out.append(word[prev:])
    return [s for s in out if s]


@lru_cache(maxsize=1)
def _cmu_dict():
    try:
        import pronouncing  # type: ignore
        pronouncing.init_cmu()
        return ("pronouncing", pronouncing)
    except Exception:
        pass
    try:
        from nltk.corpus import cmudict  # type: ignore
        return ("nltk", cmudict.dict())
    except Exception:
        return None


def cmu_syllable_count(word: str) -> Optional[int]:
    """Syllable count from CMUdict (number of stressed-digit phonemes), if available."""
    backend = _cmu_dict()
    if backend is None:
        return None
    w = word.lower()
    try:
        if backend[0] == "pronouncing":
            phones = backend[1].phones_for_word(w)
            if phones:
                return backend[1].syllable_count(phones[0])
        else:
            prons = backend[1].get(w)
            if prons:
                return sum(1 for ph in prons[0] if ph[-1].isdigit())
    except Exception:
        return None
    return None


def syllabify_en(word: str, typographic: Optional[List[str]] = None) -> List[str]:
    """English sung-syllable split.

    Priority: singing lexicon > typographic split whose count matches the phonetic count >
    rule-based split whose count matches > typographic split > rule split.
    """
    if not word:
        return []
    low = word.lower()
    if low in EN_SINGING_LEXICON:
        return _restore_case(word, EN_SINGING_LEXICON[low])
    rules = syllabify_en_rules(word)
    target = cmu_syllable_count(word)
    if target is None:
        target = len(rules)
    if typographic and len(typographic) == target and "".join(typographic) == word:
        return list(typographic)
    if len(rules) == target:
        return rules
    if typographic and "".join(typographic) == word:
        return list(typographic)
    return rules


def count_syllables_es(word: str) -> int:
    return max(1, len(syllabify_es(re.sub(r"[^\wáéíóúüñÁÉÍÓÚÜÑ]", "", word)))) if word else 0
