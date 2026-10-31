"""Módulo de Fuzzy Matching a nivel de caracteres y Rechazo de Ad-libs para UltraSinger.

Inspirado en arquitecturas SOTA (como lyric-align) para evitar desfases causados
por tarareos, ad-libs ("Ahhh", "Mmmm") y vocalizaciones no líricas al inicio de la canción.
"""

import math
import re
import unicodedata
import difflib
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional, Union

try:
    from modules.Speech_Recognition.TranscribedData import TranscribedData
except Exception:
    class TranscribedData:
        pass


def normalize_text_for_phonetic_matching(text: str) -> str:
    """Normaliza texto eliminando puntuación, acentos diacríticos y espacios superfluos.

    Ejemplo: '¡Canté!' -> 'cante', '¿Qué?' -> 'que'
    """
    if not text:
        return ""
    # Descomponer caracteres Unicode y remover tildes
    nfkd = unicodedata.normalize("NFKD", text.lower())
    without_accents = "".join(c for c in nfkd if not unicodedata.combining(c))
    # Conservar solo caracteres alfanuméricos
    return re.sub(r"[^\w]", "", without_accents)


def character_level_similarity(word_a: str, word_b: str) -> float:
    """Calcula la similitud difusa a nivel de caracteres entre dos palabras en el rango [0.0, 1.0].

    Utiliza el ratio de coincidencia de secuencias de caracteres normalizado.
    """
    clean_a = normalize_text_for_phonetic_matching(word_a)
    clean_b = normalize_text_for_phonetic_matching(word_b)

    if not clean_a and not clean_b:
        return 1.0
    if not clean_a or not clean_b:
        return 0.0
    if clean_a == clean_b:
        return 1.0

    return difflib.SequenceMatcher(None, clean_a, clean_b).ratio()


def compute_phonotactic_entropy_and_reduplication(word: str) -> Tuple[float, float]:
    """Calcula la Entropía Fonotáctica de Shannon H_char(w) en bits y el Índice de
    Reduplicación de Bigramas R_redup(w) en [0.0, 1.0].

    Fórmulas matemáticas:
      H_char(w) = - sum_{c in Sigma_w} p(c) * log2(p(c))
      R_redup(w) = 1.0 - (|Bigramas_Unicos(w)| / max(1, L - 1))
    """
    clean = normalize_text_for_phonetic_matching(word)
    length = len(clean)
    if length == 0:
        return 0.0, 1.0
    if length == 1:
        return 0.0, 0.0

    counts = Counter(clean)
    entropy = 0.0
    for cnt in counts.values():
        p = cnt / length
        entropy -= p * math.log2(p)

    bigrams = [clean[i : i + 2] for i in range(length - 1)]
    unique_bigrams = len(set(bigrams))
    redup_index = 1.0 - (unique_bigrams / max(1, len(bigrams)))
    return float(entropy), float(redup_index)


# Palabras gramaticales y léxicas comunes que NUNCA deben ser catalogadas como onomatopeya/scat
VALID_LEXICAL_WORDS = {
    # Single-vowel words/syllables are lexical ('a', 'e', 'o', 'u', 'y'), and initial
    # syllables like 'E-lla', 'u-na', 'e-se' must never be treated as scat/freestyle.
    "e", "o", "u", "i", "y",
    "la", "a", "de", "el", "un", "una", "en", "te", "me", "se", "le", "lo", "los", "las",
    "ya", "no", "si", "tu", "su", "mi", "va", "da", "ha", "he", "que", "por", "con",
    "sol", "sal", "mar", "luz", "paz", "ser", "ver", "hoy", "soy", "voy", "rey", "ley", "dios",
    "mama", "papa", "bebe", "nene", "nena", "cama", "casa", "nada", "todo", "vida",
    "como", "para", "pero", "mas", "sin", "son", "tan", "van", "dan", "pan", "ese", "esa",
    "eso", "aca", "alla", "asi", "ojo", "oro", "ala", "ola", "oso", "ama", "amo",
    "amor", "corazon", "tiempo", "mundo", "noche", "dia", "agua", "fuego", "cielo",
    "banana", "manana", "semana", "ventana", "campana", "palabra", "camino",
}

# Sufijos y prefijos morfológicos léxicos reales (español e inglés) que indican palabra con raíz semántica
LEXICAL_MORPH_AFFIXES = (
    "cion", "sion", "miento", "mente", "idad", "anza", "encia", "ando", "iendo",
    "ador", "edora", "ista", "ismo", "able", "ible", "oso", "osa", "ero", "era",
    "tion", "ment", "ness", "ling", "ight", "ough", "ould", "ther", "atio",
)

# Taxonomía Musical de Vocables No Léxicos (Súper Listado clasificado por función acústico-musical)
VOCABLE_CATEGORY_PATTERNS: Dict[str, re.Pattern] = {
    # 1. Imitación de batería/caja/guitarra/scratch (100% percusivos/sordos -> Freestyle 'F')
    "instrument_percussion_scratch": re.compile(
        r"^((rat+a*tat+)|(ts+)+(ts+)*|(pa+ta+ta+)|(chika+)+|(wah+)+|dze+n+|"
        r"shh+|psst+|tictac+|plop+|pow+|bang+|boom+|pum+)$",
        re.IGNORECASE,
    ),
    # 2. Ad-libs urbanos Hip-Hop / Reggaeton / Trap (percusivos o exclamaciones -> Freestyle 'F')
    "urban_adlib": re.compile(
        r"^(skrr+t*|brr+a*|prr+a*|wuh+|yah+|yeh+|yeah+|uepa+|wepa+|epa+|mambo+|"
        r"aj+a+|hey+|ey+|huy+|uf+|uff+|bah+|meh+)$",
        re.IGNORECASE,
    ),
    # 3. Scat & Jazz (improvisación vocal síncopada)
    "scat_jazz": re.compile(
        r"^((sho+be+do+)|(bop+ba+do+wop+)|(skid+ly(bop)?(do+)?)|(dwe+b(da)?(bop)?)|"
        r"(sco+by(do+by)*(do+)*)|(zap+(zaba+)?(do+)*)|(do+wah+(did+y)*)|(sho+wop+)+|"
        r"skid+ly|dwe+b|sco+by|do+by|zabado+|sho+be+|bop+|wop+|do+wop+)$",
        re.IGNORECASE,
    ),
    # 4. Folk & Celta (Lilting / Puirt à beul)
    "folk_lilting": re.compile(
        r"^((did+le)+(dum+)*|(to+ra+)+(lo+ra+)*|(did+ly(idle)*(um+)*)|"
        r"did+le|did+ly|to+ra+|lo+ra+)$",
        re.IGNORECASE,
    ),
    # 5. Imitación de Bajo / Bombo / Metales / Trompetas (con fundamental tonal)
    "instrument_bass_brass": re.compile(
        r"^((bum+){2,}|(dom+){2,}|ba+dum+|dum+di+dum+|pa+ra+pa+pam+|"
        r"ta+ra+ra+|ta+da+ra+ra+|tu+tu+ru+|bum+|dom+)$",
        re.IGNORECASE,
    ),
    # 6. Pop & Rock (Estribillos y coros)
    "pop_rock_chorus": re.compile(
        r"^((sha+(la+)+)|(who+a+(oh+)*)|(ye+a*h*){2,}|(na+){2,}|(la+){2,}|(da+){2,}|"
        r"(ta+|da+|ra+|na+|ba+|pa+|la+|za+|sha+){2,})$",
        re.IGNORECASE,
    ),
    # 7. Vocales puras melismáticas (exige vocal repetida o 'h' explícita: 'aah', 'oh', 'ooh';
    #    una vocal suelta 'a'/'e'/'o'/'u' es léxica, no scat)
    "pure_melismatic": re.compile(
        r"^((eh+){2,}|uh+huh+|(ay+){2,}|(o+e+y+)+|uhm+|um+|a{2,}h*|ah+|o{2,}h*|oh+|u{2,}h*|uh+|e{2,}h*|eh+|ay+|"
        r"m{2,}|u{2,}h*m*|uh+m*|w+o+a*h*|w+o+w*|u+o+h+)$",
        re.IGNORECASE,
    ),
}


def compute_markov_cv_periodicity_and_morphology(word: str) -> Tuple[float, bool]:
    """Calcula la probabilidad de transición de Cadena de Markov Consonante-Vocal P_CV(w)
    y verifica si la palabra carece de raíces/afijos morfológicos léxicos.

    Fórmula de Cadena de Markov de 2 estados {C, V}:
      P_CV(w) = (N(C -> V) + N(V -> C)) / max(1, L - 1)
    Los vocables iterativos (ej. 'l-a-l-a-l-a', 'b-a-d-u-m', 't-a-r-a-r-a') maximizan
    la alternancia periódica C <-> V o presentan plantillas silábicas reducidas sin morfemas léxicos.
    """
    clean = normalize_text_for_phonetic_matching(word)
    length = len(clean)
    if length <= 1:
        return 0.0, False

    vowels = set("aeiouy")
    states = ["V" if ch in vowels else "C" for ch in clean]
    alternations = sum(1 for i in range(length - 1) if states[i] != states[i + 1])
    p_cv = alternations / max(1, length - 1)

    has_lexical_morphology = (
        clean in VALID_LEXICAL_WORDS
        or any(clean.endswith(aff) or clean.startswith(aff) for aff in LEXICAL_MORPH_AFFIXES)
    )
    return float(p_cv), bool(has_lexical_morphology)


def compute_levenshtein_oov_distance(
    word: str, reference_vocab: Optional[set] = None
) -> Tuple[float, bool]:
    """Calcula la distancia mínima normalizada de Levenshtein d_Lev(w, V) en [0.0, 1.0]
    contra el vocabulario léxico de referencia y determina si es una jitanjáfora / vocable OOV.

    Fórmula:
      d_Lev(w, V) = min_{v in V} (1.0 - Sim_char(w, v))
      Es Jitanjáfora OOV <=> d_Lev(w, V) > 0.45 y carece de morfología léxica estándar.
    """
    clean = normalize_text_for_phonetic_matching(word)
    if not clean:
        return 1.0, True

    vocab = reference_vocab if reference_vocab is not None else VALID_LEXICAL_WORDS
    if clean in vocab:
        return 0.0, False

    _, has_morph = compute_markov_cv_periodicity_and_morphology(clean)
    if not vocab:
        return 1.0, not has_morph

    min_dist = min(1.0 - character_level_similarity(clean, v) for v in vocab)
    is_jitanjafora_oov = (min_dist > 0.45) and (not has_morph)
    return float(min_dist), bool(is_jitanjafora_oov)


def detect_closed_mouth_humming_acoustics(
    spectral_centroid_mean: float,
    f0_conf_mean: float,
    rms_cv: float = 0.25,
) -> bool:
    """Fórmula 2 (Universal / Independiente del Idioma): Detección de Tarareos ('Mmmm')
    mediante Centroide Espectral C y estabilidad de energía.

    Al tararear con la cavidad bucal cerrada, los labios y mejillas actúan como un filtro
    paso bajo natural:
      C = (sum f(n) * x(n)) / (sum x(n)) < 1000 Hz
      Conf_F0 >= 0.50  y  CV_RMS = sigma_RMS / (mu_RMS + eps) <= 0.55
    """
    return (
        0.0 < spectral_centroid_mean < 1000.0
        and f0_conf_mean >= 0.50
        and rms_cv <= 0.55
    )


def detect_rhythmic_transient_envelope(
    rms_crest_factor: float,
    sustained_duty_cycle: float = 0.35,
) -> bool:
    """Fórmula 3 (Universal / Independiente del Idioma): Detección de Onomatopeyas Rítmicas
    y Jitanjáforas ('Ta-ra-rá', 'Dum-dum') mediante transitorios de envolvente de amplitud.

      T_crest = max(RMS) / (mean(RMS) + eps) >= 2.10  y  DutyCycle_sostenido < 0.50
    """
    return rms_crest_factor >= 2.10 and sustained_duty_cycle < 0.50


def classify_vocable_category(word: str) -> Optional[str]:
    """Clasifica una palabra o expresión compuesta (ej. 'Shoo-be-doo', 'Rat-a-tat', 'Sha-la-la-la')
    dentro de las 7 categorías del Súper Listado de Vocables Musicales, o retorna None si es palabra léxica.
    """
    clean = normalize_text_for_phonetic_matching(word)
    if not clean:
        return "pure_melismatic"
    if clean in VALID_LEXICAL_WORDS:
        return None

    # 1. Coincidencia directa sobre la palabra normalizada completa
    for category, pattern in VOCABLE_CATEGORY_PATTERNS.items():
        if pattern.match(clean):
            return category

    # 2. Si viene con guiones (ej. 'shoo-be-doo', 'bop-ba-doo-wop', 'rat-a-tat'), evaluar sub-tokens
    raw_tokens = [
        normalize_text_for_phonetic_matching(tok)
        for tok in re.split(r"[\-\s_]+", word.strip())
        if normalize_text_for_phonetic_matching(tok)
    ]
    if len(raw_tokens) >= 2:
        matched_cats = []
        for tok in raw_tokens:
            if tok in VALID_LEXICAL_WORDS:
                continue
            tok_cat = None
            for category, pattern in VOCABLE_CATEGORY_PATTERNS.items():
                if pattern.match(tok):
                    tok_cat = category
                    break
            if tok_cat:
                matched_cats.append(tok_cat)
        if len(matched_cats) >= max(1, len(raw_tokens) - 1):
            return matched_cats[0]

    # 3. Evaluación matemática NLP (Entropía de Shannon + Reduplicación de Bigramas + Cadena de Markov CV + Morfología OOV)
    if len(clean) >= 4:
        h_char, r_redup = compute_phonotactic_entropy_and_reduplication(clean)
        p_cv, has_morph = compute_markov_cv_periodicity_and_morphology(clean)
        if not has_morph:
            if h_char < 1.15:
                return "pure_melismatic"
            if r_redup >= 0.50 or (r_redup >= 0.38 and p_cv >= 0.75):
                return "pop_rock_chorus"

    return None


def is_scat_or_adlib_pattern(word: str) -> bool:
    """Detecta matemática y fonotácticamente si una palabra corresponde a una onomatopeya,
    vocable no léxico clasificado, interjección expresiva, tarareo o scat ('Omatepa').
    """
    return classify_vocable_category(word) is not None


def classify_onomatopoeia_acoustics(
    word: str,
    v_mask_mean: float = 0.0,
    pitch_std_st: float = 0.5,
    spectral_flatness_mean: float = 0.2,
    rms_max: float = 0.05,
    noise_gate: float = 0.01,
    r_boca_mean: float = 1.5,
    in_reference_lyrics: bool = True,
    duration_sec: float = 0.20,
    f0_conf_mean: Optional[float] = None,
    rms_mean: Optional[float] = None,
    delta_mfcc_mean: Optional[float] = None,
    spectral_centroid_mean: Optional[float] = None,
    rms_crest_factor: Optional[float] = None,
    rms_cv: Optional[float] = None,
) -> str:
    """Clasificador Matemático-Acústico Tri-Modal para Vocables No Léxicos ('Omatepas') y Scats:
      - 'normal'   : No es vocable no léxico, o es un Vocable Melódico Cantado (Coros Pop/Rock 'Sha-la-la',
                     Vocales Melismáticas 'Whoa-oh-oh', Tarareo afinado en letra 'Mmmm' con C < 1000 Hz,
                     o Metales/Bajo entonados con V_mask >= 0.40, sigma_F0 <= 0.90 st y ||Delta MFCC||_2 estacionario).
      - 'freestyle': Imitación de percusión/batería/scratch ('Rat-a-tat', 'Tss-tss', 'Chika-chika'),
                     Ad-lib urbano ('Skrrt', 'Brrr', 'Pow', 'Wepa'), o Jitanjáfora Rítmica con transitorios
                     rápidos (T_crest >= 2.10) o alto flujo cepstral (||Delta MFCC||_2 >= 18.0) (-> Nota 'F').
      - 'purge'    : Suspiro / ruido o tarareo de boca cerrada (C < 1000 Hz) fuera de lyrics.txt.
    """
    category = classify_vocable_category(word)
    eff_v_mask = f0_conf_mean if f0_conf_mean is not None and v_mask_mean == 0.0 else v_mask_mean
    eff_rms = rms_mean if rms_mean is not None else rms_max
    has_active_energy = eff_rms >= (noise_gate * 1.25)

    # Fórmula 2: Detección acústica universal de tarareo con boca cerrada (C < 1000 Hz)
    is_closed_mouth_hum = False
    if spectral_centroid_mean is not None:
        is_closed_mouth_hum = detect_closed_mouth_humming_acoustics(
            spectral_centroid_mean=spectral_centroid_mean,
            f0_conf_mean=eff_v_mask,
            rms_cv=rms_cv if rms_cv is not None else 0.25,
        )
        if is_closed_mouth_hum and not in_reference_lyrics:
            return "purge"

    if category is None:
        return "normal"

    if not has_active_energy or (not in_reference_lyrics and r_boca_mean < 1.1 and not is_closed_mouth_hum):
        return "purge"

    # Regla Acústico-Taxonómica 1: Imitaciones de batería/caja/scratch ('rat-a-tat', 'tss-tss', 'chika-chika')
    # son intrínsecamente percusivas/no entonables -> siempre Freestyle ('F') cuando tienen energía activa
    if category == "instrument_percussion_scratch":
        return "freestyle"

    # Regla Acústico-Taxonómica 2: Flujo Cepstral de Mel (||Delta MFCC||_2) y Transitorios de Envolvente (T_crest)
    is_rapid_cepstral_articulation = delta_mfcc_mean is not None and delta_mfcc_mean >= 18.0
    is_rhythmic_transient = (
        rms_crest_factor is not None
        and detect_rhythmic_transient_envelope(rms_crest_factor=rms_crest_factor)
    )

    # Caso 1A: Tarareo melódico explícito en lyrics.txt ('Mmmm') con boca cerrada (C < 1000 Hz) y tono estable
    if is_closed_mouth_hum and in_reference_lyrics and pitch_std_st <= 0.90 and not is_rhythmic_transient:
        return "normal"

    # Caso 1B: Vocable Melódico Cantado con boca abierta (ej. 'Sha-la-la', 'Whoa-oh-oh', 'Pa-ra-pa-pam')
    if (
        eff_v_mask >= 0.40
        and pitch_std_st <= 0.90
        and spectral_flatness_mean < 0.25
        and r_boca_mean >= 1.3
        and not is_rapid_cepstral_articulation
        and not is_rhythmic_transient
        and (in_reference_lyrics or duration_sec >= 0.25)
    ):
        return "normal"

    # Caso 2: Vocable Percusivo / Ad-lib Urbano / Jitanjáfora con Transitorios -> Nota Freestyle ('F')
    if in_reference_lyrics or has_active_energy:
        return "freestyle"

    return "purge"


def _get_segment_word(segment: Union[Dict[str, Any], TranscribedData, Any]) -> str:
    """Obtiene la palabra de un segmento, ya sea objeto TranscribedData o diccionario."""
    if isinstance(segment, TranscribedData):
        return segment.word
    if isinstance(segment, dict):
        return segment.get("word", "")
    return getattr(segment, "word", str(segment))


def filter_intro_adlibs(
    whisper_segments: List[Union[Dict[str, Any], TranscribedData, Any]],
    target_first_words: List[str],
    similarity_threshold: float = 0.45,
) -> Tuple[List[Union[Dict[str, Any], TranscribedData, Any]], List[Union[Dict[str, Any], TranscribedData, Any]]]:
    """Descarta y pone en cuarentena los bloques acústicos iniciales de Whisper (ej. 'Ahhh')
    que no tengan correspondencia fonética/caracteres con la apertura de la letra.

    Si la letra oficial de referencia (target_first_words) comienza explícitamente con una
    interjección u onomatopeya (best_sim >= 0.85), se respeta y se ancla inmediatamente en ella.
    """
    if not whisper_segments or not target_first_words:
        return whisper_segments, []

    reference_candidates = [
        normalize_text_for_phonetic_matching(w)
        for w in target_first_words[:3]
        if normalize_text_for_phonetic_matching(w)
    ]

    if not reference_candidates:
        return whisper_segments, []

    anchor_idx = None
    quarantined = []

    for i, seg in enumerate(whisper_segments):
        raw_word = _get_segment_word(seg)
        clean_whisper = normalize_text_for_phonetic_matching(raw_word)

        if not clean_whisper:
            quarantined.append(seg)
            continue

        best_sim = max(
            (character_level_similarity(clean_whisper, target_cand) for target_cand in reference_candidates),
            default=0.0,
        )

        # Si coincide exactamente/altamente con la primera palabra de lyrics.txt (best_sim >= 0.85),
        # o supera el umbral sin ser un tarareo espontáneo ajeno al texto, anclamos aquí:
        if best_sim >= 0.85 or (best_sim >= similarity_threshold and not is_scat_or_adlib_pattern(raw_word)):
            anchor_idx = i
            break
        else:
            if isinstance(seg, dict):
                seg["is_vocablo"] = True
                seg["adlib_rejected"] = True
                seg["rejection_reason"] = f"Similitud insuficiente ({best_sim:.2f}) con inicio lírico {reference_candidates}"
            quarantined.append(seg)

    if anchor_idx is None:
        return whisper_segments, []

    return whisper_segments[anchor_idx:], quarantined


def fuzzy_lyric_matcher(
    whisper_segments: List[Union[Dict[str, Any], TranscribedData, Any]],
    real_lyrics_words: List[str],
    similarity_threshold: float = 0.45,
) -> Dict[str, Any]:
    """Cruza acústica y fonéticamente lo que Whisper detectó contra las palabras de lyrics.txt.

    Aplica Character-Level Fuzzy Matching y rechazo de ad-libs iniciales.

    Returns:
        Diccionario con segmentos filtrados, ad-libs en cuarentena, índice de anclaje inicial
        y métricas de alineación.
    """
    filtered_segments, quarantined = filter_intro_adlibs(
        whisper_segments, real_lyrics_words, similarity_threshold=similarity_threshold
    )

    anchor_word = _get_segment_word(filtered_segments[0]) if filtered_segments else None
    first_lyric = real_lyrics_words[0] if real_lyrics_words else None

    return {
        "filtered_segments": filtered_segments,
        "quarantined_adlibs": quarantined,
        "anchor_word": anchor_word,
        "first_target_lyric": first_lyric,
        "adlib_count": len(quarantined),
    }


def purge_instrumental_scat_segments(
    whisper_segments: List[Union[Dict[str, Any], TranscribedData, Any]],
    target_lyrics_words: List[str],
) -> Tuple[List[Union[Dict[str, Any], TranscribedData, Any]], List[Union[Dict[str, Any], TranscribedData, Any]]]:
    """Descarta segmentos que corresponden a tarareos / scat / alucinaciones instrumentales (ej. 'Tararara')
    cuando la letra de referencia no contiene esos términos explícitos.
    """
    if not whisper_segments:
        return [], []

    clean_target_set = {
        normalize_text_for_phonetic_matching(w)
        for w in target_lyrics_words
        if normalize_text_for_phonetic_matching(w)
    }

    retained = []
    purged = []

    for seg in whisper_segments:
        word = _get_segment_word(seg)
        clean_w = normalize_text_for_phonetic_matching(word)
        if is_scat_or_adlib_pattern(word) and clean_w not in clean_target_set:
            purged.append(seg)
        else:
            retained.append(seg)

    return retained, purged
