"""Genre Acoustic & Phonetic Profile Module for UltraSinger.

Provides tailored acoustic configurations, WhisperX initial_prompt conditioning,
PyAnnote VAD sensitivity thresholds, and USDX scoring rules for global musical families.
"""

from dataclasses import dataclass
import re
import unicodedata


def _strip_accents(text: str) -> str:
    """Normalize and remove diacritical marks/accents for robust keyword matching."""
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


@dataclass(frozen=True)
class GenreAcousticProfile:
    """Acoustic and phonetic parameters tailored for a specific musical genre."""
    name: str
    family: str
    initial_prompt: str
    vad_onset: float
    vad_offset: float
    no_speech_threshold: float
    prefer_freestyle_on_low_confidence: bool
    vibrato_tolerance_semitones: float
    melisma_sensitivity: float

    # Bilingual Prompts
    initial_prompt_en: str = ""
    initial_prompt_es: str = ""

    # DSP & Physical Acoustical Parameters
    vibrato_freq_range: tuple[float, float] = (4.5, 7.5)   # Target vocal modulation frequency f_v in Hz
    vibrato_max_amplitude_st: float = 1.0                 # Max expected biological vibrato depth A_v in semitones
    spectral_entropy_threshold: float = 0.85              # Shannon spectral entropy / Wiener flatness threshold for screams/growls
    rhythm_grid_mode: str = "straight"                    # "straight", "dembow_332", "swing_ternary"
    crest_factor_mode: str = "standard"                   # "standard", "high_dynamics" (ópera/clásica), "compressed" (pop)

    def get_initial_prompt(self, language: str | None = None) -> str:
        """Return the optimal initial prompt for the specified language.

        If language is Spanish ('es' or 'spanish'), returns the Spanish prompt.
        For English ('en'), unknown languages, or default, returns the English prompt.
        """
        if not language:
            return self.initial_prompt_en or self.initial_prompt
        lang_clean = language.lower().strip()
        if lang_clean.startswith("es") or lang_clean in ("spanish", "español"):
            return self.initial_prompt_es or self.initial_prompt
        return self.initial_prompt_en or self.initial_prompt


HIP_HOP_RNB = GenreAcousticProfile(
    name="Hip Hop / R&B",
    family="Urbanos, Modernos y Electrónicos",
    initial_prompt="Genre: Hip Hop / R&B. Rhythmic lyrics in strict 4/4 time with fast rhyming consonants, urban slang, contractions, double tracking, and ad-libs (yeah, uh, ey). Smooth R&B choruses with vocal harmonies.",
    initial_prompt_en="Genre: Hip Hop / R&B. Rhythmic lyrics in strict 4/4 time with fast rhyming consonants, urban slang, contractions, double tracking, and ad-libs (yeah, uh, ey). Smooth R&B choruses with vocal harmonies.",
    initial_prompt_es="Género: Hip Hop / R&B. Letra rítmica en compás 4/4 estricto con rimas rápidas y consonantes, lenguaje urbano, contracciones, dobles tomas y ad-libs (yeah, uh, ey). Coros R&B con armonías suaves.",
    vad_onset=0.350,
    vad_offset=0.320,
    no_speech_threshold=0.45,
    prefer_freestyle_on_low_confidence=True,
    vibrato_tolerance_semitones=0.8,
    melisma_sensitivity=0.8,
    vibrato_freq_range=(4.5, 7.5),
    vibrato_max_amplitude_st=0.3,
    spectral_entropy_threshold=0.80,
    rhythm_grid_mode="straight",
    crest_factor_mode="compressed",
)

REGGAETON_DEMBOW = GenreAcousticProfile(
    name="Reggaetón / Dembow",
    family="Urbanos, Modernos y Electrónicos",
    initial_prompt="Genre: Latin Reggaeton and Dembow. Danceable syncopated dembow rhythm, autotune, beat drops, pre-choruses, and Latin urban slang.",
    initial_prompt_en="Genre: Latin Reggaeton and Dembow. Danceable syncopated dembow rhythm, autotune, beat drops, pre-choruses, and Latin urban slang.",
    initial_prompt_es="Género: Reggaetón y Dembow latino. Letra bailable con ritmo sincopado de tresillo (dembow), autotune, caídas rítmicas (breaks), pre-estribillos y jerga urbana latina.",
    vad_onset=0.320,
    vad_offset=0.280,
    no_speech_threshold=0.40,
    prefer_freestyle_on_low_confidence=True,
    vibrato_tolerance_semitones=0.8,
    melisma_sensitivity=0.7,
    vibrato_freq_range=(4.5, 7.5),
    vibrato_max_amplitude_st=0.4,
    spectral_entropy_threshold=0.80,
    rhythm_grid_mode="dembow_332",
    crest_factor_mode="compressed",
)

EDM_HOUSE_TECHNO = GenreAcousticProfile(
    name="EDM / House / Techno",
    family="Urbanos, Modernos y Electrónicos",
    initial_prompt="Genre: Electronic Dance Music (EDM, House, Techno). Chopped vocal chops, hook repetitions, spatial reverb effects, and tension silences before drops.",
    initial_prompt_en="Genre: Electronic Dance Music (EDM, House, Techno). Chopped vocal chops, hook repetitions, spatial reverb effects, and tension silences before drops.",
    initial_prompt_es="Género: Música Electrónica (EDM, House, Techno). Letra con vocal chops fragmentados, repeticiones de estribillo, efectos espaciales con reverb masivo y silencios antes de las caídas (drops).",
    vad_onset=0.250,
    vad_offset=0.250,
    no_speech_threshold=0.55,
    prefer_freestyle_on_low_confidence=False,
    vibrato_tolerance_semitones=0.8,
    melisma_sensitivity=0.9,
    vibrato_freq_range=(4.5, 7.5),
    vibrato_max_amplitude_st=0.5,
    spectral_entropy_threshold=0.85,
    rhythm_grid_mode="straight",
    crest_factor_mode="compressed",
)

CUMBIA_GRUPERA = GenreAcousticProfile(
    name="Cumbia / Grupera",
    family="Latinos, Tropicales y Regionales",
    initial_prompt="Genre: Cumbia and Regional Grupera. Emotional regional love lyrics, nostalgic sentiment, natural vocal vibrato, walking basslines, and festive tropical shouts.",
    initial_prompt_en="Genre: Cumbia and Regional Grupera. Emotional regional love lyrics, nostalgic sentiment, natural vocal vibrato, walking basslines, and festive tropical shouts.",
    initial_prompt_es="Género: Cumbia y Música Grupera / Regional. Letra romántica regional con sentimiento popular, nostalgia, vibrato natural emotivo, bajo caminante y pregones tropicales.",
    vad_onset=0.280,
    vad_offset=0.220,
    no_speech_threshold=0.40,
    prefer_freestyle_on_low_confidence=False,
    vibrato_tolerance_semitones=1.2,
    melisma_sensitivity=0.9,
    vibrato_freq_range=(4.8, 5.8),
    vibrato_max_amplitude_st=1.2,
    spectral_entropy_threshold=0.88,
    rhythm_grid_mode="straight",
    crest_factor_mode="standard",
)

SALSA = GenreAcousticProfile(
    name="Salsa",
    family="Latinos, Tropicales y Regionales",
    initial_prompt="Genre: Salsa and Afro-Caribbean music. Syncopated clave rhythm, improvised vocal call-and-response (guía and coro), piano tumbao, Latin groove, and sharp brass breaks.",
    initial_prompt_en="Genre: Salsa and Afro-Caribbean music. Syncopated clave rhythm, improvised vocal call-and-response (guía and coro), piano tumbao, Latin groove, and sharp brass breaks.",
    initial_prompt_es="Género: Salsa y música afrocaribeña. Letra en clave rítmica con pregones improvisados, coros responsivos de llamada y respuesta (guía y coro), tumbao, sabor latino y cortes orquestales.",
    vad_onset=0.300,
    vad_offset=0.250,
    no_speech_threshold=0.50,
    prefer_freestyle_on_low_confidence=False,
    vibrato_tolerance_semitones=1.0,
    melisma_sensitivity=1.2,
    vibrato_freq_range=(4.5, 7.0),
    vibrato_max_amplitude_st=1.0,
    spectral_entropy_threshold=0.85,
    rhythm_grid_mode="straight",
    crest_factor_mode="standard",
)

ROCK_ALTERNATIVE = GenreAcousticProfile(
    name="Rock / Alternativo",
    family="Rock, Pop y Alternativos",
    initial_prompt="Genre: Rock and Alternative Rock. Gritty energetic vocals, controlled screams, powerful band backing vocals, punchy drums, and pauses before guitar solos.",
    initial_prompt_en="Genre: Rock and Alternative Rock. Gritty energetic vocals, controlled screams, powerful band backing vocals, punchy drums, and pauses before guitar solos.",
    initial_prompt_es="Género: Rock y Rock Alternativo. Letra con voz enérgica y rasposa, gritos controlados, coros de banda potentes, batería acústica contundente y pausas previas a solos de guitarra.",
    vad_onset=0.220,
    vad_offset=0.220,
    no_speech_threshold=0.45,
    prefer_freestyle_on_low_confidence=True,
    vibrato_tolerance_semitones=0.9,
    melisma_sensitivity=0.8,
    vibrato_freq_range=(4.5, 7.5),
    vibrato_max_amplitude_st=0.9,
    spectral_entropy_threshold=0.75,
    rhythm_grid_mode="straight",
    crest_factor_mode="standard",
)

POP_LATIN_POP = GenreAcousticProfile(
    name="Pop / Pop Latino",
    family="Rock, Pop y Alternativos",
    initial_prompt="Genre: Commercial Pop and Contemporary Pop. Clean vocal production, precise harmonies, melodic hooks, distinct verses, and radio-ready choruses.",
    initial_prompt_en="Genre: Commercial Pop and Contemporary Pop. Clean vocal production, precise harmonies, melodic hooks, distinct verses, and radio-ready choruses.",
    initial_prompt_es="Género: Pop comercial y Pop contemporáneo. Letra de canción con producción vocal limpia, armonías precisas, versos y estribillos con ganchos melódicos memorables (hooks).",
    vad_onset=0.280,
    vad_offset=0.240,
    no_speech_threshold=0.40,
    prefer_freestyle_on_low_confidence=False,
    vibrato_tolerance_semitones=0.8,
    melisma_sensitivity=1.0,
    vibrato_freq_range=(4.5, 7.5),
    vibrato_max_amplitude_st=0.8,
    spectral_entropy_threshold=0.85,
    rhythm_grid_mode="straight",
    crest_factor_mode="compressed",
)

SYMPHONIC_OPERA = GenreAcousticProfile(
    name="Música Sinfónica / Ópera",
    family="Música Clásica, Orquestal y Académica",
    initial_prompt="Genre: Opera and Symphonic Classical Music. Classical operatic vocal technique with wide expressive vibrato, orchestral acoustic projection, formal diction, and extreme dynamic range.",
    initial_prompt_en="Genre: Opera and Symphonic Classical Music. Classical operatic vocal technique with wide expressive vibrato, orchestral acoustic projection, formal diction, and extreme dynamic range.",
    initial_prompt_es="Género: Ópera y Música Sinfónica Clásica. Técnica lírica académica con vibrato amplio y natural, proyección orquestal, dicción clásica y dinámicas extremas de pianissimo a fortissimo.",
    vad_onset=0.180,
    vad_offset=0.150,
    no_speech_threshold=0.45,
    prefer_freestyle_on_low_confidence=False,
    vibrato_tolerance_semitones=1.4,
    melisma_sensitivity=1.3,
    vibrato_freq_range=(4.5, 7.5),
    vibrato_max_amplitude_st=1.5,
    spectral_entropy_threshold=0.90,
    rhythm_grid_mode="straight",
    crest_factor_mode="high_dynamics",
)

EXTREME_METAL = GenreAcousticProfile(
    name="Metal Extremo",
    family="Metal Extremo y Subgéneros Pesados",
    initial_prompt="Genre: Extreme Metal (Death Metal, Black Metal, Progressive Metal). Deep guttural growls, raspy screams, fast blast beats, and contrasts between eerie pauses and wall-of-sound distortion.",
    initial_prompt_en="Genre: Extreme Metal (Death Metal, Black Metal, Progressive Metal). Deep guttural growls, raspy screams, fast blast beats, and contrasts between eerie pauses and wall-of-sound distortion.",
    initial_prompt_es="Género: Metal Extremo (Death Metal, Black Metal, Metal Progresivo). Voces guturales (growls), gritos desgarrados (screams), blast beats rápidos y contrastes entre silencios y muros de distorsión.",
    vad_onset=0.200,
    vad_offset=0.200,
    no_speech_threshold=0.50,
    prefer_freestyle_on_low_confidence=True,
    vibrato_tolerance_semitones=1.0,
    melisma_sensitivity=0.6,
    vibrato_freq_range=(4.5, 7.5),
    vibrato_max_amplitude_st=0.6,
    spectral_entropy_threshold=0.70,
    rhythm_grid_mode="straight",
    crest_factor_mode="standard",
)

JAZZ_FUSION = GenreAcousticProfile(
    name="Jazz / Bebop / Fusion",
    family="Jazz y Músicas de Improvisación",
    initial_prompt="Genre: Jazz, Bebop, and Jazz Fusion. Flexible swing groove, warm tone, vocal improvisation (scat singing: doo-bee, ba-da, skat), and conversational live phrasing.",
    initial_prompt_en="Genre: Jazz, Bebop, and Jazz Fusion. Flexible swing groove, warm tone, vocal improvisation (scat singing: doo-bee, ba-da, skat), and conversational live phrasing.",
    initial_prompt_es="Género: Jazz, Bebop y Jazz Fusión. Ritmo con swing flexible, tesitura cálida, improvisación vocal (scat singing: doo-bee, ba-da, skat) y fraseo conversacional en vivo.",
    vad_onset=0.240,
    vad_offset=0.220,
    no_speech_threshold=0.50,
    prefer_freestyle_on_low_confidence=False,
    vibrato_tolerance_semitones=1.1,
    melisma_sensitivity=1.3,
    vibrato_freq_range=(4.5, 7.0),
    vibrato_max_amplitude_st=1.1,
    spectral_entropy_threshold=0.85,
    rhythm_grid_mode="swing_ternary",
    crest_factor_mode="standard",
)

WORLD_FLAMENCO_CELTIC = GenreAcousticProfile(
    name="World / Flamenco / Celta",
    family="World Music y Raíces Tradicionales",
    initial_prompt="Genre: Traditional World Music (Flamenco, Celtic, Afrobeats). Natural vocal breaks, expressive melismatic cries, traditional backing chants, polyrhythms, and rhythmic clapping.",
    initial_prompt_en="Genre: Traditional World Music (Flamenco, Celtic, Afrobeats). Natural vocal breaks, expressive melismatic cries, traditional backing chants, polyrhythms, and rhythmic clapping.",
    initial_prompt_es="Género: World Music tradicional (Flamenco, Música Celta, Afrobeats). Quiebres vocales naturales, lamentos expresivos, coros tradicionales, compases sincopados y palmas rítmicas.",
    vad_onset=0.250,
    vad_offset=0.220,
    no_speech_threshold=0.45,
    prefer_freestyle_on_low_confidence=False,
    vibrato_tolerance_semitones=1.2,
    melisma_sensitivity=1.4,
    vibrato_freq_range=(4.5, 8.0),
    vibrato_max_amplitude_st=1.4,
    spectral_entropy_threshold=0.85,
    rhythm_grid_mode="straight",
    crest_factor_mode="standard",
)

BALADA_ROMANTICA = GenreAcousticProfile(
    name="Balada Romántica",
    family="Latinos, Tropicales y Regionales",
    initial_prompt="Genre: Romantic Ballad. Intimate poetic lyrics, gentle whispers building into sustained high notes, long breath pauses, and expressive legato melodies.",
    initial_prompt_en="Genre: Romantic Ballad. Intimate poetic lyrics, gentle whispers building into sustained high notes, long breath pauses, and expressive legato melodies.",
    initial_prompt_es="Género: Balada Romántica. Letra emotiva y poética con susurros íntimos que escalan a notas altas sostenidas, silencios prolongados de respiración y melodía con legato expresivo.",
    vad_onset=0.200,
    vad_offset=0.180,
    no_speech_threshold=0.35,
    prefer_freestyle_on_low_confidence=False,
    vibrato_tolerance_semitones=1.2,
    melisma_sensitivity=1.1,
    vibrato_freq_range=(4.5, 6.5),
    vibrato_max_amplitude_st=1.2,
    spectral_entropy_threshold=0.88,
    rhythm_grid_mode="straight",
    crest_factor_mode="standard",
)


def match_single_genre_profile(genre_token: str) -> GenreAcousticProfile | None:
    """Match a single genre token against known genre profiles."""
    if not genre_token or not genre_token.strip():
        return None

    clean = genre_token.strip().lower()
    norm = _strip_accents(clean)

    def matches(keywords: tuple[str, ...]) -> bool:
        return any(k in clean or k in norm for k in keywords)

    # 1. Extreme Metal
    if matches(("death", "black metal", "grindcore", "deathcore", "thrash", "doom metal", "metalcore", "extreme metal")):
        return EXTREME_METAL

    # 2. Hip Hop & R&B
    if matches(("hip hop", "hip-hop", "rap", "boom bap", "trap", "drill", "r&b", "rhythm and blues", "urban")):
        return HIP_HOP_RNB

    # 3. Reggaetón & Dembow
    if matches(("reggaeton", "dembow", "perreo")):
        return REGGAETON_DEMBOW

    # 4. EDM / House / Techno
    if matches(("edm", "house", "techno", "trance", "dubstep", "electro", "dance", "synthwave", "electronic")):
        return EDM_HOUSE_TECHNO

    # 5. Salsa & Afro-Caribbean
    if matches(("salsa", "afrocaribe", "guaguanco", "mambo", "son cubano", "tumbao", "merengue", "bachata")):
        return SALSA

    # 6. Cumbia & Grupera / Regional
    if matches(("cumbia", "gruper", "norten", "banda", "ranchera", "mariachi", "tex-mex", "regional")):
        return CUMBIA_GRUPERA

    # 7. Balada Romántica & Bolero
    if matches(("balada", "bolero", "romantic")):
        return BALADA_ROMANTICA

    # 8. Jazz & Scat
    if matches(("jazz", "bebop", "bossa", "swing", "fusion", "scat")):
        return JAZZ_FUSION

    # 9. World, Flamenco & Celtic
    if matches(("flamenco", "celt", "afrobeat", "folk", "tango", "world music", "copla")):
        return WORLD_FLAMENCO_CELTIC

    # 10. Rock & Alternative
    if matches(("rock", "punk", "metal", "grunge", "indie", "hardcore", "alternative")):
        return ROCK_ALTERNATIVE

    # 11. Opera & Classical
    if matches(("opera", "sinfon", "clasic", "academic", "lirico", "aria", "choral")):
        return SYMPHONIC_OPERA

    # 12. Pop explicitly
    if matches(("pop",)):
        return POP_LATIN_POP

    return None


def split_genre_tokens(genre_str: str) -> list[str]:
    """Split genre string into individual genre tokens, supporting delimiters like /, ;, +, &, and commas."""
    if not genre_str:
        return []
    temp = re.sub(r'\s+&\s+', ',', genre_str)
    temp = re.sub(r'[/;+]', ',', temp)
    temp = re.sub(r'\s+(?:and|feat\.?|y)\s+', ',', temp, flags=re.IGNORECASE)
    parts = temp.split(',')
    return [p.strip() for p in parts if p.strip()]


def resolve_genre_profile(genre_str: str | None) -> GenreAcousticProfile:
    """Resolve a freeform genre string, multiple genres, or hybrid fusion into the best GenreAcousticProfile."""
    if not genre_str or not genre_str.strip():
        return POP_LATIN_POP

    tokens = split_genre_tokens(genre_str)
    if not tokens:
        return POP_LATIN_POP

    # Single token case: directly match, fallback for unknown/generic, or create dynamic custom profile
    if len(tokens) == 1:
        matched = match_single_genre_profile(tokens[0])
        if matched is not None:
            return matched
        if tokens[0].lower().startswith("unknown"):
            return POP_LATIN_POP
        token_title = tokens[0].strip().title()
        return GenreAcousticProfile(
            name=token_title,
            family="Género Personalizado / Adaptativo",
            initial_prompt=f"Genre: {token_title}. Clear vocal diction, natural cadence, and accurate lyrical transcription.",
            initial_prompt_en=f"Genre: {token_title}. Clear vocal diction, natural cadence, and accurate lyrical transcription.",
            initial_prompt_es=f"Género: {token_title}. Dicción vocal clara, cadencia natural y transcripción lírica precisa.",
            vad_onset=0.250,
            vad_offset=0.250,
            no_speech_threshold=0.40,
            prefer_freestyle_on_low_confidence=False,
            vibrato_tolerance_semitones=1.0,
            melisma_sensitivity=0.8,
            vibrato_freq_range=(4.5, 7.5),
            vibrato_max_amplitude_st=1.0,
            spectral_entropy_threshold=0.85,
            rhythm_grid_mode="straight",
            crest_factor_mode="standard",
        )

    # Multi-token / Mixed / Fusion genre case (e.g. "Merengue/House/Hip Hop")
    token_profiles: list[tuple[str, GenreAcousticProfile]] = []
    unmatched_tokens: list[str] = []

    for t in tokens:
        p = match_single_genre_profile(t)
        if p is not None:
            token_profiles.append((t, p))
        else:
            unmatched_tokens.append(t.strip().title())

    # If completely unmatched, build dynamic custom fusion profile
    if not token_profiles:
        combined_title = " / ".join(t.strip().title() for t in tokens)
        return GenreAcousticProfile(
            name=f"{combined_title} (Fusión)",
            family="Fusión / Género Personalizado",
            initial_prompt=f"Genre: {combined_title}. Clear vocal articulation, distinct lyrical delivery, and expressive phrasing.",
            initial_prompt_en=f"Genre: {combined_title}. Clear vocal articulation, distinct lyrical delivery, and expressive phrasing.",
            initial_prompt_es=f"Género: {combined_title}. Articulación vocal clara, entrega lírica definida y fraseo expresivo.",
            vad_onset=0.250,
            vad_offset=0.250,
            no_speech_threshold=0.40,
            prefer_freestyle_on_low_confidence=False,
            vibrato_tolerance_semitones=1.0,
            melisma_sensitivity=0.8,
            vibrato_freq_range=(4.5, 7.5),
            vibrato_max_amplitude_st=1.0,
            spectral_entropy_threshold=0.85,
            rhythm_grid_mode="straight",
            crest_factor_mode="standard",
        )

    # Primary (dominant) profile is the first recognized token in user's order
    primary_token, primary_profile = token_profiles[0]

    # Collect secondary profiles (deduplicating by name)
    secondary_profiles = [p for _, p in token_profiles[1:] if p.name != primary_profile.name]

    if not secondary_profiles and not unmatched_tokens:
        # All tokens pointed to the same profile family
        return primary_profile

    # Build hybrid profile
    all_names = [t.strip().title() for t in tokens]
    hybrid_name = " / ".join(all_names)

    sec_names_en = [p.name for p in secondary_profiles] + unmatched_tokens
    sec_names_es = [p.name for p in secondary_profiles] + unmatched_tokens

    prompt_en = (
        f"Genre: {hybrid_name}. Fusion blending {primary_profile.name} as primary base "
        f"with elements of {', '.join(sec_names_en)}. "
        f"{primary_profile.initial_prompt_en or primary_profile.initial_prompt}"
    )
    prompt_es = (
        f"Género: {hybrid_name}. Fusión combinando base de {primary_profile.name} "
        f"con influencias de {', '.join(sec_names_es)}. "
        f"{primary_profile.initial_prompt_es or primary_profile.initial_prompt}"
    )

    prefer_freestyle = primary_profile.prefer_freestyle_on_low_confidence or any(
        p.prefer_freestyle_on_low_confidence for p in secondary_profiles
    )

    grid_mode = primary_profile.rhythm_grid_mode
    if grid_mode == "straight":
        for p in secondary_profiles:
            if p.rhythm_grid_mode in ("dembow_332", "swing_ternary"):
                grid_mode = p.rhythm_grid_mode
                break

    return GenreAcousticProfile(
        name=f"{hybrid_name} (Fusión)",
        family=f"Fusión / Híbrido (Base: {primary_profile.family})",
        initial_prompt=prompt_en,
        initial_prompt_en=prompt_en,
        initial_prompt_es=prompt_es,
        vad_onset=primary_profile.vad_onset,
        vad_offset=primary_profile.vad_offset,
        no_speech_threshold=primary_profile.no_speech_threshold,
        prefer_freestyle_on_low_confidence=prefer_freestyle,
        vibrato_tolerance_semitones=primary_profile.vibrato_tolerance_semitones,
        melisma_sensitivity=primary_profile.melisma_sensitivity,
        vibrato_freq_range=primary_profile.vibrato_freq_range,
        vibrato_max_amplitude_st=primary_profile.vibrato_max_amplitude_st,
        spectral_entropy_threshold=primary_profile.spectral_entropy_threshold,
        rhythm_grid_mode=grid_mode,
        crest_factor_mode=primary_profile.crest_factor_mode,
    )
