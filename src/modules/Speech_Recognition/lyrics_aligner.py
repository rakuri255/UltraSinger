"""Reference lyrics loader and forced aligner for UltraSinger.

Aligns human-curated lyrics.txt with Whisper acoustic timestamps, eliminating
both ASR hallucinations and run-on sentences in UltraStar.
"""

import os
import re
import difflib
from typing import List, Tuple, Optional

from modules.console_colors import (
    ULTRASINGER_HEAD,
    blue_highlighted,
    green_highlighted,
    gold_highlighted,
    red_highlighted,
)
from modules.Speech_Recognition.TranscribedData import TranscribedData
from modules.Speech_Recognition.fuzzy_lyric_matcher import (
    filter_intro_adlibs,
    character_level_similarity,
    is_scat_or_adlib_pattern,
    normalize_text_for_phonetic_matching,
    purge_instrumental_scat_segments,
)


def clean_word_for_matching(word: str) -> str:
    """Normalize word for sequence comparison (lowercase, unaccented, alphanumeric only)."""
    return normalize_text_for_phonetic_matching(word)


def split_long_line(line: str, max_chars: int = 50) -> List[str]:
    """Splits a line into balanced sub-lines if it exceeds max_chars, to prevent word-wrapping in UltraStar."""
    line = line.strip()
    if len(line) <= max_chars:
        return [line]

    # Check for punctuation splits (comma, semicolon, dash)
    for delim in [", ", "; ", " - "]:
        if delim in line:
            parts = line.split(delim)
            sub_lines = []
            curr = ""
            for idx, p in enumerate(parts):
                segment = p + (delim.rstrip() if idx < len(parts) - 1 else "")
                if not curr:
                    curr = segment
                elif len(curr) + 1 + len(segment) <= max_chars:
                    curr += " " + segment
                else:
                    sub_lines.append(curr.strip())
                    curr = segment
            if curr:
                sub_lines.append(curr.strip())
            if len(sub_lines) > 1 and all(len(sl) <= max_chars + 12 for sl in sub_lines):
                return sub_lines

    # If no punctuation or punctuation split didn't help, split by word count around middle
    words = line.split()
    if len(words) <= 4:
        return [line]

    mid = len(words) // 2
    return [" ".join(words[:mid]), " ".join(words[mid:])]


def load_lyrics_file(lyrics_path: str) -> List[Tuple[str, bool]]:
    """Load reference lyrics.txt and return list of (word, is_line_end) tuples.

    Each line in lyrics.txt defines an intended UltraStar display line. Long lines
    exceeding max_chars are split at natural punctuation marks to avoid on-screen text overflow.
    """
    if not os.path.isfile(lyrics_path):
        raise FileNotFoundError(f"Lyrics file not found: {lyrics_path}")

    encodings = ["utf-8-sig", "utf-8", "latin-1", "cp1252"]
    content = None
    for enc in encodings:
        try:
            with open(lyrics_path, "r", encoding=enc) as f:
                content = f.read()
            break
        except UnicodeDecodeError:
            continue

    if content is None:
        raise ValueError(f"Could not decode lyrics file {lyrics_path} with supported encodings.")

    lines = content.splitlines()
    lyrics_words: List[Tuple[str, bool]] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        for sub_line in split_long_line(stripped, max_chars=50):
            words = sub_line.strip().split()
            if not words:
                continue
            for idx, w in enumerate(words):
                is_line_end = (idx == len(words) - 1)
                lyrics_words.append((w, is_line_end))

    return lyrics_words


def detect_language_from_lyrics(
    lyrics_path: Optional[str], fallback_lang: Optional[str] = None
) -> Optional[str]:
    """Auto-detect song language ('en' or 'es') from reference lyrics.txt to prevent
    cross-language Whisper translation in mixed-language batches.
    """
    if not lyrics_path or not os.path.isfile(lyrics_path):
        return fallback_lang

    try:
        words_tuples = load_lyrics_file(lyrics_path)
        if not words_tuples:
            return fallback_lang

        raw_text = " ".join(w[0] for w in words_tuples).lower()
        tokens = re.findall(r"[a-záéíóúüñ']+", raw_text)
        if len(tokens) < 5:
            return fallback_lang

        en_stopwords = {
            "the", "you", "every", "your", "and", "with", "that", "this", "for",
            "are", "have", "from", "one", "all", "will", "can't", "don't", "you've",
            "i'm", "i'll", "my", "me", "be", "to", "of", "in", "it", "is", "on",
            "time", "world", "make", "good", "there", "night", "day", "way",
            "love", "baby", "when", "what", "where", "who", "why", "how", "know",
        }
        es_stopwords = {
            "el", "la", "los", "las", "de", "del", "que", "en", "un", "una", "con",
            "por", "para", "su", "sus", "mi", "mis", "tu", "tus", "se", "lo", "le",
            "no", "es", "son", "como", "más", "pero", "ya", "muy", "sin", "sobre",
            "también", "me", "te", "nos", "ella", "él", "todo", "toda", "vida",
            "estaba", "casa", "mundo", "día", "gente", "cuando", "donde",
        }

        en_hits = sum(1 for t in tokens if t in en_stopwords)
        es_hits = sum(1 for t in tokens if t in es_stopwords)
        es_chars = sum(1 for c in raw_text if c in "áéíóúñ¿¡")

        if en_hits >= 5 and en_hits > es_hits * 2.0 and es_chars == 0:
            return "en"
        if (es_hits >= 5 and es_hits > en_hits * 1.5) or es_chars >= 3:
            return "es"
    except Exception:
        pass

    return fallback_lang


def _detect_voiced_segments(
    audio_path: Optional[str], time_start: float, time_end: float
) -> List[Tuple[float, float]]:
    """Detect voiced regions using Librosa RMS energy within [time_start, time_end]."""
    if not audio_path or not os.path.isfile(audio_path):
        return []

    duration = max(0.0, time_end - time_start)
    if duration < 0.15:
        return []

    try:
        import librosa
        import numpy as np

        y, sr = librosa.load(audio_path, sr=16000, mono=True, offset=max(0.0, time_start), duration=duration)
        if len(y) == 0:
            return []

        hop_length = 256
        frame_length = 1024
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

        if len(rms) == 0 or np.max(rms) < 1e-4:
            return []

        p95 = float(np.percentile(rms, 95))
        threshold = max(p95 * 0.14, float(np.max(rms)) * 0.10, 0.005)
        is_voiced = rms > threshold

        raw_segments = []
        in_seg = False
        seg_start_t = 0.0

        for i in range(len(is_voiced)):
            if is_voiced[i] and not in_seg:
                seg_start_t = float(times[i])
                in_seg = True
            elif not is_voiced[i] and in_seg:
                raw_segments.append((seg_start_t, float(times[i])))
                in_seg = False
        if in_seg:
            raw_segments.append((seg_start_t, float(times[-1]) if len(times) > 0 else duration))

        # Merge segments with short gaps (< 0.25s) — breathing pauses between vocal sounds
        merged = []
        for seg in raw_segments:
            if merged and seg[0] - merged[-1][1] < 0.25:
                merged[-1] = (merged[-1][0], seg[1])
            else:
                merged.append(seg)

        # Filter out segments shorter than 80ms
        valid_segments = [(s + time_start, e + time_start) for s, e in merged if (e - s) >= 0.08]
        return valid_segments
    except Exception:
        return []


def compute_klatt_word_weight(word: str, is_line_end: bool = False) -> float:
    """Equation III: Dennis Klatt Structural Phonetic Duration Weighting."""
    w = word.strip().lower()
    vowels = set("aeiouáéíóúü")
    n_v = max(1, sum(1 for c in w if c in vowels))
    n_c = sum(1 for c in w if c.isalpha() and c not in vowels)
    return 3.0 * n_v + 1.0 * n_c + (1.5 if is_line_end else 0.0)


def count_word_syllables(word: str) -> int:
    """Count phonetic vowel nuclei (syllables) in a word."""
    clean = re.sub(r"[^a-záéíóúüñ]", "", word.strip().lower())
    if not clean:
        return 1
    vowel_groups = re.findall(r"[aeiouáéíóúü]+", clean)
    return max(1, len(vowel_groups))


def _find_lexical_articulatory_onset(
    audio_path: Optional[str],
    win_start: float,
    win_end: float,
    fallback_t: float,
) -> float:
    """Ecuación II: Localiza el Punto de Ataque Articulatorio Léxico (t_lex*) separando
    la cola de un tarareo previo del ataque consonántico/vocálico real de la palabra:
      L_art(t) = 0.45 * ||Delta MFCC(t)||_2 + 0.35 * E_HF(t) + 0.20 * Drop_RMS(t)
    """
    if not audio_path or not os.path.isfile(audio_path) or win_end <= win_start + 0.05:
        return fallback_t

    try:
        import librosa
        import numpy as np

        dur = win_end - win_start
        y, sr = librosa.load(audio_path, sr=16000, mono=True, offset=max(0.0, win_start), duration=dur)
        if len(y) < 512:
            return fallback_t

        hop_length = 160  # 10ms resolution
        n_fft = 512
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        mag = np.abs(D)
        if mag.shape[1] < 3:
            return fallback_t

        times = librosa.frames_to_time(np.arange(mag.shape[1]), sr=sr, hop_length=hop_length) + win_start
        fft_freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        # 1. Flujo Cepstral ||Delta MFCC||_2
        mfcc = librosa.feature.mfcc(S=librosa.amplitude_to_db(mag + 1e-6), sr=sr, n_mfcc=13)
        diff_mfcc = np.diff(mfcc, axis=1)
        delta_mfcc = np.pad(np.sqrt(np.sum(diff_mfcc ** 2, axis=0)), (1, 0), mode="edge")

        # 2. Energía Consonántica de Alta Frecuencia E_HF (3.0 - 8.0 kHz)
        hf_mask = (fft_freqs >= 3000) & (fft_freqs <= 8000)
        e_hf = np.sum(mag[hf_mask, :] ** 2, axis=0) if np.any(hf_mask) else np.zeros(mag.shape[1])

        # 3. Micro-caída y ataque de envolvente RMS (transición tarareo -> oclusiva/palabra)
        rms = librosa.feature.rms(S=mag, frame_length=n_fft)[0]
        d_rms_rise = np.maximum(0.0, np.gradient(rms))

        def _norm(arr: np.ndarray) -> np.ndarray:
            p95 = float(np.percentile(arr, 95)) if len(arr) > 0 else 1.0
            return arr / (p95 + 1e-6)

        l_art = 0.45 * _norm(delta_mfcc) + 0.35 * _norm(e_hf) + 0.20 * _norm(d_rms_rise)
        best_idx = int(np.argmax(l_art))
        return float(times[best_idx])
    except Exception:
        return fallback_t


def detach_tarareo_and_reanchor_phrases(
    aligned_data: List[TranscribedData],
    audio_path: Optional[str] = None,
) -> List[TranscribedData]:
    """Ecuaciones I y II:
    1. Re-ancla palabras iniciales huérfanas ('Yo' en 2.0s separada por 9.6s de 'quiero' en 12.1s)
       que fueron atrapadas en un tarareo inicial antes de que empiece la frase real.
    2. Despega las colas de tarareo fusionadas al inicio de la primera palabra de una estrofa
       ('Quiero' 40.38s-43.56s, 'Fueron' 115.05s-119.80s, 'Yo' 168.53s-174.10s) o al final ('a' 176.28s-181.72s).
    """
    if len(aligned_data) < 2:
        return aligned_data

    # Agrupar por líneas líricas originales (según is_line_end de lyrics.txt)
    lines: List[List[int]] = []
    curr_line: List[int] = []
    for idx, td in enumerate(aligned_data):
        curr_line.append(idx)
        if td.is_line_end or idx == len(aligned_data) - 1:
            lines.append(curr_line)
            curr_line = []

    reanchored_heads = 0
    detached_tarareos = 0

    # --- ECUACIÓN I: Re-anclaje de Palabra Inicial Huérfana (N-Gram Syntactic Cohesion) ---
    for line_indices in lines:
        if len(line_indices) < 2:
            continue
        i0 = line_indices[0]
        i1 = line_indices[1]
        w0 = aligned_data[i0]
        w1 = aligned_data[i1]

        gap_01 = float(w1.start) - float(w0.end)
        n_syl_0 = count_word_syllables(w0.word)

        # Calcular brecha mediana del resto de la oración [w1 .. w_m]
        rest_gaps = [
            float(aligned_data[line_indices[k + 1]].start) - float(aligned_data[line_indices[k]].end)
            for k in range(1, len(line_indices) - 1)
        ]
        median_rest_gap = sorted(rest_gaps)[len(rest_gaps) // 2] if rest_gaps else 0.05

        if gap_01 > 1.60 and n_syl_0 <= 2 and median_rest_gap < 0.85:
            # w0 ('Yo' en 2.0s) se ancló falsamente a un tarareo previo lejos de su oración ('quiero' en 12.1s)
            prev_bound = float(aligned_data[i0 - 1].end) + 0.15 if i0 > 0 else 0.0
            d_klatt = min(0.50, max(0.22, 0.16 + 0.16 * n_syl_0))
            new_end = max(prev_bound + d_klatt, float(w1.start) - 0.015)
            new_start = max(prev_bound, new_end - d_klatt)
            old_s = w0.start
            w0.start = round(new_start, 3)
            w0.end = round(new_end, 3)
            w0.chars = None
            reanchored_heads += 1
            print(
                f"{ULTRASINGER_HEAD} {gold_highlighted('Orphan Head Re-Anchor (Eq. I):')} "
                f"Detached '{w0.word.strip()}' from tarareo at {old_s:.2f}s -> anchored before "
                f"'{w1.word.strip()}' at {w0.start:.2f}s-{w0.end:.2f}s (gap was {gap_01:.2f}s)."
            )

    # --- ECUACIÓN II: Despegue de Tarareo Pre-Estrofa y Post-Estrofa (Klatt Upper Bound D_max) ---
    UNSTRESSED_PREPS = {"a", "de", "en", "con", "por", "sin", "mi", "tu", "su", "la", "el", "un", "y", "o", "que"}
    for line_indices in lines:
        if not line_indices:
            continue

        # 2A. Caso especial: Preposición átona penúltima ('a') que absorbió solo/tarareo empujando la última palabra ('mí')
        if len(line_indices) >= 2:
            idx_pen = line_indices[-2]
            idx_last = line_indices[-1]
            w_pen = aligned_data[idx_pen]
            w_last = aligned_data[idx_last]
            dur_pen = float(w_pen.end) - float(w_pen.start)
            clean_pen = clean_word_for_matching(w_pen.word)
            if clean_pen in UNSTRESSED_PREPS and dur_pen > 1.20:
                prep_dur = 0.24
                old_pen_end = w_pen.end
                w_pen.end = round(float(w_pen.start) + prep_dur, 3)
                w_pen.chars = None
                w_last.start = round(float(w_pen.end) + 0.015, 3)
                w_last.end = round(min(float(old_pen_end), float(w_last.start) + 0.95), 3)
                w_last.chars = None
                detached_tarareos += 1

        # 2B. Despegue Izquierdo en palabras que absorbieron cola de tarareo previo (palabras tras tarareo prolongado)
        for pos, idx in enumerate(line_indices):
            td = aligned_data[idx]
            dur = float(td.end) - float(td.start)
            n_syl = count_word_syllables(td.word)
            d_max_lead = min(1.10, 0.22 + 0.32 * n_syl)

            is_phrase_start = (pos == 0) or (
                idx > 0 and (float(td.start) - float(aligned_data[idx - 1].end)) >= 0.40
            )
            has_contiguous_next = (
                pos < len(line_indices) - 1
                and (float(aligned_data[line_indices[pos + 1]].start) - float(td.end)) < 0.85
            )

            if is_phrase_start and (has_contiguous_next or dur > 1.75 * d_max_lead) and dur > max(1.35, 1.65 * d_max_lead):
                win_s = max(float(td.start), float(td.end) - 1.25 * d_max_lead)
                win_e = max(win_s + 0.08, float(td.end) - 0.35 * d_max_lead)
                fallback_t = max(float(td.start), float(td.end) - 0.75 * d_max_lead)
                new_s = _find_lexical_articulatory_onset(audio_path, win_s, win_e, fallback_t)
                old_s = td.start
                td.start = round(new_s, 3)
                td.chars = None
                detached_tarareos += 1
                print(
                    f"{ULTRASINGER_HEAD} {gold_highlighted('Tarareo Tail Detach (Eq. II):')} "
                    f"Trimmed pre-verse humming from '{td.word.strip()}' ({old_s:.2f}s -> {td.start:.2f}s, "
                    f"end={td.end:.2f}s)."
                )
            elif dur > max(2.20, 2.50 * d_max_lead):
                # 2C. Despegue Derecho si una palabra final absorbió tarareo/solo posterior
                old_e = td.end
                td.end = round(float(td.start) + min(1.20, 1.25 * d_max_lead), 3)
                td.chars = None
                detached_tarareos += 1
                print(
                    f"{ULTRASINGER_HEAD} {gold_highlighted('Post-Verse Tarareo Cap (Eq. II):')} "
                    f"Capped trailing humming on '{td.word.strip()}' ({old_e:.2f}s -> {td.end:.2f}s)."
                )

    return aligned_data


def _distribute_words_in_gap(
    missing_words: List[Tuple[str, bool]],
    time_start: float,
    time_end: float,
    audio_path: Optional[str] = None,
) -> List[TranscribedData]:
    """Distribute missing lyrics words in a gap using Dennis Klatt phonetic duration weighting and acoustic onsets."""
    num_words = len(missing_words)
    if num_words == 0:
        return []

    # Phase 4 (Hierarchical Multi-Line Outro & Gap Distribution):
    # If missing words span multiple display lines, split into line groups and space each line
    # rhythmically across the available duration to preserve musical phrasing.
    line_groups = []
    curr_line = []
    for item in missing_words:
        curr_line.append(item)
        if item[1]:  # is_line_end
            line_groups.append(curr_line)
            curr_line = []
    if curr_line:
        line_groups.append(curr_line)

    if len(line_groups) > 1:
        total_time = max(0.20 * num_words, time_end - time_start)
        n_lines = len(line_groups)
        time_per_line = total_time / n_lines
        multi_results = []
        for l_idx, l_words in enumerate(line_groups):
            l_start = time_start + l_idx * time_per_line
            # Leave inter-phrase breathing pause (0.15s)
            l_end = max(l_start + 0.12 * len(l_words), l_start + time_per_line - 0.15)
            sub_res = _distribute_words_in_gap(l_words, l_start, l_end, audio_path=audio_path)
            multi_results.extend(sub_res)
        return multi_results

    time_start = max(0.0, time_start)
    time_end = max(time_start + 0.12 * num_words, time_end)

    voiced_segments = _detect_voiced_segments(audio_path, time_start, time_end) if audio_path else []

    word_weights = [compute_klatt_word_weight(lw[0], lw[1]) for lw in missing_words]
    total_w = max(0.1, sum(word_weights))

    results = []
    if voiced_segments:
        if len(voiced_segments) > num_words:
            # Keep the num_words longest voiced segments in chronological order
            top_segs = sorted(voiced_segments, key=lambda seg: seg[1] - seg[0], reverse=True)[:num_words]
            voiced_segments = sorted(top_segs, key=lambda seg: seg[0])

        # Distribute words across detected voiced segments proportionally
        seg_durations = [e - s for s, e in voiced_segments]
        total_dur = sum(seg_durations)

        if len(voiced_segments) == 1:
            s_start, s_end = voiced_segments[0]
            avail_dur = max(0.08 * num_words, s_end - s_start)
            cur_t = s_start
            for idx, (l_word, is_line_end) in enumerate(missing_words):
                w_frac = word_weights[idx] / total_w
                w_dur = avail_dur * w_frac
                w_s = cur_t
                w_e = w_s + w_dur * 0.95
                cur_t += w_dur
                results.append(
                    TranscribedData(
                        confidence=0.70,
                        word=l_word.rstrip(" ") + " ",
                        start=round(w_s, 3),
                        end=round(w_e, 3),
                        is_word_end=True,
                        is_line_end=is_line_end,
                    )
                )
        else:
            words_per_seg = []
            rem = num_words
            for idx, dur in enumerate(seg_durations):
                if idx == len(seg_durations) - 1:
                    count = max(0, rem)
                else:
                    max_for_this = max(0, rem - (len(seg_durations) - 1 - idx))
                    count = max(1, int(round((dur / max(1e-4, total_dur)) * num_words)))
                    count = max(0, min(count, max_for_this))
                words_per_seg.append(count)
                rem -= count

            word_idx = 0
            for seg_idx, (s_start, s_end) in enumerate(voiced_segments):
                count = words_per_seg[seg_idx]
                if count <= 0 or word_idx >= len(missing_words):
                    continue
                seg_words = missing_words[word_idx:word_idx + count]
                if not seg_words:
                    continue
                seg_w_weights = [compute_klatt_word_weight(sw[0], sw[1]) for sw in seg_words]
                seg_tot_w = max(0.1, sum(seg_w_weights))
                seg_dur = max(0.08 * len(seg_words), s_end - s_start)
                cur_s = s_start

                for sub_i in range(len(seg_words)):
                    l_word, is_line_end = seg_words[sub_i]
                    sub_dur = seg_dur * (seg_w_weights[sub_i] / seg_tot_w)
                    w_s = cur_s
                    w_e = w_s + sub_dur * 0.95
                    cur_s += sub_dur
                    results.append(
                        TranscribedData(
                            confidence=0.70,
                            word=l_word.rstrip(" ") + " ",
                            start=round(w_s, 3),
                            end=round(w_e, 3),
                            is_word_end=True,
                            is_line_end=is_line_end,
                        )
                    )
                word_idx += count

    # Fallback: Klatt duration weighting across [time_start, time_end]
    if not results:
        total_gap = time_end - time_start
        cur_t = time_start
        for idx, (l_word, is_line_end) in enumerate(missing_words):
            w_frac = word_weights[idx] / total_w
            w_dur = total_gap * w_frac
            w_s = cur_t
            w_e = w_s + w_dur * 0.92
            cur_t += w_dur
            results.append(
                TranscribedData(
                    confidence=0.55,
                    word=l_word.rstrip(" ") + " ",
                    start=round(w_s, 3),
                    end=round(w_e, 3),
                    is_word_end=True,
                    is_line_end=is_line_end,
                )
            )

    return results


def _is_ctc_crushed(td: TranscribedData) -> bool:
    """Detect if a word was crushed by WhisperX CTC alignment (near-zero confidence or unphysical syllable rate)."""
    n_syl = count_word_syllables(td.word)
    dur = max(0.0, float(td.end) - float(td.start))
    conf = float(getattr(td, "confidence", 0.5) or 0.0)
    if dur < 0.075 * n_syl:
        return True
    if conf < 0.25 and dur < 0.145 * n_syl:
        return True
    if conf < 0.12 and dur < 0.22 * n_syl:
        return True
    return False


def _is_bloated(td: TranscribedData) -> bool:
    """A word that absorbed far more audio than its Klatt upper bound (e.g. 'sol' lasting 6 s)."""
    n_syl = count_word_syllables(td.word)
    dur = max(0.0, float(td.end) - float(td.start))
    return dur > max(1.05, 0.25 + 0.35 * n_syl)


def repair_crushed_line_tails(
    aligned_data: List[TranscribedData],
    audio_path: Optional[str] = None,
    max_tail_words: int = 4,
) -> List[TranscribedData]:
    """Pattern C: crushed intra-line tails ('pu-pi-la', 'el en-cuen-tro', 'su-pues-to', 'de-re-cho').

    The last 1..4 words of a lyric line with < 85 ms/syllable are redistributed over the
    voiced island that starts at the first crushed word, bounded by the next line onset:
        d_k = (b - a) * w_k / sum(w),   w_k = 3 N_V + N_C + 1.5 [line end]
    """
    n_total = len(aligned_data)
    if n_total < 2:
        return aligned_data

    def _crushed_rate(td: TranscribedData) -> bool:
        n_syl = count_word_syllables(td.word)
        dur = max(0.0, float(td.end) - float(td.start))
        return dur < 0.085 * n_syl or _is_ctc_crushed(td)

    lines: List[List[int]] = []
    curr: List[int] = []
    for idx, td in enumerate(aligned_data):
        curr.append(idx)
        if td.is_line_end or idx == n_total - 1:
            lines.append(curr)
            curr = []

    repaired = 0
    for l_idx, line in enumerate(lines):
        suffix: List[int] = []
        for idx in reversed(line):
            if len(suffix) >= max_tail_words or not _crushed_rate(aligned_data[idx]):
                break
            suffix.insert(0, idx)
        if not suffix:
            continue
        first, last = suffix[0], suffix[-1]
        syls = sum(count_word_syllables(aligned_data[k].word) for k in suffix)
        t0 = float(aligned_data[first].start)
        if first > 0:
            t0 = max(t0, float(aligned_data[first - 1].end) + 0.02)
        next_start = float(aligned_data[last + 1].start) if last + 1 < n_total else t0 + 4.0
        limit = next_start - 0.08
        if limit <= t0 + 0.09 * syls:
            continue

        desired = max(0.30, 0.42 * syls)
        span_end = min(limit, t0 + desired)
        if audio_path:
            islands = _detect_voiced_segments(audio_path, max(0.0, t0 - 0.05), limit)
            island_end = None
            for s_i, e_i in islands:
                if s_i <= t0 + 0.30 and e_i > t0:
                    island_end = e_i
                    break
            if island_end is not None:
                span_end = min(limit, max(t0 + 0.11 * syls, min(island_end, t0 + desired)))
            else:
                span_end = min(limit, t0 + max(0.25, 0.30 * syls))
        if span_end - t0 < 0.09 * syls:
            continue

        tail_slice = [(aligned_data[k].word.strip(), getattr(aligned_data[k], "is_line_end", False)) for k in suffix]
        dist_items = _distribute_words_in_gap(tail_slice, t0, span_end, audio_path=None)
        for offset, item in enumerate(dist_items):
            target_td = aligned_data[first + offset]
            target_td.start = item.start
            target_td.end = item.end
            target_td.confidence = max(float(target_td.confidence or 0.0), 0.70)
            target_td.chars = None
        repaired += 1

    if repaired:
        print(
            f"{ULTRASINGER_HEAD} {green_highlighted('Crushed Line-Tail Repair (Pattern C):')} "
            f"Expanded {blue_highlighted(str(repaired))} crushed line ending(s) onto their vocal island."
        )
    return aligned_data


def repair_ctc_collapsed_runs(
    aligned_data: List[TranscribedData],
    audio_path: Optional[str] = None,
) -> List[TranscribedData]:
    """Repair WhisperX CTC collapses where one word absorbs multiple seconds of audio
    and pushes subsequent words into a crushed micro-burst at the end of the segment.
    """
    if len(aligned_data) < 2:
        return aligned_data

    n_total = len(aligned_data)
    i = 0
    repaired_runs = 0

    while i < n_total:
        w_i = aligned_data[i]
        dur_i = float(w_i.end) - float(w_i.start)
        n_syl_i = count_word_syllables(w_i.word)
        d_thresh_i = max(1.05, 0.25 + 0.35 * n_syl_i)

        # Pattern A: Bloated word w_i immediately followed by a crushed run w_{i+1}..w_m
        if (
            i + 1 < n_total
            and dur_i > d_thresh_i
            and (float(aligned_data[i + 1].start) - float(w_i.end)) <= 0.45
            and _is_ctc_crushed(aligned_data[i + 1])
        ):
            m = i + 1
            while m + 1 < n_total:
                if getattr(aligned_data[m], "is_line_end", False):
                    break
                # Never swallow a bloated word into the crushed run: it must be processed
                # as the anchor w_i of the crushed phrase that follows it (e.g. 'Sol' before
                # 'Iluminaba la esquina').
                if _is_bloated(aligned_data[m + 1]):
                    break
                gap_next = float(aligned_data[m + 1].start) - float(aligned_data[m].end)
                if gap_next <= 0.35 and (
                    _is_ctc_crushed(aligned_data[m + 1])
                    or (
                        m + 2 < n_total
                        and (float(aligned_data[m + 2].start) - float(aligned_data[m + 1].end)) <= 0.25
                        and _is_ctc_crushed(aligned_data[m + 2])
                    )
                ):
                    m += 1
                else:
                    break

            t_win_start = float(w_i.start)
            t_win_end = float(aligned_data[m].end)

            # Subcase A1: w_i is a line end (e.g. 'sol' before 'Iluminaba la esquina') or huge multi-second span (>5.5s)
            if getattr(w_i, "is_line_end", False) or dur_i > 5.50:
                d_cap = min(0.85, 0.22 + 0.25 * n_syl_i)
                w_i.end = round(t_win_start + d_cap, 3)
                w_i.chars = None

                tail_slice = [(aligned_data[k].word.strip(), getattr(aligned_data[k], "is_line_end", False)) for k in range(i + 1, m + 1)]
                tail_syls = sum(count_word_syllables(aligned_data[k].word) for k in range(i + 1, m + 1))
                search_s = float(w_i.end) + 0.10
                search_e = max(search_s + 0.15 * len(tail_slice), t_win_end)

                voiced_islands = _detect_voiced_segments(audio_path, search_s, search_e) if audio_path else []
                if voiced_islands:
                    island_s = voiced_islands[0][0]
                    island_e = min(voiced_islands[-1][1], island_s + max(1.50, 0.42 * tail_syls))
                    dist_items = _distribute_words_in_gap(tail_slice, island_s, island_e, audio_path=None)
                else:
                    dist_items = _distribute_words_in_gap(tail_slice, search_s, min(search_e, search_s + max(1.40, 0.36 * tail_syls)), audio_path=audio_path)

                for offset, item in enumerate(dist_items):
                    target_td = aligned_data[i + 1 + offset]
                    target_td.start = item.start
                    target_td.end = item.end
                    target_td.confidence = max(float(target_td.confidence or 0.0), 0.72)
                    target_td.chars = None
                repaired_runs += 1
                i = m + 1
                continue
            else:
                # Subcase A2: Same-phrase continuous redistribution across consecutive unstressed syllables
                run_slice = [(aligned_data[k].word.strip(), getattr(aligned_data[k], "is_line_end", False)) for k in range(i, m + 1)]
                run_syls = sum(count_word_syllables(aligned_data[k].word) for k in range(i, m + 1))
                voiced_islands = _detect_voiced_segments(audio_path, t_win_start, t_win_end) if audio_path else []
                if voiced_islands:
                    span_e = min(t_win_end, max(t_win_start + 0.22 * run_syls, min(voiced_islands[0][1], t_win_start + max(1.40, 0.40 * run_syls))))
                else:
                    span_e = min(t_win_end, t_win_start + max(1.25, 0.34 * run_syls))

                dist_items = _distribute_words_in_gap(run_slice, t_win_start, span_e, audio_path=None)
                for offset, item in enumerate(dist_items):
                    target_td = aligned_data[i + offset]
                    target_td.start = item.start
                    target_td.end = item.end
                    target_td.confidence = max(float(target_td.confidence or 0.0), 0.72)
                    target_td.chars = None
                repaired_runs += 1
                i = m + 1
                continue

        # Pattern B: Crushed intra-line tail pushed across a large silence gap (> 1.10s) from w_{i-1}
        if (
            i > 0
            and not getattr(aligned_data[i - 1], "is_line_end", False)
            and (float(w_i.start) - float(aligned_data[i - 1].end)) > 1.10
            and _is_ctc_crushed(w_i)
        ):
            m = i
            while m + 1 < n_total and not getattr(aligned_data[m], "is_line_end", False) and _is_ctc_crushed(aligned_data[m + 1]):
                m += 1
            tail_slice = [(aligned_data[k].word.strip(), getattr(aligned_data[k], "is_line_end", False)) for k in range(i, m + 1)]
            tail_syls = sum(count_word_syllables(aligned_data[k].word) for k in range(i, m + 1))
            anchor_s = float(aligned_data[i - 1].end) + 0.04
            anchor_e = min(float(aligned_data[m].end), anchor_s + max(0.85, 0.34 * tail_syls))
            dist_items = _distribute_words_in_gap(tail_slice, anchor_s, anchor_e, audio_path=None)
            for offset, item in enumerate(dist_items):
                target_td = aligned_data[i + offset]
                target_td.start = item.start
                target_td.end = item.end
                target_td.confidence = max(float(target_td.confidence or 0.0), 0.72)
                target_td.chars = None
            repaired_runs += 1
            i = m + 1
            continue

        i += 1

    if repaired_runs > 0:
        print(
            f"{ULTRASINGER_HEAD} {green_highlighted('CTC Collapse Repair:')} "
            f"Redistributed {blue_highlighted(str(repaired_runs))} bloated/crushed phrase run(s) onto natural vocal islands."
        )
    return aligned_data


def run_wav2vec2_ctc_realignment(
    aligned_data: List[TranscribedData],
    audio_path: Optional[str],
    language: Optional[str] = "es",
    device: Optional[str] = "cuda",
) -> List[TranscribedData]:
    """Ecuación I: Segunda pasada neuronal con Wav2Vec2 CTC Forced Aligner (WhisperX).
    Alinea cada palabra y sílaba de lyrics.txt con timestamps exactos por fonema/carácter.
    """
    if not aligned_data or not audio_path or not os.path.isfile(audio_path):
        return aligned_data

    try:
        import torch
        import gc
        import whisperx

        use_device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        lang_code = (language or "es").strip().lower().replace("_", "-").split("-")[0]
        if len(lang_code) > 2:
            try:
                import langcodes
                lang_code = langcodes.find(lang_code).language
            except Exception:
                lang_code = "es"

        model_a, metadata = whisperx.load_align_model(language_code=lang_code, device=use_device)
        audio = whisperx.load_audio(audio_path)
        audio_dur = len(audio) / 16000.0 if len(audio) > 0 else 0.0

        # Build segments from aligned_data grouped by line breaks or pauses >= 0.7s
        raw_segments = []
        curr_words = []
        curr_text = []
        seg_start = aligned_data[0].start
        for idx, td in enumerate(aligned_data):
            curr_words.append(td)
            curr_text.append(td.word.strip())
            is_break = td.is_line_end or (idx < len(aligned_data) - 1 and (aligned_data[idx + 1].start - td.end) >= 0.70) or (idx == len(aligned_data) - 1)
            if is_break:
                seg_end = td.end
                s_pad = max(0.0, float(seg_start) - 0.35)
                e_pad = min(audio_dur - 0.05, float(seg_end) + 0.35)
                if s_pad < audio_dur - 0.20 and e_pad > s_pad + 0.10:
                    raw_segments.append({
                        "start": round(s_pad, 3),
                        "end": round(e_pad, 3),
                        "text": " ".join(curr_text),
                        "words_ref": curr_words,
                    })
                curr_words = []
                curr_text = []
                if idx < len(aligned_data) - 1:
                    seg_start = aligned_data[idx + 1].start

        # Re-align each segment using Wav2Vec2 CTC
        res = whisperx.align(raw_segments, model_a, metadata, audio, use_device, return_char_alignments=True)
        del model_a
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        aligned_res_segments = res.get("segments", [])
        realigned_count = 0
        for s_idx, r_seg in enumerate(aligned_res_segments):
            if s_idx < len(raw_segments):
                ref_tds = raw_segments[s_idx]["words_ref"]
                w_aligned = r_seg.get("words", [])
                seg_chars = r_seg.get("chars", [])

                # Map segment-level characters to individual words
                char_cursor = 0
                for w_info in w_aligned:
                    w_clean = re.sub(r"[^\w]", "", w_info.get("word", "")).lower()
                    w_matched_chars = []
                    while char_cursor < len(seg_chars) and seg_chars[char_cursor].get("char", "") == " ":
                        char_cursor += 1
                    for ch in w_clean:
                        while char_cursor < len(seg_chars) and seg_chars[char_cursor].get("char", "") == " ":
                            char_cursor += 1
                        if char_cursor < len(seg_chars) and seg_chars[char_cursor].get("char", "").lower() == ch:
                            w_matched_chars.append(seg_chars[char_cursor])
                            char_cursor += 1
                    if w_matched_chars:
                        w_info["chars"] = w_matched_chars

                VOWELS = set("aeiouáéíóúüäöüy")
                for w_i, w_info in enumerate(w_aligned):
                    if w_i < len(ref_tds):
                        if "start" in w_info and "end" in w_info:
                            cand_s = float(w_info["start"])
                            cand_e = float(w_info["end"])
                            cand_dur = cand_e - cand_s
                            w_score = float(w_info.get("score", 0.5) or 0.0)
                            n_syl = count_word_syllables(ref_tds[w_i].word)
                            is_line_end = getattr(ref_tds[w_i], "is_line_end", False)

                            # Musical & Phonetic Bound: In singing, line-ending words frequently sustain
                            # vowels/melismas for 2.0-3.5s. Intra-line words are bound by natural coarticulation.
                            if is_line_end:
                                d_max = max(2.50, 0.35 + 0.60 * n_syl)
                            else:
                                d_max = max(1.35, 0.28 + 0.45 * n_syl)

                            max_allowed_dur = 1.85 * d_max
                            if w_i + 1 < len(ref_tds):
                                next_ref_s = float(ref_tds[w_i + 1].start)
                                if next_ref_s > cand_s:
                                    max_allowed_dur = min(max_allowed_dur, max(0.20, next_ref_s - cand_s - 0.04))

                            d_min = max(0.05, 0.08 * n_syl)
                            drift_s = abs(cand_s - float(ref_tds[w_i].start))

                            # Causal Precedence & Lead Vocal Acoustic Anchor:
                            # If duration exceeds max_allowed_dur (e.g. choir or background echo extends the tail),
                            # NEVER discard the onset cand_s! Clamp cand_e to max_allowed_dur so the lead singer's
                            # physical attack and phoneme timestamps are preserved rather than lost to silence.
                            if cand_dur > max_allowed_dur:
                                cand_e = cand_s + max_allowed_dur
                                cand_dur = max_allowed_dur
                                if "chars" in w_info and w_info["chars"]:
                                    clamped_chars = []
                                    for ch in w_info["chars"]:
                                        if "start" in ch and float(ch["start"]) < cand_e:
                                            ch_c = dict(ch)
                                            if "end" in ch_c and float(ch_c["end"]) > cand_e:
                                                ch_c["end"] = cand_e
                                            clamped_chars.append(ch_c)
                                    w_info["chars"] = clamped_chars

                            # Consonant CTC Bridge Repair (Lead Singer vs Backing Choir):
                            # Non-vowel consonants (plosives, fricatives) cannot physically sustain > 0.35s in singing.
                            # If a consonant bridged across to delayed backing vocals, clamp it to natural attack.
                            if "chars" in w_info and w_info["chars"]:
                                c_list = w_info["chars"]
                                for c_idx, ch_item in enumerate(c_list):
                                    ch_char = ch_item.get("char", "").lower()
                                    if ch_char.isalpha() and ch_char not in VOWELS:
                                        c_s = float(ch_item.get("start", 0.0))
                                        c_e = float(ch_item.get("end", c_s))
                                        if c_e - c_s > 0.35:
                                            ch_item["end"] = round(c_s + min(0.12, c_e - c_s), 3)
                                            if c_idx + 1 < len(c_list):
                                                lead_w_end = min(cand_e, c_s + max(0.80, 0.35 * n_syl))
                                                rem = c_list[c_idx + 1:]
                                                s_step = (lead_w_end - float(ch_item["end"])) / max(1, len(rem))
                                                c_cur = float(ch_item["end"])
                                                for r_ch in rem:
                                                    r_ch["start"] = round(c_cur, 3)
                                                    c_cur += s_step
                                                    r_ch["end"] = round(c_cur, 3)
                                                cand_e = lead_w_end
                                                cand_dur = cand_e - cand_s

                            if cand_dur >= d_min and w_score >= 0.22 and drift_s <= 1.80:
                                ref_tds[w_i].start = round(cand_s, 3)
                                ref_tds[w_i].end = round(cand_e, 3)
                                realigned_count += 1
                                if "chars" in w_info and w_info["chars"]:
                                    ref_tds[w_i].chars = w_info["chars"]

        print(f"{ULTRASINGER_HEAD} {green_highlighted('Wav2Vec2 CTC Re-alignment:')} Aligned {blue_highlighted(str(realigned_count))} words to exact acoustic phonemes.")
        return aligned_data
    except Exception as e:
        return aligned_data


def monotonic_dp_align(
    transcribed_data: List[TranscribedData],
    lyrics_words: List[Tuple[str, bool]],
) -> List[Tuple[str, int, int, int, int]]:
    """Time-Monotonic Dynamic Programming Aligner with Temporal Prior & Acoustic Energy Penalty.

    Replaces time-blind SequenceMatcher to prevent repetitive song structures (e.g. Chorus 1 vs
    Chorus 2) from cross-matching across 80+ seconds and deleting sung verses.
    """
    import numpy as np
    from collections import Counter, defaultdict

    whisper_clean = [clean_word_for_matching(td.word) for td in transcribed_data]
    lyrics_clean = [clean_word_for_matching(lw[0]) for lw in lyrics_words]
    w_times = np.array([float(td.start) for td in transcribed_data])
    w_confs = np.array([float(getattr(td, "confidence", 0.5) or 0.5) for td in transcribed_data])

    N = len(whisper_clean)
    M = len(lyrics_clean)
    if N == 0 or M == 0:
        return []

    # 1. Multi-scale anchor identification:
    #    Combines unique unigrams (weight 2.0) with rank-matched bigrams (weight 1.0)
    #    so repetitive choruses and sections with instrumental solos have dense temporal anchors.
    candidate_anchors = []
    w_counts = Counter(whisper_clean)
    l_counts = Counter(lyrics_clean)
    for j, lw in enumerate(lyrics_clean):
        if len(lw) >= 4 and l_counts[lw] == 1 and w_counts.get(lw, 0) == 1:
            i = whisper_clean.index(lw)
            candidate_anchors.append((i, j, 2.0))

    w_bg = defaultdict(list)
    for i in range(N - 1):
        bg = (whisper_clean[i], whisper_clean[i + 1])
        if len(bg[0]) >= 3 and len(bg[1]) >= 3:
            w_bg[bg].append(i)

    l_bg = defaultdict(list)
    for j in range(M - 1):
        bg = (lyrics_clean[j], lyrics_clean[j + 1])
        if len(bg[0]) >= 3 and len(bg[1]) >= 3:
            l_bg[bg].append(j)

    for bg, l_indices in l_bg.items():
        if bg in w_bg:
            w_indices = w_bg[bg]
            for k in range(min(len(l_indices), len(w_indices))):
                candidate_anchors.append((w_indices[k], l_indices[k], 1.0))

    candidate_anchors.sort(key=lambda x: (x[1], x[0]))

    # Longest Monotonically Increasing Subsequence (LIS) on (j, i)
    n_c = len(candidate_anchors)
    dp_lis = [1.0] * n_c
    prev_lis = [-1] * n_c
    for a in range(n_c):
        ia, ja, wa = candidate_anchors[a]
        for b in range(a):
            ib, jb, wb = candidate_anchors[b]
            if ja > jb and ia > ib:
                score = dp_lis[b] + wa
                if score > dp_lis[a]:
                    dp_lis[a] = score
                    prev_lis[a] = b

    best_idx = int(np.argmax(dp_lis)) if n_c > 0 else -1
    lis_anchors = []
    curr = best_idx
    while curr >= 0:
        lis_anchors.append((candidate_anchors[curr][0], candidate_anchors[curr][1]))
        curr = prev_lis[curr]
    lis_anchors.reverse()

    filtered_anchors = [(0, 0)]
    for i_a, j_a in lis_anchors:
        if i_a > filtered_anchors[-1][0] and j_a > filtered_anchors[-1][1]:
            filtered_anchors.append((i_a, j_a))
    filtered_anchors.append((N - 1, M - 1))

    # Piecewise linear interpolation for expected time of lyric word j
    anchor_j = [a[1] for a in filtered_anchors]
    anchor_t = [w_times[a[0]] for a in filtered_anchors]
    expected_t = np.interp(np.arange(M), anchor_j, anchor_t)

    # 2. Score Matrix with Sakoe-Chiba corridor
    D = np.full((N + 1, M + 1), -1e9, dtype=np.float32)
    backtrack = np.zeros((N + 1, M + 1), dtype=np.int8)  # 0: match, 1: del, 2: ins

    D[0, 0] = 0.0
    for i in range(1, N + 1):
        c_del = 0.35 + 0.85 * w_confs[i - 1]
        D[i, 0] = D[i - 1, 0] - c_del
        backtrack[i, 0] = 1

    for j in range(1, M + 1):
        D[0, j] = D[0, j - 1] - 0.50
        backtrack[0, j] = 2

    for i in range(1, N + 1):
        t_i = w_times[i - 1]
        c_del = 0.35 + 0.85 * w_confs[i - 1]
        cw_i = whisper_clean[i - 1]
        if is_scat_or_adlib_pattern(cw_i):
            c_del = 0.10

        for j in range(1, M + 1):
            t_hat = expected_t[j - 1]
            dt = abs(t_i - t_hat)
            if dt > 35.0:
                continue

            cl_j = lyrics_clean[j - 1]
            sim = 1.0 if cw_i == cl_j else character_level_similarity(cw_i, cl_j)

            # Quadratic penalty if outside ±6.0s tolerance window
            p_time = 0.0 if dt <= 6.0 else min(3.0, ((dt - 6.0) / 8.0) ** 2)
            s_match = 2.0 * sim - 1.0 - p_time

            m_score = D[i - 1, j - 1] + s_match
            d_score = D[i - 1, j] - c_del
            i_score = D[i, j - 1] - 0.50

            best = m_score
            bt = 0
            if d_score > best:
                best = d_score
                bt = 1
            if i_score > best:
                best = i_score
                bt = 2

            D[i, j] = best
            backtrack[i, j] = bt

    # 3. Backtrack path
    i, j = N, M
    steps = []
    while i > 0 or j > 0:
        bt = backtrack[i, j]
        if i > 0 and j > 0 and bt == 0:
            cw = whisper_clean[i - 1]
            cl = lyrics_clean[j - 1]
            sim = 1.0 if cw == cl else character_level_similarity(cw, cl)
            steps.append(("match", i - 1, j - 1, sim))
            i -= 1
            j -= 1
        elif i > 0 and (bt == 1 or j == 0):
            steps.append(("del", i - 1, -1, 0.0))
            i -= 1
        else:
            steps.append(("ins", -1, j - 1, 0.0))
            j -= 1
    steps.reverse()

    # 4. Group into standard contiguous opcodes
    opcodes = []
    curr_op = None
    cur_i1, cur_i2 = 0, 0
    cur_j1, cur_j2 = 0, 0

    for step_type, i_idx, j_idx, sim in steps:
        if step_type == "match":
            tag = "equal" if sim >= 0.82 else "replace"
        elif step_type == "del":
            tag = "delete"
        else:
            tag = "insert"

        if curr_op is None:
            curr_op = tag
            cur_i1 = i_idx if i_idx >= 0 else cur_i2
            cur_i2 = cur_i1 + (1 if i_idx >= 0 else 0)
            cur_j1 = j_idx if j_idx >= 0 else cur_j2
            cur_j2 = cur_j1 + (1 if j_idx >= 0 else 0)
        elif tag == curr_op:
            if i_idx >= 0:
                cur_i2 += 1
            if j_idx >= 0:
                cur_j2 += 1
        else:
            opcodes.append((curr_op, cur_i1, cur_i2, cur_j1, cur_j2))
            curr_op = tag
            cur_i1 = cur_i2
            cur_i2 = cur_i1 + (1 if i_idx >= 0 else 0)
            cur_j1 = cur_j2
            cur_j2 = cur_j1 + (1 if j_idx >= 0 else 0)

    if curr_op is not None:
        opcodes.append((curr_op, cur_i1, cur_i2, cur_j1, cur_j2))

    return opcodes


def _split_slice_into_lines(slice_words: List[Tuple[str, bool]]) -> List[List[Tuple[str, bool]]]:
    """Divide una lista de (word, is_line_end) en sub-listas por línea."""
    lines_out: List[List[Tuple[str, bool]]] = []
    curr: List[Tuple[str, bool]] = []
    for item in slice_words:
        curr.append(item)
        if item[1]:
            lines_out.append(curr)
            curr = []
    if curr:
        lines_out.append(curr)
    return lines_out


def align_transcribed_data_with_lyrics(
    transcribed_data: List[TranscribedData],
    lyrics_words: List[Tuple[str, bool]],
    audio_path: Optional[str] = None,
    language: Optional[str] = "es",
    device: Optional[str] = "cuda",
) -> List[TranscribedData]:
    """Align Whisper transcription timestamps with user reference lyrics.

    - Replaces Whisper words with ground-truth words from lyrics.txt.
    - Sets is_line_end = True on the final word of each line.
    - Drops hallucinated words generated by Whisper.
    - Purges non-lexical tarareo/scat intervals and shields repeated lines from stretching over humming.
    - Detaches pre-verse tarareo tails and re-anchors orphan head words (Equations I-IV).
    """
    if not lyrics_words:
        return transcribed_data

    if isinstance(lyrics_words, str):
        lyrics_words = load_lyrics_file(lyrics_words)

    if not transcribed_data:
        print(f"{ULTRASINGER_HEAD} {red_highlighted('Warning:')} No audio transcription data available to align with lyrics.")
        return transcribed_data

    # Pre-filtrado SOTA: Descartar tarareos y ad-libs iniciales ("Ahhh", "Mmmm") que desfasen el t_start
    target_lyrics_words = [lw[0] for lw in lyrics_words]
    filtered_transcribed, quarantined_adlibs = filter_intro_adlibs(
        transcribed_data, target_lyrics_words, similarity_threshold=0.45
    )
    if quarantined_adlibs:
        anchor_w = filtered_transcribed[0].word.strip() if filtered_transcribed else "N/A"
        print(
            f"{ULTRASINGER_HEAD} {gold_highlighted('Intro Ad-lib Purge:')} Quarantined "
            f"{red_highlighted(str(len(quarantined_adlibs)))} non-dictionary intro ad-lib(s) (tarareos). "
            f"Onset anchored to {green_highlighted(anchor_w)}."
        )
    transcribed_data = filtered_transcribed

    # Ecuación III: Purgar bloques de tarareos ('Tararara', 'Ah, mmm, la la la') que no estén en lyrics.txt
    purged_transcribed, purged_scats = purge_instrumental_scat_segments(transcribed_data, target_lyrics_words)
    purged_scat_intervals: List[Tuple[float, float]] = []
    if purged_scats and purged_transcribed:
        for sc in purged_scats:
            s_t = float(getattr(sc, "start", 0.0))
            e_t = float(getattr(sc, "end", s_t))
            if purged_scat_intervals and (s_t - purged_scat_intervals[-1][1]) <= 2.50:
                purged_scat_intervals[-1] = (purged_scat_intervals[-1][0], max(purged_scat_intervals[-1][1], e_t))
            else:
                purged_scat_intervals.append((s_t, e_t))
        print(
            f"{ULTRASINGER_HEAD} {gold_highlighted('Scat/Tarareo Purge (Eq. III):')} "
            f"Purged {red_highlighted(str(len(purged_scats)))} tarareo token(s) across "
            f"{blue_highlighted(str(len(purged_scat_intervals)))} humming interval(s)."
        )
        transcribed_data = purged_transcribed

    whisper_clean = [clean_word_for_matching(td.word) for td in transcribed_data]
    lyrics_clean = [clean_word_for_matching(lw[0]) for lw in lyrics_words]

    # Time-Monotonic Dynamic Programming Alignment (prevents cross-matching repetitive choruses)
    try:
        opcodes = monotonic_dp_align(transcribed_data, lyrics_words)
    except Exception as e_dp:
        print(f"{ULTRASINGER_HEAD} {red_highlighted('DP Aligner fallback to SequenceMatcher:')} {e_dp}")
        matcher = difflib.SequenceMatcher(None, whisper_clean, lyrics_clean)
        opcodes = matcher.get_opcodes()

    aligned_data: List[TranscribedData] = []

    matched_count = 0
    interpolated_count = 0
    dropped_count = len(purged_scats) if purged_scats else 0

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            # Direct match between Whisper and lyrics.txt
            for offset in range(i2 - i1):
                td = transcribed_data[i1 + offset]
                l_word, is_line_end = lyrics_words[j1 + offset]

                # Format word with trailing space
                formatted_word = l_word.rstrip(" ") + " "
                aligned_item = TranscribedData(
                    confidence=td.confidence,
                    word=formatted_word,
                    start=td.start,
                    end=td.end,
                    is_word_end=True,
                    is_line_end=is_line_end,
                    chars=td.chars,
                )
                aligned_data.append(aligned_item)
                matched_count += 1

        elif tag == "replace":
            w_start = transcribed_data[i1].start
            w_end = transcribed_data[i2 - 1].end
            missing_slice = lyrics_words[j1:j2]
            replaced_items = _distribute_words_in_gap(missing_slice, w_start, w_end, audio_path=audio_path)
            aligned_data.extend(replaced_items)
            matched_count += len(replaced_items)

        elif tag == "insert":
            # Words in lyrics.txt that Whisper missed
            num_missing = j2 - j1
            missing_slice = lyrics_words[j1:j2]

            # Ecuación III (Scat Interval Shield): Verificar si las líneas faltantes son repeticiones en lyrics.txt
            # que caen sobre un intervalo donde el cantante en realidad hizo un tarareo ('Tararara') o puente largo.
            # Cubre tanto el caso en que SequenceMatcher ancla la primera repetición (dejando la 2a como insert tras prev_line)
            # como el caso en que SequenceMatcher ancla la segunda repetición (dejando la 1a como insert antes de next_line).
            sub_lines = _split_slice_into_lines(missing_slice)
            prev_line_words = []
            if aligned_data:
                for back_td in reversed(aligned_data):
                    if back_td is not aligned_data[-1] and getattr(back_td, "is_line_end", False):
                        break
                    prev_line_words.append(clean_word_for_matching(back_td.word))
            prev_line_str = " ".join(reversed(prev_line_words))

            next_lyrics_slice = lyrics_words[j2 : min(len(lyrics_words), j2 + max(3, num_missing))]
            next_sub_lines = _split_slice_into_lines(next_lyrics_slice)
            next_line_str = (
                " ".join(clean_word_for_matching(w[0]) for w in next_sub_lines[0])
                if next_sub_lines
                else ""
            )

            prev_end_check = float(aligned_data[-1].end) if aligned_data else 0.0
            next_start_check = float(transcribed_data[i1].start) if i1 < len(transcribed_data) else (prev_end_check + 10.0)
            # Ventana extendida para detectar si el gemelo de la línea repetida es adyacente a un tarareo purgado
            twin_end_idx = min(len(transcribed_data) - 1, i1 + num_missing)
            twin_horizon_end = float(transcribed_data[twin_end_idx].end) if transcribed_data else next_start_check
            overlaps_scat = any(
                (min(max(next_start_check, twin_horizon_end) + 15.0, sc_e) - max(prev_end_check, sc_s)) >= 0.80
                for sc_s, sc_e in purged_scat_intervals
            )
            gap_check = next_start_check - prev_end_check

            if overlaps_scat or gap_check >= 8.0 or (not aligned_data and purged_scat_intervals):
                kept_missing: List[Tuple[str, bool]] = []
                for s_line in sub_lines:
                    s_line_str = " ".join(clean_word_for_matching(w[0]) for w in s_line)
                    matches_prev = bool(prev_line_str and character_level_similarity(s_line_str, prev_line_str) >= 0.82)
                    matches_next = bool(next_line_str and character_level_similarity(s_line_str, next_line_str) >= 0.82)
                    if len(s_line) >= 3 and (matches_prev or matches_next):
                        print(
                            f"{ULTRASINGER_HEAD} {gold_highlighted('Scat Interval Shield (Eq. III):')} "
                            f"Skipped duplicate lyric line '{' '.join(w[0] for w in s_line)}' "
                            f"because one repetition in audio is a tarareo/solo."
                        )
                    else:
                        kept_missing.extend(s_line)
                        if s_line_str:
                            prev_line_str = s_line_str

                missing_slice = kept_missing
                num_missing = len(missing_slice)
                if num_missing == 0:
                    continue

            if not aligned_data:
                # Missing words at the VERY BEGINNING of the song
                whisper_first_start = transcribed_data[i1].start if i1 < len(transcribed_data) else 10.0
                start_bound = 0.0
                if audio_path:
                    try:
                        from modules.Audio.vocal_validator import detect_first_vocal_onset
                        v_onset = detect_first_vocal_onset(audio_path)
                        if v_onset is not None and v_onset < whisper_first_start:
                            start_bound = max(0.0, v_onset - 0.05)
                    except Exception:
                        pass

                if start_bound >= whisper_first_start:
                    start_bound = max(0.0, whisper_first_start - num_missing * 0.35)

                time_start = start_bound
                time_end = max(time_start + 0.15 * num_missing, whisper_first_start - 0.05)
            elif i1 < len(transcribed_data):
                # In-between gap
                prev_end = aligned_data[-1].end
                next_start = transcribed_data[i1].start
                is_prev_line_end = getattr(aligned_data[-1], "is_line_end", False)
                gap_between = next_start - prev_end

                # Excluir intervalos de tarareo purgados para no colocar palabras encima del tarareo
                effective_prev_end = prev_end
                for sc_s, sc_e in purged_scat_intervals:
                    if sc_s >= prev_end - 0.20 and sc_e <= next_start + 0.20:
                        if is_prev_line_end:
                            effective_prev_end = max(effective_prev_end, sc_e + 0.15)

                gap_between = next_start - effective_prev_end

                # Ecuación II (Leading Silence Gate & Voiced Island Anchoring):
                # Si las palabras faltantes incluyen un cierre de línea (ej. frase completa omitida)
                # o existe actividad vocal intermedia en el hueco, permitir que _distribute_words_in_gap
                # busque las islas de voz reales en [effective_prev_end + 0.12, next_start - 0.05].
                contains_line_end = any(is_end for _, is_end in missing_slice)
                if is_prev_line_end and gap_between >= 0.70:
                    if contains_line_end and audio_path:
                        time_start = effective_prev_end + 0.12
                        time_end = max(time_start + 0.15 * num_missing, next_start - 0.05)
                    else:
                        needed_dur = min(gap_between - 0.15, num_missing * 0.35)
                        time_start = max(effective_prev_end + 0.15, next_start - needed_dur)
                        time_end = next_start
                else:
                    time_start = effective_prev_end
                    time_end = next_start
                    if time_end <= time_start:
                        time_end = time_start + num_missing * 0.20
            else:
                # Outro gap (after last Whisper word): clamp within actual audio duration
                time_start = aligned_data[-1].end if aligned_data else 0.0
                audio_dur = 0.0
                if audio_path and os.path.isfile(audio_path):
                    try:
                        import librosa
                        audio_dur = float(librosa.get_duration(path=audio_path))
                    except Exception:
                        pass
                if audio_dur > time_start:
                    time_end = max(time_start + 0.15 * num_missing, audio_dur - 0.40)
                else:
                    time_end = time_start + num_missing * 0.35

            gap_items = _distribute_words_in_gap(missing_slice, time_start, time_end, audio_path=audio_path)
            aligned_data.extend(gap_items)
            interpolated_count += len(gap_items)

        elif tag == "delete":
            # Whisper hallucinated extra words not present in lyrics.txt -> drop them
            dropped_count += (i2 - i1)

    # Reparar colapsos CTC donde una palabra absorbió segundos de audio aplastando las siguientes
    aligned_data = repair_ctc_collapsed_runs(aligned_data, audio_path=audio_path)

    # Patrón C: Expandir colas de frase aplastadas intra-línea hacia su isla vocal
    aligned_data = repair_crushed_line_tails(aligned_data, audio_path=audio_path)

    # Ecuaciones I y II: Re-anclar palabras iniciales huérfanas ('Yo' en 2.0s -> 11.8s)
    # y despegar colas de tarareo pre-estrofa ('Quiero', 'Fueron', 'Yo') antes de Continuity Guard y CTC
    aligned_data = detach_tarareo_and_reanchor_phrases(aligned_data, audio_path=audio_path)

    # Monotonicity check
    for k in range(len(aligned_data) - 1):
        if aligned_data[k + 1].start < aligned_data[k].start:
            aligned_data[k + 1].start = aligned_data[k].end + 0.02
        if aligned_data[k].end > aligned_data[k + 1].start:
            aligned_data[k].end = max(aligned_data[k].start + 0.05, aligned_data[k + 1].start - 0.02)

    # Intra-Line Continuity Guard: Prevent tearing phrases across large silences (> 2.5s)
    for k in range(len(aligned_data) - 1):
        if not aligned_data[k].is_line_end:
            gap = aligned_data[k + 1].start - aligned_data[k].end
            if gap > 2.5:
                aligned_data[k].is_line_end = True
                print(
                    f"{ULTRASINGER_HEAD} {gold_highlighted('Continuity Guard:')} Split line at "
                    f"'{aligned_data[k].word.strip()}' due to {gap:.2f}s gap before '{aligned_data[k + 1].word.strip()}'."
                )

    total_lines = sum(1 for _, is_end in lyrics_words if is_end)
    print(
        f"{ULTRASINGER_HEAD} Aligned reference lyrics with audio: "
        f"{green_highlighted(str(len(aligned_data)))} words, "
        f"{blue_highlighted(str(total_lines))} lines "
        f"(matched: {matched_count}, interpolated: {interpolated_count}, dropped hallucinations: {dropped_count})"
    )

    if audio_path and os.path.isfile(audio_path):
        aligned_data = run_wav2vec2_ctc_realignment(
            aligned_data,
            audio_path=audio_path,
            language=language,
            device=device,
        )
        aligned_data = repair_crushed_line_tails(aligned_data, audio_path=audio_path)

    return aligned_data
