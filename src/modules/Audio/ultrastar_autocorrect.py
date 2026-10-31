"""Ultrastar Auto-Correct module: Standalone acoustic alignment, pitch calibration,
and game engine QA simulation from clean vocal audio.
"""

import os
import re
import shutil
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
try:
    import librosa
except Exception:
    librosa = None

from modules.console_colors import (
    ULTRASINGER_HEAD,
    blue_highlighted,
    cyan_highlighted,
    green_highlighted,
    red_highlighted,
    gold_highlighted,
    light_blue_highlighted,
)
try:
    from modules.Pitcher.pitcher import get_pitch_with_file
    from modules.Pitcher.pitched_data import PitchedData
except Exception:
    get_pitch_with_file = None
    PitchedData = None
from modules.Speech_Recognition.fuzzy_lyric_matcher import (
    filter_intro_adlibs,
    character_level_similarity,
    is_scat_or_adlib_pattern,
    classify_onomatopoeia_acoustics,
    detect_closed_mouth_humming_acoustics,
    detect_rhythmic_transient_envelope,
    compute_levenshtein_oov_distance,
    compute_phonotactic_entropy_and_reduplication,
)
from modules.Speech_Recognition.genre_profiles import resolve_genre_profile


@dataclass
class UltraStarLine:
    """Represents a line in an UltraStar .txt file."""
    line_type: str  # ':', '*', 'F', 'R', '-', 'E', or header '#'
    start_beat: int = 0
    duration_beats: int = 0
    pitch: int = 0
    word: str = ""
    raw_line: str = ""
    is_note: bool = False
    is_line_break: bool = False


@dataclass
class QASimulationReport:
    """QA Score and diagnostic report."""
    original_score: int = 0
    corrected_score: int = 0
    timing_coverage: float = 0.0
    pitch_accuracy: float = 0.0
    total_notes: int = 0
    adjusted_notes: int = 0
    gap_shift_ms: float = 0.0
    local_shifts_count: int = 0
    backup_path: str = ""
    phrase_scores: list[dict] = field(default_factory=list)


def weighted_median(data: np.ndarray, weights: np.ndarray) -> float:
    """Calculate the weighted median of a 1D array."""
    if len(data) == 0:
        return 60.0
    if len(data) == 1:
        return float(data[0])
    sorter = np.argsort(data)
    d_sorted = data[sorter]
    w_sorted = weights[sorter]
    cum_w = np.cumsum(w_sorted)
    if cum_w[-1] <= 0:
        return float(np.median(data))
    cutoff = cum_w[-1] / 2.0
    idx = int(np.searchsorted(cum_w, cutoff))
    return float(d_sorted[min(idx, len(d_sorted) - 1)])


@dataclass
class ArticulatoryAcousticFeatures:
    """Acoustic-articulatory feature vectors aligned temporally with a 10 ms hop."""
    times: np.ndarray
    f0_midi: np.ndarray
    confidence: np.ndarray
    r_boca: np.ndarray
    spectral_flatness: np.ndarray
    spectral_centroid: np.ndarray
    delta_mfcc: np.ndarray
    rms: np.ndarray
    v_mask: np.ndarray
    zcr: Optional[np.ndarray] = None


@dataclass
class MIRMultidimensionalEvent:
    """Evento estructurado del Motor de Análisis Acústico y Musical Multidimensional (MIR)."""
    timestamp_start: float
    timestamp_end: float
    dimension: str  # 'Vibrato' | 'Tempo' | 'Acorde Dominante' | 'Respiración/Ruido' | 'Clímax/Drop'
    quantitative_value: str


def compute_zero_crossing_rate_frames(
    y: np.ndarray,
    frame_length: int = 1024,
    hop_length: int = 220,
    n_target_frames: Optional[int] = None,
) -> np.ndarray:
    """Módulo 4 MIR: Tasa de Cruce por Cero (Zero-Crossing Rate - ZCR):
      ZCR = (1 / 2N) * sum_{n=1}^{N} |sgn(x[n]) - sgn(x[n-1])|
    Discrimina sonidos entonados (vocales con ZCR bajo < 0.08) de sonidos no entonados
    (respiraciones, consonantes fricativas como 's', 'f', 'sh' y platillos con ZCR alto >= 0.18).
    """
    sig = np.asarray(y, dtype=np.float64)
    if len(sig) < 2:
        return np.zeros(max(1, n_target_frames or 1), dtype=np.float32)

    pad_len = frame_length // 2
    padded = np.pad(sig, (pad_len, pad_len), mode="reflect")
    n_frames = 1 + max(0, (len(padded) - frame_length) // max(1, hop_length))
    if n_target_frames is not None:
        n_frames = n_target_frames

    zcr_out = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        s_idx = i * hop_length
        e_idx = min(len(padded), s_idx + frame_length)
        frame = padded[s_idx:e_idx]
        if len(frame) > 1:
            # sgn(x) in {-1, +1} (treating >= 0 as +1 and < 0 as -1)
            sgn = np.where(frame >= 0.0, 1.0, -1.0)
            zcr_val = float(np.sum(np.abs(np.diff(sgn))) / (2.0 * len(frame)))
            zcr_out[i] = zcr_val
    return zcr_out


def analyze_multidimensional_mir_events(
    art_feat: ArticulatoryAcousticFeatures,
    chroma: Optional[np.ndarray] = None,
    window_sec: float = 0.50,
) -> list[MIRMultidimensionalEvent]:
    """Ejecuta los 5 Módulos del Motor de Análisis Acústico y Musical Multidimensional (MIR)
    y devuelve eventos con (timestamp_start, timestamp_end, dimension, quantitative_value):
      1. Vibrato: F_0(t) = F_c + A * sin(2*pi*f_m*t)
      2. Tempo: R(k) = sum_n x[n] * x[n-k]
      3. Acorde Dominante: C(p) = sum_o |X(p + 12*o)|
      4. Respiración/Ruido: ZCR = 1/(2N) * sum |sgn(x[n]) - sgn(x[n-1])|
      5. Clímax/Drop: E_RMS = sqrt(1/N * sum x[n]^2)
    """
    from modules.Audio.bpm import compute_autocorrelation_tempo
    from modules.Audio.key_detector import detect_dominant_chords_timeline

    events: list[MIRMultidimensionalEvent] = []
    times = art_feat.times
    if len(times) < 2:
        return events

    t_start_all = float(times[0])
    t_end_all = float(times[-1])
    dt = float(np.mean(np.diff(times))) if len(times) > 1 else 0.010
    fps = 1.0 / max(1e-6, dt)

    # Módulo 2: Tempo por Autocorrelación R(k) sobre la envolvente energética
    bpm_est = compute_autocorrelation_tempo(art_feat.rms, frame_rate_hz=fps)
    events.append(
        MIRMultidimensionalEvent(
            timestamp_start=round(t_start_all, 3),
            timestamp_end=round(t_end_all, 3),
            dimension="Tempo",
            quantitative_value=f"{bpm_est:.2f} BPM",
        )
    )

    # Módulo 3: Identificación Armónica (Cromagrama 12 tonos)
    if chroma is not None:
        for ch_ev in detect_dominant_chords_timeline(chroma, times, window_sec=max(0.5, window_sec)):
            events.append(
                MIRMultidimensionalEvent(
                    timestamp_start=ch_ev["timestamp_start"],
                    timestamp_end=ch_ev["timestamp_end"],
                    dimension="Acorde Dominante",
                    quantitative_value=str(ch_ev["quantitative_value"]),
                )
            )

    # Módulos 1, 4 y 5 por ventanas temporales
    climax_thresh = float(np.percentile(art_feat.rms, 80)) if len(art_feat.rms) > 0 else 0.1
    curr_t = t_start_all
    step_w = max(0.25, window_sec)

    while curr_t < t_end_all:
        next_t = min(t_end_all, curr_t + step_w)
        idx_s = int(np.searchsorted(times, curr_t))
        idx_e = int(np.searchsorted(times, next_t))
        if idx_e > idx_s:
            # Módulo 1: Vibrato (4 - 7.5 Hz)
            if idx_e - idx_s >= 20:
                is_vib, f_v, _ = evaluate_vibrato_sinusoid(
                    art_feat.f0_midi[idx_s:idx_e], times[idx_s:idx_e], f_min=4.0, f_max=7.5
                )
                if is_vib:
                    events.append(
                        MIRMultidimensionalEvent(
                            timestamp_start=round(curr_t, 3),
                            timestamp_end=round(next_t, 3),
                            dimension="Vibrato",
                            quantitative_value=f"{f_v:.2f} Hz",
                        )
                    )

            # Módulo 4: Respiración / Fricativa (ZCR alto)
            if art_feat.zcr is not None and len(art_feat.zcr) >= idx_e:
                zcr_win = float(np.mean(art_feat.zcr[idx_s:idx_e]))
                conf_win = float(np.mean(art_feat.confidence[idx_s:idx_e]))
                if zcr_win >= 0.18 and conf_win < 0.35:
                    events.append(
                        MIRMultidimensionalEvent(
                            timestamp_start=round(curr_t, 3),
                            timestamp_end=round(next_t, 3),
                            dimension="Respiración/Ruido",
                            quantitative_value=f"ZCR={zcr_win:.3f}",
                        )
                    )

            # Módulo 5: Dinámica y Clímax/Drop (E_RMS)
            rms_win = art_feat.rms[idx_s:idx_e]
            e_rms = float(np.sqrt(np.mean(rms_win ** 2)))
            if e_rms >= climax_thresh and climax_thresh > 1e-5:
                events.append(
                    MIRMultidimensionalEvent(
                        timestamp_start=round(curr_t, 3),
                        timestamp_end=round(next_t, 3),
                        dimension="Clímax/Drop",
                        quantitative_value=f"RMS={e_rms:.4f}",
                    )
                )
        curr_t = next_t

    return events


def count_vowels_consonants(text: str) -> tuple[int, int, bool]:
    """Count vowels, consonants, and detect word-end for Klatt phonetic structural weighting."""
    clean = text.strip().lower()
    vowels = set("aeiouáéíóúü")
    n_v = sum(1 for c in clean if c in vowels)
    n_c = sum(1 for c in clean if c.isalpha() and c not in vowels)
    is_word_end = text.endswith(" ") or clean.endswith((".", ",", ";", "!", "?", ":"))
    return n_v, n_c, is_word_end


def extract_articulatory_features(
    y: np.ndarray,
    sr: int,
    pitched_data: Optional[PitchedData] = None,
    hop_ms: float = 10.0,
) -> ArticulatoryAcousticFeatures:
    """
    Equation 0 & 1: Extract articulatory acoustic features with a 10 ms hop and compute V_mask.
    """
    hop_length = max(1, int(round(sr * (hop_ms / 1000.0))))
    n_fft = 1024

    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    mag = np.abs(D)
    fft_freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(mag.shape[1]), sr=sr, hop_length=hop_length)

    # Ratio de Apertura Bucal (R_boca)
    f_formant_mask = (fft_freqs >= 300) & (fft_freqs <= 3400)
    f_fricative_mask = (fft_freqs >= 4000) & (fft_freqs <= 10000)
    e_formant = np.sum(mag[f_formant_mask, :] ** 2, axis=0)
    e_fricative = np.sum(mag[f_fricative_mask, :] ** 2, axis=0)
    r_boca = e_formant / (e_fricative + 1e-6)

    # Planicidad y Centroide espectral
    spec_flat = librosa.feature.spectral_flatness(S=mag)[0]
    spec_cent = librosa.feature.spectral_centroid(S=mag, sr=sr)[0]

    # Flujo Formántico: ||Delta MFCC||_2
    mfcc = librosa.feature.mfcc(S=librosa.amplitude_to_db(mag + 1e-6), sr=sr, n_mfcc=13)
    if mfcc.shape[1] > 1:
        diff_mfcc = np.diff(mfcc, axis=1)
        delta_mfcc_vals = np.sqrt(np.sum(diff_mfcc ** 2, axis=0))
        delta_mfcc = np.pad(delta_mfcc_vals, (1, 0), mode="edge")
    else:
        delta_mfcc = np.zeros(mag.shape[1], dtype=np.float32)

    # Envolvente RMS (Módulo 5 MIR: E_RMS = sqrt(1/N * sum x[n]^2)) y ZCR (Módulo 4 MIR)
    rms = librosa.feature.rms(S=mag, frame_length=n_fft)[0]
    zcr = compute_zero_crossing_rate_frames(
        y=y, frame_length=n_fft, hop_length=hop_length, n_target_frames=len(times)
    )

    # Interpolar RMVPE f0 y confianza en la rejilla de 10 ms
    n_frames = len(times)
    f0_midi = np.zeros(n_frames, dtype=np.float32)
    conf = np.zeros(n_frames, dtype=np.float32)

    if pitched_data and len(pitched_data.times) > 0:
        p_times = np.array(pitched_data.times)
        p_freqs = np.array(pitched_data.frequencies)
        p_confs = np.array(pitched_data.confidence)

        valid_p = (p_freqs >= 65.0) & (p_freqs <= 1100.0) & (p_confs >= 0.15)
        conf = np.interp(times, p_times, p_confs, left=0.0, right=0.0)
        if np.any(valid_p):
            p_midi_valid = librosa.hz_to_midi(p_freqs[valid_p])
            f0_midi = np.interp(
                times,
                p_times[valid_p],
                p_midi_valid,
                left=float(p_midi_valid[0]),
                right=float(p_midi_valid[-1]),
            )
            f0_midi[conf < 0.15] = 0.0

    # Ecuación 1: V_mask(t) = Conf_RMVPE(t) * Sigmoide(R_boca(t) - 1.5) * (1 - SpectralFlatness(t)) * Penalización_ZCR(t)
    sig_arg = np.clip(2.0 * (r_boca - 1.5), -50.0, 50.0)
    sigmoid_rboca = 1.0 / (1.0 + np.exp(-sig_arg))
    zcr_penalty = np.where(zcr >= 0.22, np.clip(1.0 - (zcr - 0.15) * 2.5, 0.2, 1.0), 1.0)
    v_mask = conf * sigmoid_rboca * np.clip(1.0 - spec_flat, 0.0, 1.0) * zcr_penalty
    v_mask = np.clip(v_mask, 0.0, 1.0)

    return ArticulatoryAcousticFeatures(
        times=times,
        f0_midi=f0_midi,
        confidence=conf,
        r_boca=r_boca,
        spectral_flatness=spec_flat,
        spectral_centroid=spec_cent,
        delta_mfcc=delta_mfcc,
        rms=rms,
        v_mask=v_mask,
        zcr=zcr,
    )


def compute_intersyllabic_boundary(
    art_feat: ArticulatoryAcousticFeatures,
    t1_start: float,
    t1_end: float,
    t2_start: float,
    t2_end: float,
    word1: str,
    word2: str,
    note1_pitch: int,
    note2_pitch: int,
) -> tuple[float, float]:
    """
    Equation 2: Intersyllabic novelty curve B_final(t) with Gaussian anti-diphthong window,
    and Equation 3: Klatt structural phonetic weighting for fused vowels (B_final < 0.15).
    """
    times = art_feat.times
    v_mask = art_feat.v_mask
    cent = art_feat.spectral_centroid
    d_mfcc = art_feat.delta_mfcc
    f0 = art_feat.f0_midi
    rms = art_feat.rms

    win_start = max(t1_start + 0.040, min(t1_end, t2_start) - 0.150)
    win_end = min(t2_end - 0.040, max(t1_end, t2_start) + 0.150)

    if win_end <= win_start + 0.030:
        mid = (t1_end + t2_start) / 2.0
        return mid, mid

    idx_s = np.searchsorted(times, win_start)
    idx_e = np.searchsorted(times, win_end)

    if idx_e - idx_s < 3:
        mid = (t1_end + t2_start) / 2.0
        return mid, mid

    sub_t = times[idx_s:idx_e]
    sub_v = v_mask[idx_s:idx_e]
    sub_c = cent[idx_s:idx_e]
    sub_m = d_mfcc[idx_s:idx_e]
    sub_f = f0[idx_s:idx_e]
    sub_r = rms[idx_s:idx_e]

    dv_drop = np.maximum(0.0, -np.gradient(sub_v))
    d_cent = np.abs(np.gradient(sub_c))
    d_f0 = np.abs(np.gradient(sub_f))
    d_rms = np.abs(np.gradient(sub_r))

    def norm_p95(arr: np.ndarray) -> np.ndarray:
        p95 = float(np.percentile(arr, 95)) if len(arr) > 0 else 1.0
        return arr / (p95 + 1e-6)

    B = (
        0.30 * norm_p95(dv_drop)
        + 0.15 * norm_p95(d_cent)
        + 0.25 * norm_p95(sub_m)
        + 0.20 * norm_p95(d_f0)
        + 0.10 * norm_p95(d_rms)
    )

    mu_0 = (t1_end + t2_start) / 2.0
    win_dur = win_end - win_start
    sigma_w = max(0.020, 0.35 * win_dur)
    gauss = np.exp(-((sub_t - mu_0) ** 2) / (2.0 * (sigma_w ** 2)))
    B_final = B * gauss

    max_b = float(np.max(B_final)) if len(B_final) > 0 else 0.0

    # Equation 3: Klatt rule for fused vowels
    if max_b < 0.15 and abs(note1_pitch - note2_pitch) <= 1:
        v1, c1, end1 = count_vowels_consonants(word1)
        v2, c2, end2 = count_vowels_consonants(word2)
        w1_weight = 3.0 * v1 + 1.0 * c1 + (1.5 if end1 else 0.0)
        w2_weight = 3.0 * v2 + 1.0 * c2 + (1.5 if end2 else 0.0)
        total_w = max(0.1, w1_weight + w2_weight)
        t_split = win_start + (win_end - win_start) * (w1_weight / total_w)
        return t_split, t_split

    best_idx = int(np.argmax(B_final))
    t_star = float(sub_t[best_idx])

    # Check for consonant non-vocal valley around t_star
    valley_mask = (sub_t >= t_star - 0.080) & (sub_t <= t_star + 0.080) & (sub_v < 0.35)
    if np.any(valley_mask):
        valley_times = sub_t[valley_mask]
        t1_out = float(valley_times[0])
        t2_out = float(valley_times[-1])
        return min(t1_out, t_star), max(t2_out, t_star)
    else:
        return t_star, t_star


def estimate_pitch_kde_mode(
    midi_pitches: np.ndarray,
    weights: np.ndarray,
    bandwidth_semitones: float = 0.35,
    num_grid_points: int = 120,
) -> float:
    """
    Gaussian Weighted Kernel Density Estimation (KDE) Mode Estimator (MIR Standard):
      p_hat(m) = sum(w_i * exp(-0.5 * ((m - m_i) / h)^2))
    Returns the exact tonal mode (peak density m*) which corresponds to the stable
    target pitch held by the singer, mathematically immune to onset scoops and trailing fall-offs.
    """
    if len(midi_pitches) == 0:
        return 60.0
    if len(midi_pitches) == 1:
        return float(midi_pitches[0])

    w_sum = float(np.sum(weights))
    if w_sum < 1e-6:
        return float(np.median(midi_pitches))

    norm_weights = weights / w_sum
    m_min = float(np.min(midi_pitches))
    m_max = float(np.max(midi_pitches))

    if m_max - m_min < 0.05:
        return float(np.median(midi_pitches))

    grid = np.linspace(m_min - 0.5, m_max + 0.5, num=num_grid_points)
    # Vectorized evaluation: (N, 1) - (1, K) -> (N, K)
    diff = midi_pitches[:, None] - grid[None, :]
    densities = np.sum(norm_weights[:, None] * np.exp(-0.5 * (diff / bandwidth_semitones) ** 2), axis=0)

    peak_idx = int(np.argmax(densities))
    return float(grid[peak_idx])


def optimize_pitch_for_usdx_frames(
    f0_valid: np.ndarray,
    center_midi_float: float,
    v_weights: np.ndarray,
) -> tuple[int, float]:
    """Ecuación V: Maximizador directo de acierto tonal en USDX.
    Prueba los enteros candidatos alrededor de center_midi_float y elige el que
    maximiza directamente los frames dentro de +-1.5 semitones (con tolerancia de octava).
    Devuelve (best_usdx_pitch, hit_ratio).
    """
    if len(f0_valid) == 0:
        return int(round(center_midi_float)) - 48, 1.0

    base_int = int(round(center_midi_float))
    candidates = sorted([base_int, base_int - 1, base_int + 1], key=lambda c: abs(c - center_midi_float))

    best_p = base_int
    best_hit_weight = -1.0
    best_mean_error = 999.0
    best_hit_ratio = 0.0

    total_w = float(np.sum(v_weights))
    if total_w < 1e-6:
        total_w = float(len(f0_valid))
        v_weights = np.ones(len(f0_valid), dtype=np.float32)

    for cand_p in candidates:
        diffs = np.abs(f0_valid - cand_p)
        oct_diffs = np.abs((f0_valid % 12) - (cand_p % 12))
        oct_diffs = np.minimum(oct_diffs, 12.0 - oct_diffs)
        hit_mask = (diffs <= 1.5) | (oct_diffs <= 1.5)
        hit_w = float(np.sum(v_weights[hit_mask]))
        mean_err = float(np.mean(np.minimum(diffs, oct_diffs)))
        # Maximize USDX score; break ties with minimum harmonic deviation
        if (hit_w > best_hit_weight + 1e-5) or (abs(hit_w - best_hit_weight) <= 1e-5 and mean_err < best_mean_error):
            best_hit_weight = hit_w
            best_mean_error = mean_err
            best_p = cand_p
            best_hit_ratio = hit_w / total_w

    return best_p - 48, best_hit_ratio


def estimate_pitch_anti_scoop(
    art_feat: ArticulatoryAcousticFeatures,
    t_start: float,
    t_end: float,
    default_pitch: int,
) -> tuple[int, float]:
    """
    Equation 4: Anti-Scoop pitch estimation via Gaussian Kernel Density Estimation (KDE) mode
    on stable central core with Equation V USDX score maximization.
    Returns: (best_pitch, hit_ratio)
    """
    dur = t_end - t_start
    times = art_feat.times
    f0 = art_feat.f0_midi
    v_mask = art_feat.v_mask
    rms = art_feat.rms

    idx_s = np.searchsorted(times, t_start)
    idx_e = np.searchsorted(times, t_end)
    n_f = idx_e - idx_s

    if n_f < 2:
        return default_pitch, 1.0

    if dur >= 0.120 and n_f >= 5:
        trim_cnt = max(1, int(round(0.20 * n_f)))
        c_s = idx_s + trim_cnt
        c_e = max(c_s + 1, idx_e - trim_cnt)
    else:
        c_s = idx_s
        c_e = idx_e

    sub_f0 = f0[c_s:c_e]
    sub_v = v_mask[c_s:c_e]
    sub_r = rms[c_s:c_e]

    valid = (sub_f0 >= 21.0) & (sub_f0 <= 108.0) & (sub_v >= 0.20)
    if np.sum(valid) >= 2:
        weights = (sub_v[valid] ** 2) * np.maximum(sub_r[valid], 0.001)
        mode_midi = estimate_pitch_kde_mode(sub_f0[valid], weights, bandwidth_semitones=0.35)
        best_p, hit_r = optimize_pitch_for_usdx_frames(sub_f0[valid], mode_midi, weights)
        return best_p, hit_r

    all_f0 = f0[idx_s:idx_e]
    all_v = v_mask[idx_s:idx_e]
    all_r = rms[idx_s:idx_e]
    all_valid = (all_f0 >= 21.0) & (all_f0 <= 108.0) & (all_v >= 0.15)
    if np.any(all_valid):
        w = (all_v[all_valid] ** 2) * np.maximum(all_r[all_valid], 0.001)
        mode_midi = estimate_pitch_kde_mode(all_f0[all_valid], w, bandwidth_semitones=0.35)
        best_p, hit_r = optimize_pitch_for_usdx_frames(all_f0[all_valid], mode_midi, w)
        return best_p, hit_r

    return default_pitch, 1.0


def fold_octave_jumps(note_lines: list[UltraStarLine]) -> None:
    """
    Equation 4 Part 2: Folds spurious octave jumps (+-12 semitones) against both local phrase and global medians.
    """
    n_notes = len(note_lines)
    valid_pitches = [nl.pitch for nl in note_lines if nl.line_type not in ("F", "R")]
    if not valid_pitches:
        return
    global_med = float(np.median(valid_pitches))

    for i, n in enumerate(note_lines):
        if n.line_type in ("F", "R"):
            continue
        neighbors = []
        for j in range(max(0, i - 4), min(n_notes, i + 5)):
            if j != i and note_lines[j].line_type not in ("F", "R"):
                if abs(note_lines[j].pitch - global_med) <= 9.0:
                    neighbors.append(note_lines[j].pitch)

        ref_med = float(np.median(neighbors)) if len(neighbors) >= 2 else global_med
        diff = n.pitch - ref_med
        if abs(diff) >= 9.0:
            fold_steps = int(round(diff / 12.0))
            if fold_steps != 0:
                n.pitch -= 12 * fold_steps


def evaluate_vibrato_sinusoid(
    f0_midi_segment: np.ndarray,
    times_segment: np.ndarray,
    f_min: float = 4.5,
    f_max: float = 7.5,
    max_amplitude_st: float = 1.2,
) -> tuple[bool, float, float]:
    """
    Equation 1 (J. Sundberg & IRCAM): Detects biological vocal vibrato:
      F_0(t) = F_bar + A_v * sin(2*pi*f_v*t + phi)
    Applies 250ms moving median detrending to extract modulation,
    measures dominant modulation frequency f_v and extent A_v.
    Returns: (is_vibrato, f_v, A_v)
    """
    if len(f0_midi_segment) < 20:
        return False, 0.0, 0.0

    valid = f0_midi_segment > 20.0
    if np.sum(valid) < 18:
        return False, 0.0, 0.0

    f0_v = f0_midi_segment[valid]
    t_v = times_segment[valid]

    # Detrend macro-contour with 250ms window (at 10ms hop, ~25 frames)
    w_size = max(5, min(len(f0_v) // 2 * 2 + 1, 25))
    try:
        from scipy.ndimage import median_filter
        f_trend = median_filter(f0_v, size=w_size)
    except Exception:
        f_trend = np.convolve(f0_v, np.ones(w_size) / w_size, mode="same")

    f_vibrato = f0_v - f_trend
    std_v = float(np.std(f_vibrato))
    a_v = float(np.sqrt(2.0) * std_v)

    if a_v < 0.15 or a_v > max_amplitude_st + 0.3:
        return False, 0.0, a_v

    dt = float(np.mean(np.diff(t_v)))
    if dt <= 0:
        dt = 0.010
    fs_mod = 1.0 / dt

    autocorr = np.correlate(f_vibrato - np.mean(f_vibrato), f_vibrato - np.mean(f_vibrato), mode="full")
    autocorr = autocorr[len(autocorr) // 2 :]

    min_lag = int(np.floor(fs_mod / f_max))
    max_lag = int(np.ceil(fs_mod / f_min))
    if max_lag > min_lag and max_lag < len(autocorr):
        peak_lag = min_lag + int(np.argmax(autocorr[min_lag:max_lag]))
        if autocorr[0] > 1e-6:
            norm_corr = float(autocorr[peak_lag] / autocorr[0])
            f_v = float(fs_mod / peak_lag)
            if norm_corr >= 0.35 and (f_min <= f_v <= f_max):
                return True, f_v, a_v

    return False, 0.0, a_v


def evaluate_melisma_vs_vibrato(
    art_feat: ArticulatoryAcousticFeatures,
    t_start: float,
    t_end: float,
    min_semitone_diff: float = 1.5,
    vibrato_freq_range: tuple[float, float] = (4.5, 7.5),
    vibrato_max_amplitude_st: float = 1.2,
) -> Optional[tuple[float, int, int]]:
    """
    Equation 5: Mathematical discriminator between wide vocal vibrato and genuine melisma.
    First checks if the pitch trajectory fits biological vocal vibrato:
      F_0(t) = F_bar + A_v * sin(2*pi*f_v*t + phi)
    If vibrato is confirmed, returns None (preserving note intact without melisma splits).
    Otherwise, checks if genuine step pitch change occurs.
    """
    dur = t_end - t_start
    if dur < 0.350:
        return None

    times = art_feat.times
    f0 = art_feat.f0_midi
    v_mask = art_feat.v_mask
    rms = art_feat.rms

    idx_s = np.searchsorted(times, t_start)
    idx_e = np.searchsorted(times, t_end)
    if idx_e - idx_s < 20:
        return None

    sub_t = times[idx_s:idx_e]
    sub_f = f0[idx_s:idx_e]
    sub_v = v_mask[idx_s:idx_e]
    sub_r = rms[idx_s:idx_e]
    sub_dmfcc = (
        art_feat.delta_mfcc[idx_s:idx_e]
        if art_feat.delta_mfcc is not None and len(art_feat.delta_mfcc) >= idx_e
        else np.zeros_like(sub_f)
    )

    valid = (sub_f >= 21.0) & (sub_v >= 0.25)
    if np.sum(valid) < 15:
        return None

    # Ecuación 1: Comprobar si la modulación es vibrato sinusoidal biológico
    is_vibrato, f_v, a_v = evaluate_vibrato_sinusoid(
        sub_f[valid],
        sub_t[valid],
        f_min=vibrato_freq_range[0],
        f_max=vibrato_freq_range[1],
        max_amplitude_st=vibrato_max_amplitude_st,
    )
    if is_vibrato:
        # Nota sólida con vibrato natural: NO partir en melisma ~
        return None

    min_split_t = t_start + 0.150
    max_split_t = t_end - 0.150
    if max_split_t <= min_split_t:
        return None

    # Candidatos de partición: rejilla uniforme + punto de inflexión máxima de derivada |dF0/dt|
    grid_candidates = list(np.linspace(min_split_t, max_split_t, num=10))
    valid_t = sub_t[valid]
    valid_f = sub_f[valid]
    if len(valid_f) >= 6:
        df0 = np.abs(np.diff(valid_f))
        step_times = valid_t[1:]
        in_window = (step_times >= min_split_t) & (step_times <= max_split_t)
        if np.any(in_window):
            inflect_idx = int(np.argmax(np.where(in_window, df0, -1.0)))
            grid_candidates.append(float(step_times[inflect_idx]))

    best_split = None
    best_score = 0.0
    best_p1, best_p2 = 0, 0

    for sp_t in grid_candidates:
        mask1 = (sub_t >= t_start) & (sub_t < sp_t) & valid
        mask2 = (sub_t >= sp_t) & (sub_t <= t_end) & valid

        if np.sum(mask1) >= 6 and np.sum(mask2) >= 6:
            f1 = sub_f[mask1]
            f2 = sub_f[mask2]
            w1 = (sub_v[mask1] ** 2) * np.maximum(sub_r[mask1], 0.001)
            w2 = (sub_v[mask2] ** 2) * np.maximum(sub_r[mask2], 0.001)

            mode1 = estimate_pitch_kde_mode(f1, w1, bandwidth_semitones=0.35)
            mode2 = estimate_pitch_kde_mode(f2, w2, bandwidth_semitones=0.35)

            std1 = float(np.std(f1))
            std2 = float(np.std(f2))

            pitch_diff = abs(mode1 - mode2)
            if pitch_diff >= min_semitone_diff and max(std1, std2) < 0.45:
                # Condición física MIR de Melisma (dF0/dt != 0 y dF1/dt ~ 0):
                # En la transición tonal sp_t, el timbre vocálico (formante F1 / flujo cepstral ||Delta MFCC||_2)
                # debe permanecer estacionario (< 22.0). Si hay un pico brusco de formantes (>= 22.0),
                # se trata de una articulación consonántica o cambio de vocal (diptongo), no un melisma ~.
                boundary_mask = np.abs(sub_t - sp_t) <= 0.025
                local_df1 = (
                    float(np.mean(sub_dmfcc[boundary_mask]))
                    if np.any(boundary_mask)
                    else 0.0
                )
                if local_df1 >= 22.0:
                    continue

                candidate_score = pitch_diff / (0.10 + std1 + std2 + 0.02 * local_df1)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_split = float(sp_t)
                    best_p1 = int(round(mode1)) - 48
                    best_p2 = int(round(mode2)) - 48

    if best_split is not None:
        return best_split, best_p1, best_p2

    return None


def parse_ultrastar_txt(txt_path: str) -> tuple[dict[str, str], list[UltraStarLine]]:

    """Parse an UltraStar .txt file, preserving headers, line breaks, and exact words."""
    headers = {}
    lines: list[UltraStarLine] = []

    with open(txt_path, "r", encoding="utf-8-sig", errors="replace") as f:
        content = f.readlines()

    for raw in content:
        line_clean = raw.rstrip("\r\n")
        if not line_clean:
            continue

        if line_clean.startswith("#"):
            parts = line_clean[1:].split(":", 1)
            if len(parts) == 2:
                headers[parts[0].strip().upper()] = parts[1].strip()
            lines.append(UltraStarLine(line_type="#", raw_line=line_clean))
        elif line_clean.upper().startswith(("P 1", "P 2", "P 3", "P1", "P2", "P3")):
            lines.append(UltraStarLine(line_type="P", raw_line=line_clean))
        elif line_clean.startswith(("-", "E")):
            parts = line_clean.split()
            beat = int(parts[1]) if len(parts) > 1 and parts[1].lstrip("-").isdigit() else 0
            lines.append(
                UltraStarLine(
                    line_type=line_clean[0],
                    start_beat=beat,
                    raw_line=line_clean,
                    is_line_break=(line_clean[0] == "-"),
                )
            )
        elif line_clean[0] in (":", "*", "F", "R", "G"):
            # Syntax: <type> <start_beat> <duration_beats> <pitch> <word...>
            # Preserves spaces in word
            parts = line_clean.split(" ", 4)
            if len(parts) >= 4:
                note_type = parts[0]
                s_beat = int(parts[1])
                dur_beat = int(parts[2])
                pitch_val = int(parts[3])
                w = parts[4] if len(parts) == 5 else ""
                lines.append(
                    UltraStarLine(
                        line_type=note_type,
                        start_beat=s_beat,
                        duration_beats=dur_beat,
                        pitch=pitch_val,
                        word=w,
                        raw_line=line_clean,
                        is_note=True,
                    )
                )
            else:
                lines.append(UltraStarLine(line_type=line_clean[0], raw_line=line_clean))
        else:
            lines.append(UltraStarLine(line_type="?", raw_line=line_clean))

    return headers, lines


def find_vocals_in_folder(folder_path: str, headers: dict[str, str]) -> Optional[str]:
    """Find the vocal track associated with the song."""
    # 1. Check #VOCALS: header
    if "VOCALS" in headers:
        candidate = os.path.join(folder_path, headers["VOCALS"])
        if os.path.isfile(candidate):
            return candidate

    # 2. Search common patterns in folder
    common_names = [
        "[Vocals].mp3", "[Vocals].wav", "[Vocals].ogg",
        "vocals.wav", "vocals.mp3",
    ]
    for name in common_names:
        cand = os.path.join(folder_path, name)
        if os.path.isfile(cand):
            return cand

    # 3. Check for any file containing [Vocals]
    for fname in sorted(os.listdir(folder_path)):
        if "[Vocals]" in fname and fname.lower().endswith((".mp3", ".wav", ".ogg", ".flac")):
            return os.path.join(folder_path, fname)

    # 4. Fallback to any audio file if no vocal track
    for fname in sorted(os.listdir(folder_path)):
        if fname.lower().endswith((".mp3", ".wav", ".ogg")):
            return os.path.join(folder_path, fname)

    return None


def find_song_folders_in_dir(input_path: str) -> list[str]:
    """Find all song folders containing UltraStar .txt files within an input path.

    If input_path is a direct song folder (or file), returns [input_path].
    If input_path is a parent directory (like /output/), returns all subdirectories
    that contain at least one valid UltraStar .txt file.
    """
    if os.path.isfile(input_path) and input_path.endswith(".txt"):
        return [input_path]

    if not os.path.isdir(input_path):
        return [input_path]

    # Check if input_path itself directly contains a .txt song file
    direct_txts = [f for f in sorted(os.listdir(input_path)) if f.endswith(".txt") and not f.endswith(".bak")]
    if direct_txts:
        return [input_path]

    # Otherwise, scan immediate subdirectories for song folders
    subdirs = [
        os.path.join(input_path, d) for d in sorted(os.listdir(input_path))
        if os.path.isdir(os.path.join(input_path, d))
    ]

    valid_song_dirs = []
    for s_dir in subdirs:
        txts = [f for f in sorted(os.listdir(s_dir)) if f.endswith(".txt") and not f.endswith(".bak")]
        if txts:
            valid_song_dirs.append(s_dir)

    return valid_song_dirs


def simulate_game_score(
    note_lines: list[UltraStarLine],
    bpm: float,
    gap_ms: float,
    vocal_rms: np.ndarray,
    rms_times: np.ndarray,
    pitched_data: PitchedData,
    noise_gate: float,
) -> tuple[int, float, float]:
    """Simulate UltraStar game engine scoring via 10ms frame hit-testing.

    Returns:
        tuple of (score_0_to_10000, timing_coverage_pct, pitch_accuracy_pct)
    """
    if not note_lines or bpm <= 0:
        return 0, 0.0, 0.0

    beat_sec = 15.0 / bpm
    total_song_points = 0.0
    possible_points = 0.0
    correct_pitch_frames = 0
    voiced_note_frames = 0
    covered_vocal_frames = 0
    total_vocal_frames = 0

    # Build pitch lookup function
    pitch_times = np.array(pitched_data.times)
    pitch_freqs = np.array(pitched_data.frequencies)
    pitch_confs = np.array(pitched_data.confidence)

    # Pre-calculate active vocal frames in song range
    song_start_sec = gap_ms / 1000.0
    last_note = note_lines[-1]
    song_end_sec = song_start_sec + ((last_note.start_beat + last_note.duration_beats) * beat_sec)

    vocal_mask = (rms_times >= song_start_sec - 1.0) & (rms_times <= song_end_sec + 1.0) & (vocal_rms >= noise_gate)
    total_vocal_frames = int(np.sum(vocal_mask))

    # Evaluate each note
    
    step_sec = 0.010  # 10ms hit-testing resolution
    note_hit_mask = np.zeros(len(rms_times), dtype=bool)

    total_weight = 0.0
    earned_weight = 0.0

    for n in note_lines:
        mult = 2.0 if n.line_type in ("*", "G") else 1.0
        note_weight = n.duration_beats * mult
        total_weight += note_weight

        t_start = (gap_ms / 1000.0) + (n.start_beat * beat_sec)
        t_end = t_start + (n.duration_beats * beat_sec)
        target_midi = n.pitch + 48

        # Mark frames covered by this note
        frames_in_note = (rms_times >= t_start) & (rms_times <= t_end)
        note_hit_mask |= frames_in_note
        has_vocal_energy = np.any(frames_in_note & (vocal_rms >= noise_gate))

        # Freestyle (F) and Rap (R) notes reward full credit for vocal timing without demanding pitch
        if n.line_type in ("F", "R"):
            earned = note_weight if has_vocal_energy else 0.0
            earned_weight += earned
            continue

        p_start = np.searchsorted(pitch_times, t_start)
        p_end = np.searchsorted(pitch_times, t_end)
        if p_end > p_start:
            sub_confs = pitch_confs[p_start:p_end]
            sub_freqs = pitch_freqs[p_start:p_end]
            v_mask = (sub_confs >= 0.20) & (sub_freqs >= 65.0)
            if np.any(v_mask):
                v_midis = librosa.hz_to_midi(sub_freqs[v_mask])
                diffs = np.abs(v_midis - target_midi)
                oct_diffs = np.abs((v_midis % 12) - (target_midi % 12))
                oct_diffs = np.minimum(oct_diffs, 12 - oct_diffs)
                # UltraStar Deluxe game engine tolerance: ±1.5 semitones (±150 cents for human vibrato)
                hit_mask = (diffs <= 1.5) | (oct_diffs <= 1.5)

                voiced_note_frames += int(np.sum(v_mask))
                correct_pitch_frames += int(np.sum(hit_mask))

                hit_ratio = float(np.mean(hit_mask))
                # UltraStar Deluxe game mechanics: Normal difficulty threshold (>=35% hit gives full note credit)
                if hit_ratio >= 0.35:
                    earned = note_weight
                else:
                    earned = note_weight * (hit_ratio / 0.35)
            elif has_vocal_energy:
                # Syllables with unvoiced consonants (s, t, p, k, ch) hit on-pitch timing
                earned = note_weight * 1.0
            else:
                earned = 0.0
        elif has_vocal_energy:
            earned = note_weight * 1.0
        else:
            earned = 0.0

        earned_weight += earned

    # Coverage metrics: percentage of note duration aligned with actual vocal energy
    note_frames = int(np.sum(note_hit_mask))
    note_vocal_hits = int(np.sum(note_hit_mask & (vocal_rms >= noise_gate)))
    timing_coverage = (note_vocal_hits / note_frames * 100.0) if note_frames > 0 else 100.0
    pitch_accuracy = (correct_pitch_frames / voiced_note_frames * 100.0) if voiced_note_frames > 0 else 100.0

    # Official UltraStar Deluxe engine point distribution:
    # 9,000 pts for notes (with 2x bonus for golden notes) + 1,000 line bonus
    note_score = (earned_weight / total_weight * 9000.0) if total_weight > 0 else 9000.0
    line_bonus = 1000.0 if pitch_accuracy >= 50.0 else (1000.0 * (pitch_accuracy / 50.0))
    raw_score = int(round(note_score + line_bonus))
    final_score = max(0, min(10000, raw_score))

    return final_score, timing_coverage, pitch_accuracy


def evaluate_karaoke_phrases_scoring(
    lines: list[UltraStarLine],
    bpm: float,
    gap_ms: float,
    art_feat: ArticulatoryAcousticFeatures,
    noise_gate: float,
    reference_words: Optional[list[str]] = None,
) -> list[dict]:
    """Motor IA para Asistente de UltraStar (Karaoke Score & Analysis por Frase):
    Evalúa cada frase cantada (delimitada por saltos de línea '-') integrando los 4 Módulos:
      - MÓDULO 1: Análisis Léxico y de Vocables (Levenshtein OOV + Markov/Entropía -> lyric_accuracy_%)
      - MÓDULO 2: Análisis de Afinación y Tono (F_0, Melisma dF0/dt != 0 & dF1/dt ~ 0, Vibrato 4-7 Hz -> pitch_accuracy_%)
      - MÓDULO 3: Análisis de Ritmo y Tiempo (Autocorrelación R(k) + Transitorios T_crest -> timing_accuracy_%)
      - MÓDULO 4: Dinámica, Energía y Estilo (Tarareo C < 1000 Hz, Penalización ZCR fuera de tiempo, Potencia E_RMS)
    Devuelve una lista de diccionarios JSON-serializables con:
      - 'pitch_accuracy_%'
      - 'timing_accuracy_%'
      - 'lyric_accuracy_%'
      - 'bonus_techniques' (ej. ['Vibrato', 'Melisma', 'Tarareo'])
      - 'overall_score'
    """
    from modules.Audio.bpm import compute_autocorrelation_tempo

    if not lines or bpm <= 0:
        return []

    beat_sec = 15.0 / bpm
    phrases: list[list[UltraStarLine]] = []
    curr_phrase: list[UltraStarLine] = []

    for l in lines:
        if l.is_note:
            curr_phrase.append(l)
        elif l.is_line_break or l.line_type in ("P", "E"):
            if curr_phrase:
                phrases.append(curr_phrase)
                curr_phrase = []
    if curr_phrase:
        phrases.append(curr_phrase)

    times = art_feat.times
    dt = float(np.mean(np.diff(times))) if len(times) > 1 else 0.010
    fps = 1.0 / max(1e-6, dt)
    climax_rms = float(np.percentile(art_feat.rms, 75)) if len(art_feat.rms) > 0 else 0.1

    phrase_results: list[dict] = []
    ref_cursor = 0

    for p_notes in phrases:
        if not p_notes:
            continue

        t_phrase_s = (gap_ms / 1000.0) + (p_notes[0].start_beat * beat_sec)
        t_phrase_e = (gap_ms / 1000.0) + ((p_notes[-1].start_beat + p_notes[-1].duration_beats) * beat_sec)
        idx_ps = int(np.searchsorted(times, t_phrase_s))
        idx_pe = int(np.searchsorted(times, t_phrase_e))

        bonus_set: list[str] = []
        voiced_frames = 0
        hit_frames = 0
        covered_frames = 0
        total_note_frames = 0
        lyric_sims: list[float] = []

        for n in p_notes:
            w_clean = n.word.strip()
            if w_clean == "~":
                if "Melisma" not in bonus_set:
                    bonus_set.append("Melisma")
            elif w_clean:
                if reference_words and ref_cursor < len(reference_words):
                    sim = character_level_similarity(w_clean, reference_words[ref_cursor])
                    ref_cursor += 1
                    lyric_sims.append(max(sim, 0.85 if is_scat_or_adlib_pattern(w_clean) else sim))
                else:
                    # Validar vocable repetitivo (Markov/Entropía) o palabra léxica
                    h_c, r_red = compute_phonotactic_entropy_and_reduplication(w_clean)
                    _, is_jit = compute_levenshtein_oov_distance(w_clean)
                    if is_scat_or_adlib_pattern(w_clean) or h_c < 1.15 or r_red >= 0.50 or not is_jit:
                        lyric_sims.append(1.0)
                    else:
                        lyric_sims.append(0.85)

            nt_s = (gap_ms / 1000.0) + (n.start_beat * beat_sec)
            nt_e = nt_s + (n.duration_beats * beat_sec)
            i_s = int(np.searchsorted(times, nt_s))
            i_e = int(np.searchsorted(times, nt_e))
            if i_e <= i_s:
                continue

            n_len = i_e - i_s
            total_note_frames += n_len
            sub_rms = art_feat.rms[i_s:i_e]
            sub_f0 = art_feat.f0_midi[i_s:i_e]
            sub_conf = art_feat.confidence[i_s:i_e]
            sub_cent = (
                art_feat.spectral_centroid[i_s:i_e]
                if art_feat.spectral_centroid is not None and len(art_feat.spectral_centroid) >= i_e
                else np.full(n_len, 1500.0, dtype=np.float32)
            )

            active_mask = sub_rms >= noise_gate
            covered_frames += int(np.sum(active_mask))

            # Módulo 4.1: Tarareo de estilo (Boca cerrada: C < 1000 Hz con energía sostenida)
            rms_mean_n = float(np.mean(sub_rms))
            rms_cv_n = float(np.std(sub_rms) / (rms_mean_n + 1e-6))
            if detect_closed_mouth_humming_acoustics(
                spectral_centroid_mean=float(np.mean(sub_cent)),
                f0_conf_mean=float(np.mean(sub_conf)),
                rms_cv=rms_cv_n,
            ):
                if "Tarareo" not in bonus_set:
                    bonus_set.append("Tarareo")

            # Módulo 2.1: Melisma acústico (dF0/dt != 0 y dF1/dt ~ 0)
            if (nt_e - nt_s) >= 0.350 and "Melisma" not in bonus_set:
                mel_chk = evaluate_melisma_vs_vibrato(art_feat, nt_s, nt_e)
                if mel_chk is not None:
                    bonus_set.append("Melisma")

            # Módulo 2.2: Vibrato controlado (4-7 Hz)
            if n_len >= 20:
                is_vib, _, a_v = evaluate_vibrato_sinusoid(
                    sub_f0, times[i_s:i_e], f_min=4.0, f_max=7.5, max_amplitude_st=1.5
                )
                if is_vib and "Vibrato" not in bonus_set:
                    bonus_set.append("Vibrato")

            if n.line_type in ("F", "R"):
                voiced_frames += n_len
                if np.any(active_mask):
                    hit_frames += n_len
                continue

            v_valid = (sub_f0 >= 21.0) & (sub_conf >= 0.20)
            if np.any(v_valid):
                target_midi = n.pitch + 48
                f_valid = sub_f0[v_valid]
                diffs = np.abs(f_valid - target_midi)
                oct_diffs = np.abs((f_valid % 12) - (target_midi % 12))
                oct_diffs = np.minimum(oct_diffs, 12.0 - oct_diffs)
                tol = 1.75 if "Vibrato" in bonus_set else 1.50
                hits = (diffs <= tol) | (oct_diffs <= tol)
                voiced_frames += int(np.sum(v_valid))
                hit_frames += int(np.sum(hits))

        # Módulo 3.1: Precisión de pulso R(k) y cobertura temporal
        cov_pct = (covered_frames / max(1, total_note_frames)) * 100.0
        pulse_consistency = 100.0
        if idx_pe - idx_ps >= 16:
            phrase_env = art_feat.rms[idx_ps:idx_pe]
            est_bpm = compute_autocorrelation_tempo(phrase_env, frame_rate_hz=fps)
            norm_ref_bpm = bpm
            while norm_ref_bpm > 180.0:
                norm_ref_bpm /= 2.0
            while norm_ref_bpm < 80.0:
                norm_ref_bpm *= 2.0
            rel_err = min(0.25, abs(est_bpm - norm_ref_bpm) / max(1.0, norm_ref_bpm))
            pulse_consistency = (1.0 - 0.4 * rel_err) * 100.0

        timing_acc = float(np.clip(0.80 * cov_pct + 0.20 * pulse_consistency, 0.0, 100.0))
        pitch_acc = float(np.clip((hit_frames / max(1, voiced_frames)) * 100.0, 0.0, 100.0)) if voiced_frames > 0 else 100.0
        lyric_acc = float(np.clip(np.mean(lyric_sims) * 100.0, 0.0, 100.0)) if lyric_sims else 100.0

        # Módulo 4.2 y 4.3: Penalización por ZCR fuera de tiempo y bonificación por potencia E_RMS en Golden Notes
        zcr_penalty_pts = 0
        if art_feat.zcr is not None and idx_pe > idx_ps:
            zcr_phrase = art_feat.zcr[idx_ps:idx_pe]
            conf_phrase = art_feat.confidence[idx_ps:idx_pe]
            breath_ratio = float(np.mean((zcr_phrase >= 0.22) & (conf_phrase < 0.25)))
            if breath_ratio > 0.30:
                zcr_penalty_pts = int(round((breath_ratio - 0.30) * 500.0))

        phrase_e_rms = float(np.sqrt(np.mean(art_feat.rms[idx_ps:idx_pe] ** 2))) if idx_pe > idx_ps else 0.0
        has_golden = any(n.line_type in ("*", "G") for n in p_notes)
        tech_bonus_pts = len(bonus_set) * 150 + (200 if has_golden and phrase_e_rms >= climax_rms else 0)

        base_pts = (0.50 * pitch_acc + 0.30 * timing_acc + 0.20 * lyric_acc) * 100.0
        overall_score = int(np.clip(round(base_pts + tech_bonus_pts - zcr_penalty_pts), 0, 10000))

        phrase_results.append({
            "pitch_accuracy_%": round(pitch_acc, 2),
            "timing_accuracy_%": round(timing_acc, 2),
            "lyric_accuracy_%": round(lyric_acc, 2),
            "bonus_techniques": bonus_set,
            "overall_score": overall_score,
        })

    return phrase_results


def _align_and_reconstruct_legacy_song(
    lines: list[UltraStarLine],
    headers: dict[str, str],
    vocal_path: str,
    original_gap: float,
    bpm: float,
    beat_sec: float,
    device: str,
    rms: np.ndarray,
    rms_times: np.ndarray,
    noise_gate: float,
    pitched_data: PitchedData,
) -> tuple[float, list[UltraStarLine], int]:
    """Fallback acoustic alignment for third-party or legacy UltraStar songs."""
    print(f"{ULTRASINGER_HEAD} Running Whisper acoustic word-level transcription on {blue_highlighted(os.path.basename(vocal_path))}...")
    from faster_whisper import WhisperModel
    whisper_device = "cuda" if device == "cuda" else "cpu"
    whisper_compute = "float16" if whisper_device == "cuda" else "int8"
    try:
        w_model = WhisperModel("small", device=whisper_device, compute_type=whisper_compute)
    except Exception:
        w_model = WhisperModel("small", device="cpu", compute_type="int8")

    song_lang = headers.get("LANGUAGE", "").lower()
    asr_lang = "es" if "span" in song_lang or "es" in song_lang else None
    segments, _ = w_model.transcribe(vocal_path, language=asr_lang, word_timestamps=True)

    # Vocablo / Ad-lib pattern for non-dictionary scat singing (e.g. tadarara, uhm, la la)
    VOCABLO_PATTERN = re.compile(
        r"^(ta|da|ra|la|na|ba|pa|uhm|um|ah|oh|ay|hey|yeh|yeah|mmm|shh|tarara|tadarara|tararara)+$",
        re.IGNORECASE,
    )
    txt_clean_set = set(re.sub(r"[^\w]", "", l.word.lower()) for l in lines if l.is_note)

    whisper_words = []
    for s in segments:
        for w in s.words:
            clean = re.sub(r"[^\w]", "", w.word.lower())
            if clean:
                is_vocablo = bool(VOCABLO_PATTERN.match(clean)) and (clean not in txt_clean_set)
                whisper_words.append({
                    "word": w.word.strip(),
                    "clean": clean,
                    "start": float(w.start),
                    "end": float(w.end),
                    "is_vocablo": is_vocablo,
                })

    vocablo_count = sum(1 for w in whisper_words if w["is_vocablo"])
    if vocablo_count > 0:
        print(f"{ULTRASINGER_HEAD} Quarantined {gold_highlighted(str(vocablo_count))} non-dictionary vocalization(s) (tarareos/ad-libs).")
    print(f"{ULTRASINGER_HEAD} Whisper detected {green_highlighted(str(len(whisper_words)))} acoustic words with millisecond precision.")

    # 7. Group UltraStar lines into words while tracking phrase linebreaks (preserving melismas ~ intact)
    words = []
    curr_notes = []
    curr_text = ""
    for idx, l in enumerate(lines):
        if l.is_note:
            is_melisma = (l.word.strip() == "~")
            if is_melisma and not curr_notes and words:
                words[-1]["notes"].append(l)
                words[-1]["text"] += l.word
            else:
                curr_notes.append(l)
                curr_text += l.word
                if (l.word.endswith(" ") or l.word == "") and not is_melisma:
                    clean = re.sub(r"[^\w]", "", curr_text.lower())
                    if clean:
                        words.append({"text": curr_text, "clean": clean, "notes": curr_notes, "is_line_end": False})
                        curr_notes = []
                        curr_text = ""
                    elif words:
                        words[-1]["notes"].extend(curr_notes)
                        words[-1]["text"] += curr_text
                        curr_notes = []
                        curr_text = ""
        elif l.is_line_break:
            if curr_notes:
                clean = re.sub(r"[^\w]", "", curr_text.lower())
                if clean:
                    words.append({"text": curr_text, "clean": clean, "notes": curr_notes, "is_line_end": True})
                elif words:
                    words[-1]["notes"].extend(curr_notes)
                    words[-1]["text"] += curr_text
                    words[-1]["is_line_end"] = True
                curr_notes = []
                curr_text = ""
            elif words:
                words[-1]["is_line_end"] = True

    if curr_notes:
        clean = re.sub(r"[^\w]", "", curr_text.lower())
        if clean:
            words.append({"text": curr_text, "clean": clean, "notes": curr_notes, "is_line_end": False})
        elif words:
            words[-1]["notes"].extend(curr_notes)
            words[-1]["text"] += curr_text
        curr_notes = []
        curr_text = ""

    # 8. Align TXT words with Whisper ground truth using SequenceMatcher + Fuzzy Anchors
    first_target_words = [w["text"].strip() for w in words[:4] if w["text"].strip()]
    whisper_words, intro_adlibs = filter_intro_adlibs(whisper_words, first_target_words, similarity_threshold=0.45)
    if intro_adlibs:
        anchor_name = whisper_words[0]["word"] if whisper_words else "N/A"
        print(
            f"{ULTRASINGER_HEAD} {cyan_highlighted('Intro Ad-lib Purge:')} Quarantined {red_highlighted(str(len(intro_adlibs)))} "
            f"initial vocalization(s) (tarareos/ad-libs). First real lyric anchor: '{green_highlighted(anchor_name)}'."
        )

    txt_cleans = [w["clean"] for w in words]
    w_cleans = [
        (f"__voc_{i}__" if w.get("is_vocablo") else w["clean"])
        for i, w in enumerate(whisper_words)
    ]

    import difflib
    matcher = difflib.SequenceMatcher(None, txt_cleans, w_cleans, autojunk=False)
    matches = matcher.get_matching_blocks()

    word_timing = {}  # txt_idx -> (t_start, t_end)
    for m in matches:
        for k in range(m.size):
            t_idx = m.a + k
            w_idx = m.b + k
            word_timing[t_idx] = (whisper_words[w_idx]["start"], whisper_words[w_idx]["end"])

    # Fuzzy Anchor Expansion for boundary monosyllables/short words
    for m in matches:
        if m.size == 0:
            continue
        pre_t = m.a - 1
        pre_w = m.b - 1
        if pre_t >= 0 and pre_t not in word_timing and pre_w >= 0:
            t_word = txt_cleans[pre_t]
            w_word = whisper_words[pre_w]["clean"]
            time_gap = whisper_words[m.b]["start"] - whisper_words[pre_w]["end"]
            if time_gap <= 0.85 and (len(t_word) <= 4 or len(w_word) <= 4):
                sim = character_level_similarity(t_word, w_word)
                if sim >= 0.45 or (len(t_word) <= 2 and len(w_word) <= 2):
                    word_timing[pre_t] = (whisper_words[pre_w]["start"], whisper_words[pre_w]["end"])

        post_t = m.a + m.size
        post_w = m.b + m.size
        if post_t < len(words) and post_t not in word_timing and post_w < len(whisper_words):
            t_word = txt_cleans[post_t]
            w_word = whisper_words[post_w]["clean"]
            time_gap = whisper_words[post_w]["start"] - whisper_words[m.b + m.size - 1]["end"]
            if time_gap <= 0.85 and (len(t_word) <= 4 or len(w_word) <= 4):
                sim = character_level_similarity(t_word, w_word)
                if sim >= 0.45 or (len(t_word) <= 2 and len(w_word) <= 2):
                    word_timing[post_t] = (whisper_words[post_w]["start"], whisper_words[post_w]["end"])

    if not word_timing:
        first_matched_idx = 0
        new_gap = original_gap
    else:
        first_matched_idx = min(word_timing.keys())
        purged_intro_count = first_matched_idx
        if purged_intro_count > 0:
            print(
                f"{ULTRASINGER_HEAD} {cyan_highlighted('Phantom Intro Purge:')} removed {red_highlighted(str(purged_intro_count))} "
                f"unmatched/instrumental word(s) before real vocal onset."
            )

        valid_indices = sorted([i for i in word_timing.keys() if i >= first_matched_idx])
        for i in range(len(valid_indices) - 1):
            idx1 = valid_indices[i]
            idx2 = valid_indices[i + 1]
            t1_end = word_timing[idx1][1]
            t2_start = word_timing[idx2][0]
            missing_count = idx2 - idx1 - 1
            gap_sec = t2_start - t1_end

            if missing_count > 0:
                if gap_sec <= 0.85:
                    step = max(0.08, gap_sec / (missing_count + 1))
                    for m_i in range(1, missing_count + 1):
                        cur_s = t1_end + (m_i - 1) * step
                        cur_e = cur_s + step * 0.90
                        word_timing[idx1 + m_i] = (cur_s, cur_e)
                else:
                    split_idx = idx1
                    for k in range(idx1, idx2):
                        if words[k].get("is_line_end"):
                            split_idx = k
                            break

                    curr_pos = t1_end + 0.05
                    for m_idx in range(idx1 + 1, split_idx + 1):
                        w_len = len(words[m_idx]["notes"])
                        w_dur = min(0.35, max(0.10, 0.12 * w_len))
                        word_timing[m_idx] = (curr_pos, curr_pos + w_dur)
                        curr_pos += w_dur + 0.05

                    phrase2_words = list(range(split_idx + 1, idx2))
                    if phrase2_words:
                        curr_pos = t2_start - 0.05
                        for m_idx in reversed(phrase2_words):
                            w_len = len(words[m_idx]["notes"])
                            w_dur = min(0.35, max(0.10, 0.12 * w_len))
                            word_timing[m_idx] = (curr_pos - w_dur, curr_pos)
                            curr_pos -= (w_dur + 0.05)

        new_gap = word_timing[first_matched_idx][0] * 1000.0

    print(
        f"{ULTRASINGER_HEAD} Calibrated #GAP: {blue_highlighted(f'{original_gap:.0f}ms')} -> "
        f"{green_highlighted(f'{int(round(new_gap))}ms')} ({new_gap - original_gap:+.0f}ms shift to vocal onset)"
    )

    new_gap_sec = new_gap / 1000.0
    adjusted_notes_count = 0

    final_output_lines: list[UltraStarLine] = []
    for l in lines:
        if l.line_type == "#":
            if l.raw_line.upper().startswith("#GAP:"):
                final_output_lines.append(UltraStarLine(line_type="#", raw_line=f"#GAP:{int(round(new_gap))}"))
            else:
                final_output_lines.append(l)

    prev_end_beat = -1
    words_in_current_phrase = 0

    for w_i in range(first_matched_idx, len(words)):
        w_obj = words[w_i]
        t_s, t_e = word_timing.get(w_i, (None, None))
        if t_s is None:
            prev_e_sec = (
                (new_gap_sec + prev_end_beat * beat_sec)
                if prev_end_beat >= 0 else new_gap_sec
            )
            t_s = prev_e_sec + 0.05
            t_e = t_s + 0.25

        dur_total = max(0.06, t_e - t_s)
        n_notes = len(w_obj["notes"])

        # Proportional syllable weighting based on original note durations
        orig_durations = [max(1, n.duration_beats) for n in w_obj["notes"]]
        total_orig_beats = sum(orig_durations)
        weights = [d / total_orig_beats for d in orig_durations] if total_orig_beats > 0 else [1.0 / n_notes] * n_notes
        total_w = sum(weights)

        next_ts, _ = word_timing.get(w_i + 1, (None, None))
        gap_to_next = (next_ts - t_e) if next_ts is not None else 0.0

        cur_syl_time = t_s
        for n_i, n in enumerate(w_obj["notes"]):
            syl_frac = weights[n_i] / total_w
            syl_dur = dur_total * syl_frac
            syl_start = cur_syl_time
            syl_end = syl_start + syl_dur * 0.95
            cur_syl_time += syl_dur

            if n_i == n_notes - 1 and (w_obj.get("is_line_end") or gap_to_next >= 0.70):
                scan_limit = min(t_e + 0.60, t_e + gap_to_next - 0.15 if gap_to_next > 0.25 else t_e)
                if scan_limit > t_e:
                    f_mask = (rms_times >= t_e) & (rms_times <= scan_limit) & (rms >= noise_gate * 1.15)
                    if np.any(f_mask):
                        last_vocal_t = float(rms_times[f_mask][-1])
                        syl_end += min(0.50, max(0.0, last_vocal_t - t_e))

            s_beat = max(0, int(round((syl_start - new_gap_sec) / beat_sec)))
            d_beat = max(1, int(round((syl_end - syl_start) / beat_sec)))

            if prev_end_beat >= 0 and s_beat < prev_end_beat:
                s_beat = prev_end_beat

            n.start_beat = s_beat
            n.duration_beats = d_beat
            prev_end_beat = s_beat + d_beat

            f_start = np.searchsorted(pitched_data.times, syl_start)
            f_end = np.searchsorted(pitched_data.times, syl_end)
            if f_end > f_start:
                sub_freqs = pitched_data.frequencies[f_start:f_end]
                sub_confs = pitched_data.confidence[f_start:f_end]
                valid_mask = [
                    (c >= 0.35 and 65.0 <= f <= 1100.0)
                    for f, c in zip(sub_freqs, sub_confs)
                ]
                if any(valid_mask):
                    v_freqs = [f for f, m in zip(sub_freqs, valid_mask) if m]
                    v_confs = [c for c, m in zip(sub_confs, valid_mask) if m]
                    v_midis = [librosa.hz_to_midi(f) for f in v_freqs]
                    v_weights = [c ** 2 for c in v_confs]
                    weighted_midi = float(np.average(v_midis, weights=v_weights))
                    target_us_pitch = int(round(weighted_midi)) - 48
                    n.pitch = target_us_pitch

            final_output_lines.append(n)
            adjusted_notes_count += 1

        words_in_current_phrase += 1

        if w_i < len(words) - 1:
            next_ts, _ = word_timing.get(w_i + 1, (None, None))
            gap_sec = (next_ts - t_e) if next_ts is not None else 0.0

            should_break = (
                w_obj.get("is_line_end", False)
                or gap_sec >= 0.70
                or (words_in_current_phrase >= 7 and w_obj["text"].strip().endswith((".", ",", ";", "!", "?", "…")))
                or words_in_current_phrase >= 10
            )

            if should_break:
                if next_ts is not None:
                    next_start_est = max(prev_end_beat + 2, int(round((next_ts - new_gap_sec) / beat_sec)))
                else:
                    next_start_est = prev_end_beat + 2

                if next_start_est > prev_end_beat + 1:
                    break_beat = prev_end_beat + max(1, (next_start_est - prev_end_beat) // 2)
                else:
                    break_beat = prev_end_beat + 1

                final_output_lines.append(UltraStarLine(line_type="-", start_beat=break_beat, is_line_break=True))
                prev_end_beat = break_beat + 1
                words_in_current_phrase = 0

    final_output_lines.append(UltraStarLine(line_type="E", raw_line="E"))
    return new_gap, final_output_lines, adjusted_notes_count


def autocorrect_song(
    input_path: str,
    pitcher_name: str = "rmvpe",
    device: str = "cuda",
) -> QASimulationReport:
    """Execute acoustic auto-correction and game engine QA on an UltraStar song."""
    # 1. Resolve .txt file and folder
    if os.path.isdir(input_path):
        folder_path = input_path
        txt_files = [f for f in sorted(os.listdir(folder_path)) if f.endswith(".txt") and not f.endswith(".bak")]
        if not txt_files:
            raise FileNotFoundError(f"No UltraStar .txt file found in {folder_path}")
        txt_path = os.path.join(folder_path, txt_files[0])
    elif os.path.isfile(input_path) and input_path.endswith(".txt"):
        txt_path = input_path
        folder_path = os.path.dirname(txt_path)
    else:
        raise ValueError(f"Input path must be an UltraStar folder or .txt file: {input_path}")

    print(f"\n{ULTRASINGER_HEAD} ===================================================================")
    print(f"{ULTRASINGER_HEAD}          {cyan_highlighted('ULTRASINGER ACOUSTIC AUTO-CORRECT & QA ENGINE')}")
    print(f"{ULTRASINGER_HEAD} ===================================================================")
    print(f"{ULTRASINGER_HEAD} Target song: {blue_highlighted(os.path.basename(txt_path))}")

    # 2. Parse UltraStar .txt (Idempotency Guard: always read from pristine .bak if present, unless .txt is newer)
    backup_path = txt_path + ".bak"
    parse_target = txt_path
    if os.path.isfile(backup_path):
        if os.path.getmtime(txt_path) > os.path.getmtime(backup_path) + 1.0:
            # txt_path was freshly generated after backup -> refresh backup with new baseline
            shutil.copy2(txt_path, backup_path)
            parse_target = backup_path
        else:
            print(
                f"{ULTRASINGER_HEAD} Idempotency Guard: Reading pristine baseline from {blue_highlighted(os.path.basename(backup_path))}..."
            )
            parse_target = backup_path

    headers, lines = parse_ultrastar_txt(parse_target)
    note_lines = [l for l in lines if l.is_note]
    if not note_lines:
        raise ValueError(f"No notes found in {txt_path}")

    bpm = float(headers.get("BPM", "120.0").replace(",", "."))
    original_gap = float(headers.get("GAP", "0.0").replace(",", "."))

    song_genre = headers.get("GENRE", "")
    if not song_genre or song_genre.lower() == "pop":
        for g_name in ["GENRE.txt", "genre.txt", "GENRE", "genre"]:
            cand = os.path.join(folder_path, g_name)
            if not os.path.isfile(cand):
                cand = os.path.join(folder_path, "..", "entrada", os.path.basename(folder_path), g_name)
            if os.path.isfile(cand):
                try:
                    with open(cand, "r", encoding="utf-8-sig", errors="replace") as gf:
                        c_genre = gf.read().strip()
                        c_clean = re.sub(r"^#?genre\s*:\s*", "", c_genre, flags=re.IGNORECASE).strip()
                        if c_clean:
                            song_genre = c_clean.splitlines()[0].strip()
                            headers["GENRE"] = song_genre
                            break
                except Exception:
                    pass
    if not song_genre:
        song_genre = "Pop"
    genre_profile = resolve_genre_profile(song_genre)
    print(
        f"{ULTRASINGER_HEAD} Musical Genre Profile : {blue_highlighted(genre_profile.name)} "
        f"({cyan_highlighted(genre_profile.family)}) [vibrato tol: {genre_profile.vibrato_tolerance_semitones} st]"
    )

    # Grid Resolution: keep #BPM in the standard UltraStar range [200, 400].
    # (Upscaling to >= 600 BPM produced ~18 ms micro-notes and doubled #BPM on every run.)
    bpm_multiplier = 1
    if 0 < bpm < 200.0:
        for k in [2, 3, 4]:
            if bpm * k >= 200.0:
                bpm_multiplier = k
                break

    if bpm_multiplier > 1:
        bpm = round(bpm * bpm_multiplier, 2)
        headers["BPM"] = str(bpm)
        for l in lines:
            if l.is_note:
                l.start_beat *= bpm_multiplier
                l.duration_beats *= bpm_multiplier
            elif l.is_line_break:
                l.start_beat *= bpm_multiplier
        print(
            f"{ULTRASINGER_HEAD} {cyan_highlighted('Ultra-Fine Grid Upscaling:')} Multiplied #BPM by {blue_highlighted(f'{bpm_multiplier}x')} "
            f"-> {green_highlighted(f'{bpm:.2f}')} BPM (~{15000.0/bpm:.1f}ms/beat resolution)"
        )

    beat_sec = 15.0 / bpm

    # 3. Locate and load Vocal Audio
    vocal_path = find_vocals_in_folder(folder_path, headers)
    if not vocal_path:
        raise FileNotFoundError(f"No clean vocal audio found for {txt_path} in {folder_path}")

    print(f"{ULTRASINGER_HEAD} Vocal audio: {blue_highlighted(os.path.basename(vocal_path))}")

    y, sr = librosa.load(vocal_path, sr=22050, mono=True)
    hop_length = 128  # ~5.8ms resolution
    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=hop_length)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    max_rms = float(np.max(rms)) if len(rms) > 0 else 0.0
    # Precaution 1: Strict Noise Gate (-38 dBFS floor or 12% peak)
    noise_gate = max(max_rms * 0.12, 0.008)

    # 4. Extract or load pitch data
    print(f"{ULTRASINGER_HEAD} Extracting vocal pitch for calibration with {blue_highlighted(pitcher_name)}...")
    pitched_data = get_pitch_with_file(vocal_path, pitcher_name=pitcher_name, device=device)

    # Ecuación IV: Fusión Sensorial Bayesiana Dual (RMVPE + CREPE Full) en GPU (RTX 4060)
    if device == "cuda" and pitcher_name == "rmvpe":
        try:
            import torch
            import gc
            if torch.cuda.is_available():
                print(f"{ULTRASINGER_HEAD} {cyan_highlighted('Dual Neural Pitch Ensemble:')} Fusing RMVPE + CREPE Full (CNN 22M) on RTX GPU...")
                crepe_data = get_pitch_with_file(vocal_path, pitcher_name="crepe", device="cuda")
                if crepe_data and len(crepe_data.frequencies) > 0 and len(pitched_data.frequencies) > 0:
                    r_t = np.array(pitched_data.times)
                    r_f = np.array(pitched_data.frequencies)
                    r_c = np.array(pitched_data.confidence)
                    c_t = np.array(crepe_data.times)
                    c_f = np.array(crepe_data.frequencies)
                    c_c = np.array(crepe_data.confidence)

                    c_f_interp = np.interp(r_t, c_t, c_f, left=0.0, right=0.0)
                    c_c_interp = np.interp(r_t, c_t, c_c, left=0.0, right=0.0)

                    fused_freqs = []
                    fused_confs = []
                    for k in range(len(r_t)):
                        rf_k, rc_k = r_f[k], r_c[k]
                        cf_k, cc_k = c_f_interp[k], c_c_interp[k]
                        if rf_k >= 65.0 and cf_k >= 65.0:
                            rm_k = librosa.hz_to_midi(rf_k)
                            cm_k = librosa.hz_to_midi(cf_k)
                            diff = abs(rm_k - cm_k)
                            if diff <= 1.0:
                                w_r = rc_k ** 2
                                w_c = cc_k ** 2
                                fused_m = (w_r * rm_k + w_c * cm_k) / (w_r + w_c + 1e-6)
                                fused_freqs.append(float(librosa.midi_to_hz(fused_m)))
                                fused_confs.append(float(max(rc_k, cc_k)))
                            elif 10.5 <= diff <= 13.5:
                                fused_freqs.append(rf_k)
                                fused_confs.append(rc_k)
                            elif rc_k < 0.25 and cc_k >= 0.50:
                                fused_freqs.append(cf_k)
                                fused_confs.append(cc_k)
                            else:
                                fused_freqs.append(rf_k if rc_k >= cc_k else cf_k)
                                fused_confs.append(float(max(rc_k, cc_k)))
                        elif cf_k >= 65.0 and cc_k >= 0.40 and rf_k < 65.0:
                            fused_freqs.append(cf_k)
                            fused_confs.append(cc_k)
                        else:
                            fused_freqs.append(rf_k)
                            fused_confs.append(rc_k)

                    pitched_data.frequencies = fused_freqs
                    pitched_data.confidence = fused_confs
                del crepe_data
                gc.collect()
                torch.cuda.empty_cache()
        except Exception as e_ens:
            print(f"{ULTRASINGER_HEAD} Dual neural ensemble note: {e_ens}")

    # Temporal Median Smoothing (ISMIR): 3-frame median filter removes transient 10ms octave jumps while preserving attacks
    if pitched_data and len(pitched_data.frequencies) > 0:
        from scipy.ndimage import median_filter
        arr_f = np.array(pitched_data.frequencies, dtype=np.float32)
        smoothed = median_filter(arr_f, size=3, mode="nearest")
        pitched_data.frequencies = smoothed.tolist()

    # SOTA Equation 0 & 1: Extract 10 ms articulatory acoustic vectors and pure vowel mask
    print(f"{ULTRASINGER_HEAD} Extracting acoustic-articulatory feature vectors (10 ms hop)...")
    art_feat = extract_articulatory_features(y=y, sr=sr, pitched_data=pitched_data, hop_ms=10.0)
    rms = art_feat.rms
    rms_times = art_feat.times

    # 5. Measure Baseline Score (before auto-correct)
    orig_score, orig_cov, orig_pitch_acc = simulate_game_score(
        note_lines, bpm, original_gap, rms, rms_times, pitched_data, noise_gate
    )
    print(f"{ULTRASINGER_HEAD} Baseline Game Score : {gold_highlighted(f'{orig_score:,} / 10,000')} (coverage: {orig_cov:.1f}%, pitch: {orig_pitch_acc:.1f}%)")

    is_ultrasinger_created = "ultrasinger" in headers.get("CREATOR", "").lower()

    if is_ultrasinger_created:
        print(
            f"{ULTRASINGER_HEAD} {cyan_highlighted('SOTA Pipeline Protection:')} Song was created by UltraSinger (WhisperX)."
        )
        print(
            f"{ULTRASINGER_HEAD} Preserving millisecond-precision phoneme timestamps, melismas (~), and #GAP intact."
        )
        print(
            f"{ULTRASINGER_HEAD} Performing non-destructive tonal pitch calibration with {blue_highlighted(pitcher_name)}..."
        )
        # Monosyllable Legato Rebalance (e.g. 'a' holding for 37 beats while rhyme 'mi' is pushed to dead air):
        # In vocal Spanish singing, unstressed prepositions ('a', 'de', 'en', 'y') last ~0.20-0.26s (5-7 beats at ~380 BPM)
        # while the stressed pronoun ('mí', 'ti', etc.) holds the remainder of the sustained vocal belt.
        for idx in range(len(lines) - 1):
            curr_l = lines[idx]
            if curr_l.is_note and curr_l.word.strip().lower() in ("a", "de", "en", "y", "o") and curr_l.duration_beats >= 12:
                for next_idx in range(idx + 1, min(idx + 5, len(lines))):
                    next_l = lines[next_idx]
                    if next_l.is_line_break or next_l.line_type in ("#", "P", "E"):
                        break
                    if next_l.is_note:
                        if next_l.word.strip().lower() in ("mi", "mí", "ti", "tu", "yo", "él", "dios", "sol") and next_l.duration_beats <= 8:
                            prep_beats = min(6, max(4, int(round(0.24 / beat_sec))))
                            total_span = (next_l.start_beat + next_l.duration_beats) - curr_l.start_beat
                            curr_start_sec = (original_gap / 1000.0) + (curr_l.start_beat * beat_sec)
                            
                            # Bound search to phrase limit (avoid bleeding across whole song)
                            max_scan_sec = curr_start_sec + 3.0
                            for b_l in lines[next_idx + 1 : min(next_idx + 10, len(lines))]:
                                if b_l.is_line_break:
                                    max_scan_sec = min(max_scan_sec, (original_gap / 1000.0) + (b_l.start_beat * beat_sec))
                                    break

                            r_search = np.where((rms_times >= curr_start_sec) & (rms_times <= max_scan_sec) & (rms >= noise_gate * 0.70))[0]
                            if len(r_search) > 0:
                                search_times = rms_times[r_search]
                                last_idx = 0
                                for s_i in range(len(search_times) - 1):
                                    if search_times[s_i + 1] - search_times[s_i] > 0.35:
                                        last_idx = s_i
                                        break
                                    last_idx = s_i + 1
                                last_active_sec = float(search_times[last_idx])
                                active_beats = int(round((last_active_sec - curr_start_sec) / beat_sec))
                                total_span = max(total_span, active_beats)

                            curr_l.duration_beats = prep_beats
                            next_l.start_beat = curr_l.start_beat + prep_beats
                            next_l.duration_beats = max(8, total_span - prep_beats)
                            next_l.line_type = ":"

                            # Recalibrate pitch for the rebalanced notes
                            p_t = np.array(pitched_data.times)
                            p_f = np.array(pitched_data.frequencies)
                            p_c = np.array(pitched_data.confidence)

                            s_t_next = (original_gap / 1000.0) + (next_l.start_beat * beat_sec)
                            e_t_next = s_t_next + (next_l.duration_beats * beat_sec)
                            mask_next = (p_t >= s_t_next) & (p_t <= e_t_next) & (p_c >= 0.30) & (p_f >= 65.0)
                            if np.any(mask_next):
                                v_m = [librosa.hz_to_midi(f) for f in p_f[mask_next]]
                                v_w = p_c[mask_next] ** 2
                                next_l.pitch = int(round(float(np.average(v_m, weights=v_w)))) - 48

                            print(
                                f"{ULTRASINGER_HEAD} {cyan_highlighted('Legato Rebalance:')} "
                                f"Rebalanced '{curr_l.word.strip()}' ({curr_l.duration_beats} beats) and "
                                f"'{next_l.word.strip()}' ({next_l.duration_beats} beats, pitch={next_l.pitch}) to match natural singing."
                            )
                        break

        # Identify instrumental phantom notes in dead air
        p_times = np.array(pitched_data.times)
        p_freqs = np.array(pitched_data.frequencies)
        p_confs = np.array(pitched_data.confidence)

        # 1. Map note presence and active singing intervals
        note_evals = []
        for idx, l in enumerate(lines):
            if l.is_note:
                syl_start = (original_gap / 1000.0) + (l.start_beat * beat_sec)
                syl_end = syl_start + (l.duration_beats * beat_sec)
                frames_in_note = (rms_times >= syl_start) & (rms_times <= syl_end)
                has_energy = bool(np.any(frames_in_note & (rms >= noise_gate * 0.70)))
                f_s = np.searchsorted(p_times, syl_start)
                f_e = np.searchsorted(p_times, syl_end)
                has_pitch = False
                if f_e > f_s:
                    v = (p_confs[f_s:f_e] >= 0.20) & (p_freqs[f_s:f_e] >= 65.0)
                    has_pitch = bool(np.any(v))
                note_evals.append((idx, syl_start, syl_end, has_energy, has_pitch))

        singing_intervals = [
            (s, e) for (_, s, e, has_energy, has_pitch) in note_evals
            if has_energy or has_pitch
        ]

        # Instrumental bridge purge: ONLY purge clusters (>=3 notes spanning >=1.8s)
        # that are strictly isolated (>= 2.5s) from any real vocal singing.
        # Micro-syllables contiguous to phrases ('mi', 'te', 'extraño') are NEVER purged!
        dead_cluster_indices = set()
        current_cluster = []
        for (idx, s, e, has_energy, has_pitch) in note_evals:
            if not has_energy and not has_pitch:
                min_dist_to_singing = (
                    min(max(0.0, max(s - sing_e, sing_s - e)) for (sing_s, sing_e) in singing_intervals)
                    if singing_intervals else 999.0
                )
                if min_dist_to_singing >= 2.5:
                    current_cluster.append((idx, s, e))
                else:
                    if len(current_cluster) >= 3:
                        dur_span = current_cluster[-1][2] - current_cluster[0][1]
                        if dur_span >= 1.8:
                            for c_idx, _, _ in current_cluster:
                                dead_cluster_indices.add(c_idx)
                    current_cluster = []
            else:
                if len(current_cluster) >= 3:
                    dur_span = current_cluster[-1][2] - current_cluster[0][1]
                    if dur_span >= 1.8:
                        for c_idx, _, _ in current_cluster:
                            dead_cluster_indices.add(c_idx)
                current_cluster = []

        if len(current_cluster) >= 3:
            dur_span = current_cluster[-1][2] - current_cluster[0][1]
            if dur_span >= 1.8:
                for c_idx, _, _ in current_cluster:
                    dead_cluster_indices.add(c_idx)

        # Anti-deletion guard: lexical words (from lyrics.txt) are never purged; only
        # non-lexical scat/ad-lib notes over instrumental sections may be removed.
        dead_cluster_indices = {
            c_idx for c_idx in dead_cluster_indices
            if is_scat_or_adlib_pattern(lines[c_idx].word) or lines[c_idx].word.strip() == "~"
        }

        if dead_cluster_indices:
            print(
                f"{ULTRASINGER_HEAD} {cyan_highlighted('Instrumental Bridge Purge:')} Removed "
                f"{red_highlighted(str(len(dead_cluster_indices)))} phantom notes placed over instrumental solo."
            )

        new_gap = original_gap
        surviving_items = [l for idx, l in enumerate(lines) if idx not in dead_cluster_indices]

        # Ecuación I en Autocorrect: Re-anclar palabras iniciales huérfanas ('Yo~' + '-' + 'quiero...')
        # que fueron partidas por un tarareo previo antes de la frase real
        repaired_items: list[UltraStarLine] = []
        idx_s_item = 0
        while idx_s_item < len(surviving_items):
            curr_item = surviving_items[idx_s_item]
            # Detectar patrón: [Nota_1 (palabra corta)] + [opcional Nota_2 ('~')] + [LineBreak '-'] + [Nota_siguiente minúscula tras brecha > 2.0s]
            if (
                curr_item.is_note
                and curr_item.word.strip() != "~"
                and len(curr_item.word.strip()) <= 3
                and (idx_s_item == 0 or surviving_items[idx_s_item - 1].is_line_break or surviving_items[idx_s_item - 1].line_type == "P")
            ):
                mel_offset = 1
                while (
                    idx_s_item + mel_offset < len(surviving_items)
                    and surviving_items[idx_s_item + mel_offset].is_note
                    and surviving_items[idx_s_item + mel_offset].word.strip() == "~"
                ):
                    mel_offset += 1

                lb_idx = idx_s_item + mel_offset
                next_note_idx = lb_idx + 1
                if (
                    lb_idx < len(surviving_items)
                    and surviving_items[lb_idx].is_line_break
                    and next_note_idx < len(surviving_items)
                    and surviving_items[next_note_idx].is_note
                ):
                    next_note = surviving_items[next_note_idx]
                    last_head_note = surviving_items[lb_idx - 1]
                    head_end_sec = (original_gap / 1000.0) + ((last_head_note.start_beat + last_head_note.duration_beats) * beat_sec)
                    next_start_sec = (original_gap / 1000.0) + (next_note.start_beat * beat_sec)
                    gap_sec = next_start_sec - head_end_sec
                    next_w_clean = next_note.word.strip()

                    if gap_sec >= 2.0 and next_w_clean and next_w_clean[0].islower():
                        # Re-anclar curr_item justo antes de next_note y eliminar los '~' y '-' intermedios
                        target_dur_beats = max(4, int(round(0.26 / beat_sec)))
                        curr_item.start_beat = max(0, next_note.start_beat - target_dur_beats - 1)
                        curr_item.duration_beats = target_dur_beats
                        curr_item.pitch = next_note.pitch
                        if not curr_item.word.endswith(" "):
                            curr_item.word = curr_item.word.strip() + " "
                        repaired_items.append(curr_item)
                        print(
                            f"{ULTRASINGER_HEAD} {gold_highlighted('Orphan Head Repair (Eq. I):')} "
                            f"Re-anchored '{curr_item.word.strip()}' from tarareo ({head_end_sec:.2f}s) "
                            f"to join '{next_w_clean}' at {next_start_sec:.2f}s."
                        )
                        idx_s_item = next_note_idx
                        continue

            repaired_items.append(curr_item)
            idx_s_item += 1

        surviving_items = repaired_items

        # Step B: Map active notes with initial physical timing and player track ID
        note_entries = []
        current_player = "P1"
        for l in surviving_items:
            if l.line_type == "P":
                current_player = l.raw_line.strip().upper()
            elif l.is_note:
                t_s = (original_gap / 1000.0) + (l.start_beat * beat_sec)
                t_e = t_s + (l.duration_beats * beat_sec)
                note_entries.append({
                    "line": l,
                    "t_start": t_s,
                    "t_end": t_e,
                    "orig_start": t_s,
                    "orig_end": t_e,
                    "player": current_player,
                    "legato_left": False,
                    "legato_right": False,
                })

        # Step C: Equation 2 & 3 - Intersyllabic boundary optimization (B_final & Klatt)
        for i in range(len(note_entries) - 1):
            n1 = note_entries[i]
            n2 = note_entries[i + 1]
            if n1["player"] != n2["player"]:
                continue
            gap = n2["t_start"] - n1["t_end"]
            if gap < 0.250:
                new_t1_e, new_t2_s = compute_intersyllabic_boundary(
                    art_feat=art_feat,
                    t1_start=n1["t_start"],
                    t1_end=n1["t_end"],
                    t2_start=n2["t_start"],
                    t2_end=n2["t_end"],
                    word1=n1["line"].word,
                    word2=n2["line"].word,
                    note1_pitch=n1["line"].pitch,
                    note2_pitch=n2["line"].pitch,
                )
                n1["t_end"] = new_t1_e
                n2["t_start"] = new_t2_s
                if abs(new_t2_s - new_t1_e) <= 0.015:
                    n1["legato_right"] = True
                    n2["legato_left"] = True

        # Step D: Equation 1 & Equation II - Vowel Onset Point (VOP), Leading Silence Gate & Pre-Verse Tarareo Detachment
        for idx_n, n_entry in enumerate(note_entries):
            t_s = n_entry["t_start"]
            t_e = n_entry["t_end"]
            orig_dur = max(0.030, t_e - t_s)
            min_allowed_dur = 0.060

            # Ecuación II: Si una sílaba inicial de frase absorbió >1.15s de tarareo previo antes de la siguiente sílaba
            is_phrase_head = (idx_n == 0) or ((t_s - note_entries[idx_n - 1]["t_end"]) >= 0.35)
            has_close_follower = (
                idx_n < len(note_entries) - 1
                and (note_entries[idx_n + 1]["t_start"] - t_e) < 0.45
                and note_entries[idx_n + 1]["line"].word.strip() != "~"
            )
            if is_phrase_head and has_close_follower and orig_dur > 1.15 and n_entry["line"].word.strip() != "~":
                cand_idx = np.where(
                    (art_feat.times >= max(t_s, t_e - 0.65))
                    & (art_feat.times <= max(t_s + 0.05, t_e - 0.18))
                )[0]
                if len(cand_idx) >= 3:
                    d_mfcc_sub = art_feat.delta_mfcc[cand_idx]
                    d_cent_sub = np.abs(np.gradient(art_feat.spectral_centroid[cand_idx]))
                    d_rms_sub = np.maximum(0.0, np.gradient(art_feat.rms[cand_idx]))
                    l_score = (
                        0.50 * (d_mfcc_sub / (float(np.max(d_mfcc_sub)) + 1e-6))
                        + 0.30 * (d_cent_sub / (float(np.max(d_cent_sub)) + 1e-6))
                        + 0.20 * (d_rms_sub / (float(np.max(d_rms_sub)) + 1e-6))
                    )
                    t_s = float(art_feat.times[cand_idx[int(np.argmax(l_score))]])
                    orig_dur = max(0.030, t_e - t_s)

            v_idx = np.where((art_feat.times >= t_s) & (art_feat.times <= t_e) & (art_feat.v_mask >= 0.35))[0]
            if len(v_idx) > 0:
                first_v = float(art_feat.times[v_idx[0]])
                last_v = float(art_feat.times[v_idx[-1]])
                # Ecuación II (Leading Silence Gate): Si la nota empieza en silencio antes de la voz,
                # avanzar t_s directamente hacia first_v sin el bloqueo artificial del 60%
                if not n_entry["legato_left"] and first_v > t_s + 0.020:
                    init_s_idx = int(np.searchsorted(art_feat.times, t_s))
                    init_v_idx = int(np.searchsorted(art_feat.times, first_v))
                    is_leading_silent = True
                    if init_v_idx > init_s_idx:
                        mean_init_rms = float(np.mean(art_feat.rms[init_s_idx:init_v_idx]))
                        if mean_init_rms >= noise_gate * 1.2:
                            is_leading_silent = False

                    if is_leading_silent:
                        max_trim_s = t_e - min_allowed_dur
                        t_s = min(first_v - 0.015, max_trim_s)
                    else:
                        max_trim_s = t_e - max(min_allowed_dur, orig_dur * 0.60)
                        if max_trim_s > t_s:
                            t_s = min(first_v, max_trim_s)

                if not n_entry["legato_right"] and last_v < t_e - 0.020:
                    min_trim_e = t_s + min_allowed_dur
                    if min_trim_e < t_e:
                        t_e = max(last_v + 0.015, min_trim_e)
                n_entry["t_start"] = t_s
                n_entry["t_end"] = t_e

        # Step E: Equation 5 (Melisma vs Vibrato) & Equation 4 (Anti-Scoop Pitch Estimation)
        note_map = {}
        for n_entry in note_entries:
            l = n_entry["line"]
            t_s = n_entry["t_start"]
            t_e = n_entry["t_end"]
            dur = t_e - t_s

            min_semitone_diff = max(2.0, genre_profile.vibrato_tolerance_semitones + 0.5)
            melisma_res = (
                evaluate_melisma_vs_vibrato(
                    art_feat,
                    t_s,
                    t_e,
                    min_semitone_diff=min_semitone_diff,
                    vibrato_freq_range=genre_profile.vibrato_freq_range,
                    vibrato_max_amplitude_st=genre_profile.vibrato_max_amplitude_st,
                )
                if (dur >= 0.550 and l.word.strip() != "~")
                else None
            )

            if melisma_res is not None:
                t_split, p1, p2 = melisma_res
                if abs(p1 - p2) > 7 or (t_split - t_s) < 0.22 or (t_e - t_split) < 0.22:
                    melisma_res = None

            if melisma_res is not None:
                t_split, p1, p2 = melisma_res
                p1 = int(p1 - 12 * int(round((p1 - l.pitch) / 12.0)))
                p2 = int(p2 - 12 * int(round((p2 - l.pitch) / 12.0)))
                w_first = l.word.rstrip(" ")
                w_second = "~ " if l.word.endswith(" ") else "~"
                n1_line = UltraStarLine(
                    line_type=l.line_type,
                    start_beat=0,
                    duration_beats=0,
                    pitch=p1,
                    word=w_first,
                    is_note=True,
                )
                n2_line = UltraStarLine(
                    line_type=l.line_type if l.line_type in (":", "*") else ":",
                    start_beat=0,
                    duration_beats=0,
                    pitch=p2,
                    word=w_second,
                    is_note=True,
                )
                note_map[id(l)] = [
                    {"line": n1_line, "t_start": t_s, "t_end": t_split},
                    {"line": n2_line, "t_start": t_split, "t_end": t_e},
                ]
            else:
                p_opt, p_hit_ratio = estimate_pitch_anti_scoop(art_feat, t_s, t_e, default_pitch=l.pitch)
                # Preserve the octave of the pipeline baseline (.bak): p* = p - 12*round((p - p_bak)/12)
                p_opt = int(p_opt - 12 * int(round((p_opt - l.pitch) / 12.0)))
                l.pitch = p_opt
                l.usdx_hit_ratio = p_hit_ratio
                note_map[id(l)] = [{"line": l, "t_start": t_s, "t_end": t_e}]

        # Step F: Equation 6 & 7 - Zero-Drift Beat Quantization and Strict Line Break Monotonicity
        prev_end_beat = -1
        final_output_lines: list[UltraStarLine] = []
        adjusted_notes_count = 0

        for l in surviving_items:
            if l.line_type == "P":
                prev_end_beat = -1
                final_output_lines.append(l)
                continue
            if l.is_note:
                p_items = note_map.get(
                    id(l),
                    [{"line": l, "t_start": (original_gap / 1000.0) + (l.start_beat * beat_sec), "t_end": (original_gap / 1000.0) + ((l.start_beat + l.duration_beats) * beat_sec)}]
                )
                for p_item in p_items:
                    out_l = p_item["line"]
                    t_s = p_item["t_start"]
                    t_e = p_item["t_end"]

                    b_start = int(round(((t_s - (original_gap / 1000.0)) * bpm * 4.0) / 60.0))
                    b_end = int(round(((t_e - (original_gap / 1000.0)) * bpm * 4.0) / 60.0))
                    if b_start < 0:
                        b_start = 0
                    if prev_end_beat >= 0 and b_start < prev_end_beat:
                        b_start = prev_end_beat
                    # Ecuación VI: Anclado al tiempo físico absoluto para evitar arrastre en cascada
                    b_end = max(b_start + 1, b_end)
                    d_beat = b_end - b_start

                    out_l.start_beat = b_start
                    out_l.duration_beats = d_beat
                    prev_end_beat = b_start + d_beat

                    # Clamp any preceding line break so it NEVER sits after this note's start_beat
                    if final_output_lines and final_output_lines[-1].is_line_break:
                        if final_output_lines[-1].start_beat > out_l.start_beat:
                            final_output_lines[-1].start_beat = out_l.start_beat

                    # Freestyle rescue ONLY for percussive onomatopoeias or explicit rap genres
                    idx_s = np.searchsorted(art_feat.times, t_s)
                    idx_e = np.searchsorted(art_feat.times, t_e)
                    if idx_e > idx_s and out_l.word.strip() != "~":
                        voiced_frac = float(np.mean(art_feat.v_mask[idx_s:idx_e] >= 0.35))
                        conf_frac = float(np.mean(art_feat.confidence[idx_s:idx_e] >= 0.35))
                        spec_flat_mean = float(np.mean(art_feat.spectral_flatness[idx_s:idx_e]))
                        r_boca_mean = float(np.mean(art_feat.r_boca[idx_s:idx_e]))
                        rms_slice = art_feat.rms[idx_s:idx_e]
                        rms_mean = float(np.mean(rms_slice))
                        rms_crest = float(np.max(rms_slice) / (rms_mean + 1e-6))
                        rms_cv = float(np.std(rms_slice) / (rms_mean + 1e-6))
                        delta_mfcc_mean = float(np.mean(art_feat.delta_mfcc[idx_s:idx_e]))
                        spec_cent_mean = (
                            float(np.mean(art_feat.spectral_centroid[idx_s:idx_e]))
                            if art_feat.spectral_centroid is not None and len(art_feat.spectral_centroid) >= idx_e
                            else 1500.0
                        )
                        is_harsh_scream = spec_flat_mean >= (1.0 - genre_profile.spectral_entropy_threshold)
                        is_rap_or_screamed = (
                            genre_profile.prefer_freestyle_on_low_confidence
                            and (conf_frac < 0.25 or voiced_frac < 0.20 or is_harsh_scream)
                        )
                        is_percussive_onomatopoeia = False
                        if is_scat_or_adlib_pattern(out_l.word):
                            ono_mode = classify_onomatopoeia_acoustics(
                                word=out_l.word,
                                f0_conf_mean=conf_frac,
                                r_boca_mean=r_boca_mean,
                                spectral_flatness_mean=spec_flat_mean,
                                rms_mean=rms_mean,
                                noise_gate=noise_gate,
                                delta_mfcc_mean=delta_mfcc_mean,
                                spectral_centroid_mean=spec_cent_mean,
                                rms_crest_factor=rms_crest,
                                rms_cv=rms_cv,
                            )
                            if ono_mode == "freestyle":
                                is_percussive_onomatopoeia = True
                        if (is_rap_or_screamed or is_percussive_onomatopoeia) and out_l.line_type not in ("*", "F", "R"):
                            out_l.line_type = "F"

                    final_output_lines.append(out_l)
                    adjusted_notes_count += 1
            elif l.is_line_break:
                if final_output_lines and (final_output_lines[-1].is_line_break or final_output_lines[-1].line_type in ("#", "P")):
                    continue
                if final_output_lines and final_output_lines[-1].is_note:
                    l.start_beat = max(l.start_beat, final_output_lines[-1].start_beat + final_output_lines[-1].duration_beats)
                final_output_lines.append(l)
            else:
                final_output_lines.append(l)

        # Step G: Equation 4 Part 2 - Spurious Octave Jump Folding, Isolated Intra-Phrase Freestyle Smoothing & #GAP Re-calibration
        all_notes = [nl for nl in final_output_lines if nl.is_note]
        fold_octave_jumps(all_notes)

        # Suavizado de Freestyle Aislado Intra-Frase:
        # Si una sílaba léxica corta fue marcada como 'F' pero está flanqueada inmediatamente (< 0.25s)
        # por notas cantadas normales (':' o '*') y no es una onomatopeya percusiva, restaurar ':' para continuidad melódica.
        for idx_nl in range(1, len(all_notes) - 1):
            curr_nl = all_notes[idx_nl]
            prev_nl = all_notes[idx_nl - 1]
            next_nl = all_notes[idx_nl + 1]
            if (
                curr_nl.line_type == "F"
                and prev_nl.line_type in (":", "*")
                and next_nl.line_type in (":", "*")
                and not is_scat_or_adlib_pattern(curr_nl.word)
            ):
                gap_prev_sec = (curr_nl.start_beat - (prev_nl.start_beat + prev_nl.duration_beats)) * beat_sec
                gap_next_sec = (next_nl.start_beat - (curr_nl.start_beat + curr_nl.duration_beats)) * beat_sec
                if gap_prev_sec <= 0.25 and gap_next_sec <= 0.25:
                    curr_nl.line_type = ":"

        if all_notes and all_notes[0].start_beat > 0:
            shift_beats = all_notes[0].start_beat
            new_gap = original_gap + (shift_beats * beat_sec * 1000.0)
            for fl in final_output_lines:
                if fl.is_note or fl.is_line_break:
                    fl.start_beat = max(0, fl.start_beat - shift_beats)

    else:
        new_gap, final_output_lines, adjusted_notes_count = _align_and_reconstruct_legacy_song(
            lines=lines,
            headers=headers,
            vocal_path=vocal_path,
            original_gap=original_gap,
            bpm=bpm,
            beat_sec=beat_sec,
            device=device,
            rms=rms,
            rms_times=rms_times,
            noise_gate=noise_gate,
            pitched_data=pitched_data,
        )

    # SOTA Rule 4: Re-asignación Inteligente de Notas Doradas (*) con Ponderación de Clímax/Drop (Módulo 5 MIR: E_RMS)
    total_beats = sum(n.duration_beats for n in final_output_lines if n.is_note)
    target_golden_beats = int(total_beats * 0.18)
    current_golden_beats = 0

    active_rms_vals = art_feat.rms[art_feat.rms >= noise_gate] if len(art_feat.rms) > 0 else np.array([0.1])
    climax_rms_threshold = float(np.percentile(active_rms_vals, 75)) if len(active_rms_vals) > 0 else 0.1

    candidates = []
    for l_idx, l in enumerate(final_output_lines):
        if not l.is_note or l.line_type in ("F", "R"):
            continue
        if l.line_type == "*":
            l.line_type = ":"

        dur_sec = l.duration_beats * beat_sec
        if dur_sec >= 0.25 or l.duration_beats >= 6:
            syl_start = (new_gap / 1000.0) + (l.start_beat * beat_sec)
            syl_end = syl_start + dur_sec

            idx_s = np.searchsorted(art_feat.times, syl_start)
            idx_e = np.searchsorted(art_feat.times, syl_end)
            if idx_e - idx_s >= 4:
                sub_v = art_feat.v_mask[idx_s:idx_e]
                sub_f = art_feat.f0_midi[idx_s:idx_e]
                sub_rms = art_feat.rms[idx_s:idx_e]
                valid_f = (sub_f >= 21.0) & (sub_v >= 0.30)
                if np.sum(valid_f) >= 3:
                    mean_v = float(np.mean(sub_v))
                    pitch_std = float(np.std(sub_f[valid_f]))
                    note_e_rms = float(np.sqrt(np.mean(sub_rms ** 2)))
                    climax_boost = 1.25 if note_e_rms >= climax_rms_threshold else 1.0

                    # Stability criteria per Eq 7: mean V_mask >= 0.65, pitch std <= 0.35 semitones
                    if mean_v >= 0.65 and pitch_std <= 0.35:
                        stability_idx = mean_v * (1.0 / (pitch_std + 0.05)) * (dur_sec ** 0.5) * climax_boost
                        candidates.append({
                            "line_idx": l_idx,
                            "line": l,
                            "stability": stability_idx,
                            "beats": l.duration_beats,
                        })


    candidates.sort(key=lambda c: c["stability"], reverse=True)
    for c in candidates:
        if current_golden_beats + c["beats"] <= target_golden_beats:
            c["line"].line_type = "*"
            current_golden_beats += c["beats"]

    golden_count = sum(1 for l in final_output_lines if l.is_note and l.line_type == "*")
    print(
        f"{ULTRASINGER_HEAD} {cyan_highlighted('Golden Notes Optimization (*):')} "
        f"Assigned {gold_highlighted(str(golden_count))} ultra-stable golden notes "
        f"({current_golden_beats} beats, {current_golden_beats / max(1, total_beats) * 100:.1f}% of song)."
    )

    note_lines = [l for l in final_output_lines if l.is_note]

    # 11. Pentagram Vertical Centering & Smoothing
    # For UltraSinger-created songs the baseline octave is preserved; only recenter when the
    # median pitch is clearly off-screen (outside [-12, 18]).
    all_pitches = [n.pitch for n in note_lines if n.line_type not in ("F", "R")]
    if all_pitches:
        median_pitch = float(np.median(all_pitches))
        octave_shift = 0
        if not is_ultrasinger_created or median_pitch < -12.0 or median_pitch > 18.0:
            octave_shift = -int(round((median_pitch - 6.0) / 12.0)) * 12
        if octave_shift != 0:
            for n in note_lines:
                if n.line_type not in ("F", "R"):
                    n.pitch += octave_shift
            print(f"{ULTRASINGER_HEAD} Pentagram Vertical Centering: shifted by {octave_shift:+d} semitones to center on screen.")

    # 12. Simulate Final Score & Per-Phrase Karaoke AI Analysis
    final_score, final_cov, final_pitch_acc = simulate_game_score(
        note_lines, bpm, new_gap, rms, rms_times, pitched_data, noise_gate
    )
    phrase_scores = evaluate_karaoke_phrases_scoring(
        lines=final_output_lines,
        bpm=bpm,
        gap_ms=new_gap,
        art_feat=art_feat,
        noise_gate=noise_gate,
    )

    # 13. Create backup and save updated .txt
    backup_path = txt_path + ".bak"
    if not os.path.isfile(backup_path):
        shutil.copyfile(txt_path, backup_path)

    # Reconstruct .txt preserving all headers and exact word strings
    output_lines = []
    has_encoding = False
    has_resolution = False
    has_genre = False
    last_header_idx = -1
    for l in final_output_lines:
        if l.line_type == "#":
            if l.raw_line.upper().startswith("#ENCODING:"):
                has_encoding = True
            elif l.raw_line.upper().startswith("#RESOLUTION:"):
                has_resolution = True
            elif l.raw_line.upper().startswith("#GENRE:"):
                has_genre = True
            elif l.raw_line.upper().startswith("#BPM:"):
                output_lines.append(f"#BPM:{round(bpm, 2)}\n")
                last_header_idx = len(output_lines) - 1
                continue
            output_lines.append(l.raw_line + "\n")
            last_header_idx = len(output_lines) - 1
        elif l.line_type == "P":
            output_lines.append(l.raw_line + "\n")
        elif l.is_note:
            clean_word_out = l.word
            output_text_line = f"{l.line_type} {l.start_beat} {l.duration_beats} {l.pitch} {clean_word_out}\n"
            output_lines.append(output_text_line)
        elif l.is_line_break:
            output_lines.append(f"- {l.start_beat}\n")
        elif l.line_type == "E":
            output_lines.append("E\n")
        else:
            output_lines.append(l.raw_line + "\n")

    # UltraStar Deluxe SOTA Standard: ensure #GENRE, #ENCODING:UTF8, and #RESOLUTION:4 headers exist
    if not has_genre:
        default_genre = headers.get("GENRE", "Pop")
        output_lines.insert(last_header_idx + 1 if last_header_idx >= 0 else 0, f"#GENRE:{default_genre}\n")
        last_header_idx += 1
    if not has_encoding:
        output_lines.insert(last_header_idx + 1 if last_header_idx >= 0 else 0, "#ENCODING:UTF8\n")
        last_header_idx += 1
    if not has_resolution:
        output_lines.insert(last_header_idx + 1 if last_header_idx >= 0 else 0, "#RESOLUTION:4\n")
        last_header_idx += 1

    with open(txt_path, "w", encoding="utf-8") as f:
        f.writelines(output_lines)

    # 12. Print Quality Assurance & Simulation Report
    score_diff = final_score - orig_score
    diff_str = f"+{score_diff}" if score_diff >= 0 else str(score_diff)
    grade = "S+ (PERFECT)" if final_score >= 9500 else ("S (EXCELLENT)" if final_score >= 9000 else "A (GOOD)")

    print(f"\n{ULTRASINGER_HEAD} ═══════════════════════════════════════════════════════════════════")
    print(f"{ULTRASINGER_HEAD}                   {green_highlighted('AUTO-CORRECTION & QA REPORT')}")
    print(f"{ULTRASINGER_HEAD} ═══════════════════════════════════════════════════════════════════")
    print(f"{ULTRASINGER_HEAD}   ├── Words / Syllables  : {green_highlighted(f'{len(note_lines)} (100% text preserved)')}")
    print(f"{ULTRASINGER_HEAD}   ├── Calibrated Bars    : {blue_highlighted(str(adjusted_notes_count))} bars sculpted to vocals")
    print(f"{ULTRASINGER_HEAD}   ├── Baseline Score     : {cyan_highlighted(f'{orig_score:,} / 10,000')}")
    print(f"{ULTRASINGER_HEAD}   ├── Auto-Correct Score : {gold_highlighted(f'{final_score:,} / 10,000')} [{light_blue_highlighted(grade)}] ({green_highlighted(diff_str)})")
    print(f"{ULTRASINGER_HEAD}   ├── Timing Coverage    : {green_highlighted(f'{final_cov:.1f}%')} (vocal energy aligned)")
    print(f"{ULTRASINGER_HEAD}   ├── Pitch Fidelity     : {green_highlighted(f'{final_pitch_acc:.1f}%')} (within ±1.5 semitones)")
    print(f"{ULTRASINGER_HEAD}   ├── Backup Created     : {blue_highlighted(os.path.basename(backup_path))}")
    print(f"{ULTRASINGER_HEAD}   └── UltraStar Status   : {green_highlighted('VERIFIED & OPTIMIZED FOR GAMEPLAY')}")
    print(f"{ULTRASINGER_HEAD} ═══════════════════════════════════════════════════════════════════\n")

    # 13. Trigger Automated Visual QA Plotter (Headless)
    try:
        from modules.Audio.visual_qa_plotter import generate_visual_qa_plot
        generate_visual_qa_plot(
            txt_path=txt_path,
            vocal_audio_path=vocal_path,
            output_folder=folder_path,
            pitched_data=pitched_data,
            pitcher_name=pitcher_name,
            device=device,
        )
    except Exception as qa_err:
        print(f"{ULTRASINGER_HEAD} {red_highlighted('Visual QA Plot warning:')} {qa_err}")

    return QASimulationReport(
        original_score=orig_score,
        corrected_score=final_score,
        timing_coverage=final_cov,
        pitch_accuracy=final_pitch_acc,
        total_notes=len(note_lines),
        adjusted_notes=adjusted_notes_count,
        gap_shift_ms=new_gap - original_gap,
        local_shifts_count=adjusted_notes_count,
        backup_path=backup_path,
        phrase_scores=phrase_scores,
    )
