try:
    import librosa
except Exception:
    librosa = None
try:
    import soundfile as sf
except Exception:
    sf = None
import numpy as np

from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted


def _octave_fix(bpm: float) -> float:
    """Normalize BPM into standard musical range (80–180 BPM)."""
    while bpm < 80.0:
        bpm *= 2.0
    while bpm > 180.0:
        bpm /= 2.0
    return bpm


def compute_autocorrelation_tempo(
    envelope: np.ndarray,
    frame_rate_hz: float,
    min_bpm: float = 80.0,
    max_bpm: float = 180.0,
) -> float:
    """Módulo 2 MIR: Extracción de Tempo y Pulso (BPM) mediante Autocorrelación:
      R(k) = sum_n x[n] * x[n - k]
    Los picos máximos de R(k) respecto al desfase k determinan la distancia exacta entre beats.
    """
    if envelope is None or len(envelope) < 8 or frame_rate_hz <= 0:
        return 120.0

    x = np.asarray(envelope, dtype=np.float64)
    x = x - np.mean(x)
    if np.max(np.abs(x)) < 1e-8:
        return 120.0

    # R(k) = sum_n x[n] x[n - k] para k >= 0
    full_corr = np.correlate(x, x, mode="full")
    r_k = full_corr[len(full_corr) // 2 :]

    min_lag = max(1, int(np.floor(60.0 * frame_rate_hz / max_bpm)))
    max_lag = min(len(r_k) - 1, int(np.ceil(60.0 * frame_rate_hz / min_bpm)))
    if max_lag <= min_lag:
        return 120.0

    search_region = r_k[min_lag : max_lag + 1]
    peak_offset = int(np.argmax(search_region))
    best_lag = min_lag + peak_offset

    # Interpolación parabólica sub-muestra alrededor del pico k*
    refined_lag = float(best_lag)
    if 1 <= best_lag < len(r_k) - 1:
        y0, y1, y2 = float(r_k[best_lag - 1]), float(r_k[best_lag]), float(r_k[best_lag + 1])
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 1e-9:
            delta = 0.5 * (y0 - y2) / denom
            if -0.5 <= delta <= 0.5:
                refined_lag = best_lag + delta

    bpm = (60.0 * frame_rate_hz) / max(1e-6, refined_lag)
    return _octave_fix(float(bpm))


def get_bpm_from_data(data, sampling_rate):
    """Get real bpm from audio data with autocorrelation R(k) and beat-position regression refinement."""
    hop_length = 512
    env_fps = float(sampling_rate) / float(hop_length)
    onset_env = None

    try:
        onset_env = librosa.onset.onset_strength(y=data, sr=sampling_rate, hop_length=hop_length)
        tempo, beat_frames = librosa.beat.beat_track(
            onset_envelope=onset_env, sr=sampling_rate, hop_length=hop_length
        )
        base_bpm = float(tempo[0]) if hasattr(tempo, "__len__") else float(tempo)

        # Refine using beat-position regression (more precise than tempogram bins)
        if len(beat_frames) > 10:
            beat_times = librosa.frames_to_time(beat_frames, sr=sampling_rate, hop_length=hop_length)
            indices = np.arange(len(beat_times))
            slope, _ = np.polyfit(indices, beat_times, 1)
            if slope > 0:
                bpm_reg = 60.0 / slope
                if abs(bpm_reg - base_bpm) / max(1.0, base_bpm) < 0.12:
                    base_bpm = bpm_reg

        # Refinar con Autocorrelación directa R(k) = sum_n x[n] x[n-k]
        ac_bpm = compute_autocorrelation_tempo(onset_env, env_fps)
        norm_base = _octave_fix(base_bpm)
        if abs(ac_bpm - norm_base) / max(1.0, norm_base) < 0.08:
            base_bpm = 0.65 * norm_base + 0.35 * ac_bpm
        else:
            base_bpm = norm_base
    except Exception:
        if onset_env is None and librosa is not None:
            onset_env = librosa.onset.onset_strength(y=data, sr=sampling_rate, hop_length=hop_length)
        if onset_env is not None:
            base_bpm = compute_autocorrelation_tempo(onset_env, env_fps)
        else:
            base_bpm = 120.0

    final_bpm = _octave_fix(base_bpm)
    print(f"{ULTRASINGER_HEAD} BPM is {blue_highlighted(str(round(final_bpm, 2)))}")
    return final_bpm


def get_bpm_from_file(wav_file: str) -> float:
    """Get real bpm from audio file"""
    try:
        data, sampling_rate = sf.read(wav_file, dtype='float32')
        # Transpose if stereo to match librosa's expected format
        if len(data.shape) > 1:
            data = data.T
        # Convert to mono if stereo
        if data.ndim > 1:
            data = librosa.to_mono(data)
    except Exception:
        data, sampling_rate = librosa.load(wav_file, sr=None, mono=True)
    return get_bpm_from_data(data, sampling_rate)
