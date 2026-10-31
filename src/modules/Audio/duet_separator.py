"""Duet and backing vocals separation module using spectral clustering."""

import os
import numpy as np
import librosa
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from modules.console_colors import (
    ULTRASINGER_HEAD,
    blue_highlighted,
    green_highlighted,
    cyan_highlighted,
)
from modules.Midi.MidiSegment import MidiSegment
from modules.Audio.stereo_vocal_analyzer import compute_stereo_vocal_spatial_metrics


def group_midi_segments_into_phrases(
    midi_segments: list[MidiSegment],
    silence_split_duration: float,
) -> list[list[MidiSegment]]:
    """Group continuous midi segments into natural vocal phrases based on pauses.

    Args:
        midi_segments: List of MidiSegment objects
        silence_split_duration: Pause threshold in seconds for line breaks

    Returns:
        List of phrases, where each phrase is a list of MidiSegment objects.
    """
    if not midi_segments:
        return []

    phrases = []
    current_phrase = []
    separated_word_silence = []

    for i, segment in enumerate(midi_segments):
        current_phrase.append(segment)

        if i < len(midi_segments) - 1:
            silence = midi_segments[i + 1].start - segment.end
        else:
            silence = 0.0

        if not segment.word.endswith(" "):
            separated_word_silence.append(silence)
            continue

        is_last = i == len(midi_segments) - 1
        threshold = silence_split_duration if silence_split_duration > 0 else 0.35
        has_long_pause = (
            silence > threshold or any(s > threshold for s in separated_word_silence)
        )

        if is_last or has_long_pause:
            phrases.append(current_phrase)
            current_phrase = []
            separated_word_silence = []

    if current_phrase:
        phrases.append(current_phrase)

    return phrases


def extract_phrase_features(
    vocal_audio_path: str,
    phrase: list[MidiSegment],
) -> np.ndarray:
    """Extract acoustic timbre and pitch features for a single phrase.

    Features include:
    - Median and standard deviation of MIDI pitch
    - Mean spectral centroid (brightness)
    - Mean spectral rolloff
    - 13 MFCC coefficients (vocal tract timbre)

    Args:
        vocal_audio_path: Path to vocals audio file
        phrase: List of MidiSegment objects in this phrase

    Returns:
        1D float32 numpy array of features.
    """
    # 1. Pitch features from phrase notes
    midi_numbers = []
    for seg in phrase:
        try:
            m = librosa.note_to_midi(seg.note)
            midi_numbers.append(m)
        except Exception:
            pass

    median_pitch = float(np.median(midi_numbers)) if midi_numbers else 60.0
    std_pitch = float(np.std(midi_numbers)) if len(midi_numbers) > 1 else 0.0

    # 2. Timbre features from audio
    start_time = float(phrase[0].start)
    end_time = float(phrase[-1].end)
    duration = max(0.05, end_time - start_time)

    spectral_centroid = 1500.0
    spectral_rolloff = 3000.0
    mfccs = np.zeros(13, dtype=np.float32)

    if os.path.isfile(vocal_audio_path):
        try:
            y, sr = librosa.load(
                vocal_audio_path,
                sr=22050,
                mono=True,
                offset=start_time,
                duration=duration,
            )
            if len(y) > 512:
                sc = librosa.feature.spectral_centroid(y=y, sr=sr)
                spectral_centroid = float(np.mean(sc))

                ro = librosa.feature.spectral_rolloff(y=y, sr=sr)
                spectral_rolloff = float(np.mean(ro))

                mfcc_feat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                mfccs = np.mean(mfcc_feat, axis=1).astype(np.float32)
        except Exception:
            pass

    # 3. Spatial stereo features (IACC and CSR for discriminating center lead vs wide backing vocals)
    spatial_metrics = compute_stereo_vocal_spatial_metrics(
        vocal_audio_path, start_time=start_time, duration=duration
    )
    iacc = float(spatial_metrics["iacc"])
    csr_db = float(spatial_metrics["csr_db"])

    features = [median_pitch, std_pitch, spectral_centroid, spectral_rolloff, iacc, csr_db]
    features.extend(mfccs.tolist())
    return np.array(features, dtype=np.float32)


def extract_phrase_ecapa_embeddings(
    vocal_audio_path: str,
    phrases: list[list[MidiSegment]],
    device: str = "cpu",
) -> np.ndarray | None:
    """Extract ECAPA-TDNN speaker d-vector embeddings for each phrase using SpeechBrain.

    Args:
        vocal_audio_path: Path to vocal audio file.
        phrases: List of phrases.
        device: 'cuda' or 'cpu'.

    Returns:
        2D numpy array of shape (num_phrases, embedding_dim) or None if unavailable.
    """
    try:
        import gc
        import torch
        import torchaudio
        from speechbrain.inference.speaker import EncoderClassifier

        cache_dir = os.path.expanduser("~/.cache/speechbrain/spkrec-ecapa-voxceleb")
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=cache_dir,
            run_opts={"device": device},
        )

        waveform, sr = torchaudio.load(vocal_audio_path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
            sr = 16000

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        embeddings = []
        for phrase in phrases:
            start_sec = float(phrase[0].start)
            end_sec = float(phrase[-1].end)
            start_sample = max(0, int(start_sec * sr))
            end_sample = min(waveform.shape[1], int(end_sec * sr))

            if end_sample - start_sample < 1600:  # Minimum 100ms
                end_sample = min(waveform.shape[1], start_sample + 1600)

            chunk = waveform[:, start_sample:end_sample].to(device)
            with torch.inference_mode():
                emb = classifier.encode_batch(chunk)
                emb_np = emb.squeeze().detach().cpu().numpy()
                embeddings.append(emb_np)

        del classifier
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return np.vstack(embeddings)
    except Exception as exc:
        print(
            f"{ULTRASINGER_HEAD} ECAPA-TDNN embedding notice: ({exc}). "
            f"Falling back smoothly to MFCC + spatial clustering."
        )
        return None


def separate_duet_phrases(
    vocal_audio_path: str,
    midi_segments: list[MidiSegment],
    silence_split_duration: float,
    p1_name: str = "Singer 1",
    p2_name: str = "Singer 2",
    engine: str = "mfcc_kmeans",
) -> tuple[list[int], list[list[MidiSegment]], str, str]:
    """Classify song phrases into Player 1 (P1) and Player 2 (P2) using vocal clustering.

    Args:
        vocal_audio_path: Path to vocal audio file
        midi_segments: Full list of MidiSegment objects
        silence_split_duration: Silence split duration in seconds
        p1_name: Display name for Player 1
        p2_name: Display name for Player 2
        engine: Diarization engine ('mfcc_kmeans' or 'ecapa_embeddings')

    Returns:
        tuple of (phrase_player_ids [1 or 2], phrases, p1_name, p2_name)
    """
    phrases = group_midi_segments_into_phrases(midi_segments, silence_split_duration)
    if not phrases:
        return [], [], p1_name, p2_name

    if len(phrases) < 2:
        return [1] * len(phrases), phrases, p1_name, p2_name

    X = None
    engine_key = (engine or "mfcc_kmeans").lower().strip()

    if "ecapa" in engine_key or "speechbrain" in engine_key:
        print(
            f"{ULTRASINGER_HEAD} Duet Diarization Engine: SOTA {blue_highlighted('ECAPA-TDNN embeddings')} "
            f"(speechbrain d-vector projection)"
        )
        X = extract_phrase_ecapa_embeddings(vocal_audio_path, phrases)

    if X is None:
        if engine_key != "mfcc_kmeans":
            print(
                f"{ULTRASINGER_HEAD} Duet Diarization Engine: {blue_highlighted('MFCC + Spatial Clustering')} (Default)"
            )
        feature_list = [extract_phrase_features(vocal_audio_path, p) for p in phrases]
        X = np.vstack(feature_list)

    # Standardize features for balanced clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-Means clustering into 2 singers
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    # Guarantee P1 is the singer who sings the first phrase
    p1_cluster = labels[0]
    player_ids = [1 if label == p1_cluster else 2 for label in labels]

    # Explicitly stamp player_id onto every MidiSegment within each phrase
    for phrase, pid in zip(phrases, player_ids):
        for seg in phrase:
            seg.player_id = pid

    p1_count = sum(1 for pid in player_ids if pid == 1)
    p2_count = sum(1 for pid in player_ids if pid == 2)

    print(f"{ULTRASINGER_HEAD} {cyan_highlighted('Duet Vocal Separation (P1 & P2):')}")
    print(
        f"{ULTRASINGER_HEAD}   ├── Player 1 ({blue_highlighted(p1_name)}): "
        f"{green_highlighted(str(p1_count))} phrases"
    )
    print(
        f"{ULTRASINGER_HEAD}   └── Player 2 ({blue_highlighted(p2_name)}): "
        f"{green_highlighted(str(p2_count))} phrases"
    )

    return player_ids, phrases, p1_name, p2_name
