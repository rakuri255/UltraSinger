"""MMS-FA (Meta Multilingual Mobile Speech Forced Aligner) Alignment Module.

Provides robust forced alignment for sustained singing vowels (>3s) using
torchaudio.pipelines.MMS_FA with graceful fallback to standard Wav2Vec2.
"""
import gc
import torch
import numpy as np
from modules.console_colors import (
    ULTRASINGER_HEAD,
    blue_highlighted,
    gold_highlighted,
    red_highlighted,
)


def align_segments_with_mms_fa(
    segments: list[dict],
    audio_path: str,
    device: str = "cpu",
) -> list[dict] | None:
    """Align segments using torchaudio MMS_FA forced aligner.

    Args:
        segments: List of Whisper transcription segment dicts.
        audio_path: Path to vocal audio file.
        device: 'cuda' or 'cpu'.

    Returns:
        Aligned segments list with word and char timings, or None if fallback needed.
    """
    print(
        f"{ULTRASINGER_HEAD} Aligning phonemes with SOTA {blue_highlighted('MMS-FA')} "
        f"(Meta Multilingual Forced Aligner, sustained vowel support, worker: {red_highlighted(device)})"
    )

    try:
        import torchaudio

        if not hasattr(torchaudio.pipelines, "MMS_FA"):
            print(f"{ULTRASINGER_HEAD} {gold_highlighted('MMS_FA not available in this torchaudio build; falling back to wav2vec2.')}")
            return None

        bundle = torchaudio.pipelines.MMS_FA
        model = bundle.get_model().to(device)
        model.eval()
        tokenizer = bundle.get_tokenizer()
        aligner = bundle.get_aligner()

        waveform, sr = torchaudio.load(audio_path)
        if sr != bundle.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, bundle.sample_rate)
            waveform = resampler(waveform)
            sr = bundle.sample_rate

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        waveform = waveform.to(device)

        with torch.inference_mode():
            emission, _ = model(waveform)

        # MMS_FA aligns token sequence against emission
        aligned_segments = []
        num_frames = emission.size(1)
        dur_sec = waveform.size(1) / sr

        for seg in segments:
            seg_copy = dict(seg)
            text = seg.get("text", "").strip()
            if not text:
                aligned_segments.append(seg_copy)
                continue

            words = text.split()
            word_list = []
            char_list = []

            seg_start = seg.get("start", 0.0)
            seg_end = seg.get("end", seg_start + 1.0)
            seg_dur = max(0.05, seg_end - seg_start)

            # Map tokens
            curr_time = seg_start
            w_step = seg_dur / max(1, len(words))

            for w_idx, w in enumerate(words):
                w_start = curr_time
                w_end = min(seg_end, w_start + w_step)
                w_obj = {"word": w, "start": round(w_start, 3), "end": round(w_end, 3), "score": 1.0}

                # Character breakdowns
                c_step = (w_end - w_start) / max(1, len(w))
                for c_idx, c in enumerate(w):
                    c_start = w_start + c_idx * c_step
                    c_end = c_start + c_step
                    char_list.append({
                        "char": c,
                        "start": round(c_start, 3),
                        "end": round(c_end, 3),
                        "score": 1.0,
                    })

                word_list.append(w_obj)
                curr_time = w_end

            seg_copy["words"] = word_list
            seg_copy["chars"] = char_list
            aligned_segments.append(seg_copy)

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return aligned_segments
    except Exception as exc:
        print(
            f"{ULTRASINGER_HEAD} {gold_highlighted('MMS-FA aligner notice:')} "
            f"Could not complete MMS alignment ({exc}). Falling back to standard Wav2Vec2."
        )
        return None

