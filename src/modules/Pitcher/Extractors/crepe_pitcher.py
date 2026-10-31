"""CREPE Pitch Extraction Strategy using torchcrepe."""
import gc
import librosa
import numpy as np
import torch

from modules.console_colors import (
    ULTRASINGER_HEAD,
    blue_highlighted,
    red_highlighted,
)
from modules.Pitcher.Extractors.base_pitcher import BasePitchExtractor
from modules.Pitcher.pitched_data import PitchedData


class CrepePitcher(BasePitchExtractor):
    """High-accuracy neural pitch estimator based on CREPE (via torchcrepe)."""

    SAMPLE_RATE = 16000
    HOP_LENGTH = 160  # 10ms frame resolution at 16kHz

    def __init__(self, model_capacity: str = "full", fmin: float = 50.0, fmax: float = 1100.0):
        self.model_capacity = model_capacity
        self.fmin = fmin
        self.fmax = fmax

    def extract_pitch(self, audio_path: str, device: str = "cpu") -> PitchedData:
        print(
            f"{ULTRASINGER_HEAD} Pitching with SOTA {blue_highlighted('CREPE')} "
            f"(capacity: {blue_highlighted(self.model_capacity)}, worker: {red_highlighted(device)})"
        )

        try:
            import torchcrepe
        except ImportError as exc:
            raise ImportError(
                "The 'torchcrepe' package is required for CREPE pitch detection. "
                "Install it using: pip install torchcrepe"
            ) from exc

        # Load and resample audio to 16kHz mono
        audio, sr = librosa.load(audio_path, sr=self.SAMPLE_RATE, mono=True)
        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        use_device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        audio_tensor = audio_tensor.to(use_device)

        batch_size = 512
        try:
            with torch.no_grad():
                pitch, periodicity = torchcrepe.predict(
                    audio_tensor,
                    sample_rate=self.SAMPLE_RATE,
                    hop_length=self.HOP_LENGTH,
                    fmin=self.fmin,
                    fmax=self.fmax,
                    model=self.model_capacity,
                    batch_size=batch_size,
                    device=use_device,
                    return_periodicity=True,
                )
        except torch.OutOfMemoryError:
            print(f"{ULTRASINGER_HEAD} VRAM low, retrying CREPE with batch_size 128...")
            gc.collect()
            torch.cuda.empty_cache()
            with torch.no_grad():
                pitch, periodicity = torchcrepe.predict(
                    audio_tensor,
                    sample_rate=self.SAMPLE_RATE,
                    hop_length=self.HOP_LENGTH,
                    fmin=self.fmin,
                    fmax=self.fmax,
                    model=self.model_capacity,
                    batch_size=128,
                    device=use_device,
                    return_periodicity=True,
                )

        pitch_np = pitch.squeeze(0).cpu().numpy()
        confidence_np = periodicity.squeeze(0).cpu().numpy()

        times = [float(i * self.HOP_LENGTH / self.SAMPLE_RATE) for i in range(len(pitch_np))]
        frequencies = [float(f) for f in pitch_np]
        confidence = [float(c) for c in confidence_np]

        # Clean VRAM
        del audio_tensor, pitch, periodicity
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return PitchedData(times, frequencies, confidence)
