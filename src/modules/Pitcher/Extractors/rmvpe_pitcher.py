"""RMVPE Pitch Extraction Strategy using rmvpe-onnx."""
import gc
import librosa
import numpy as np

from modules.console_colors import (
    ULTRASINGER_HEAD,
    blue_highlighted,
    red_highlighted,
)
from modules.Pitcher.Extractors.base_pitcher import BasePitchExtractor
from modules.Pitcher.pitched_data import PitchedData


class RMVPEPitcher(BasePitchExtractor):
    """Robust Model for Vocal Pitch Estimation (RMVPE) for polyphonic and singing voice."""

    SAMPLE_RATE = 16000
    HOP_SEC = 0.010  # 10ms standard hop size

    def extract_pitch(self, audio_path: str, device: str = "cpu") -> PitchedData:
        print(
            f"{ULTRASINGER_HEAD} Pitching with SOTA {blue_highlighted('RMVPE')} "
            f"(worker: {red_highlighted(device)})"
        )

        try:
            from rmvpe_onnx import RMVPE
        except ImportError as exc:
            raise ImportError(
                "The 'rmvpe-onnx' package is required for RMVPE pitch detection. "
                "Install it using: pip install rmvpe-onnx"
            ) from exc

        audio, _ = librosa.load(audio_path, sr=self.SAMPLE_RATE, mono=True)

        import os
        import torch
        use_device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        model_path = os.path.expanduser("~/.cache/rmvpe/rmvpe.onnx")
        rmvpe = RMVPE(model_path=model_path, device=use_device)
        time, frequency, conf, _ = rmvpe.predict(audio, self.SAMPLE_RATE)

        times = [float(t) for t in time]
        frequencies = [float(f) for f in frequency]
        confidence = [float(c) for c in conf]

        del rmvpe
        gc.collect()

        return PitchedData(times, frequencies, confidence)
