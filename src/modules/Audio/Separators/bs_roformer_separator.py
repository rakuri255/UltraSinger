"""BS-Roformer Vocal Separation Strategy using audio-separator."""
import gc
import os
import shutil

from modules.console_colors import (
    ULTRASINGER_HEAD,
    blue_highlighted,
    red_highlighted,
)
from modules.Audio.Separators.base_separator import BaseVocalSeparator


class BSRoformerSeparator(BaseVocalSeparator):
    """Concrete separator using BS-Roformer (State-Of-The-Art vocal isolation)."""

    DEFAULT_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
    MEL_BAND_MODEL = "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt"
    KIM_VOCAL_2_MODEL = "Kim_Vocal_2.onnx"

    MODEL_ALIASES = {
        "bs_roformer": DEFAULT_MODEL,
        "bs-roformer": DEFAULT_MODEL,
        "roformer": DEFAULT_MODEL,
        "mel_band": MEL_BAND_MODEL,
        "mel_band_roformer": MEL_BAND_MODEL,
        "mel-band-roformer": MEL_BAND_MODEL,
        "mel-band": MEL_BAND_MODEL,
        "melband": MEL_BAND_MODEL,
        "kim": KIM_VOCAL_2_MODEL,
        "kim_vocal": KIM_VOCAL_2_MODEL,
        "kim_vocal_2": KIM_VOCAL_2_MODEL,
    }

    def separate(
        self,
        input_audio_path: str,
        cache_dir: str,
        device: str = "cpu",
        model_name: str = None,
    ) -> tuple[str, str]:
        model_key = (model_name or "").lower().strip()
        model = self.MODEL_ALIASES.get(model_key, model_name or self.DEFAULT_MODEL)
        print(
            f"{ULTRASINGER_HEAD} Separating vocals with SOTA {blue_highlighted('BS-Roformer / Mel-Band')} "
            f"(model: {blue_highlighted(model)}, worker: {red_highlighted(device)})"
        )

        try:
            from audio_separator.separator import Separator
        except ImportError as exc:
            raise ImportError(
                "The 'audio-separator' package is required for BS-Roformer separation. "
                "Install it using: pip install 'audio-separator[gpu]'"
            ) from exc

        song_stem = os.path.splitext(os.path.basename(input_audio_path))[0]
        out_dir = os.path.join(cache_dir, "separated", "bs_roformer", song_stem)
        os.makedirs(out_dir, exist_ok=True)

        std_vocals = os.path.join(out_dir, "vocals.wav")
        std_instrumental = os.path.join(out_dir, "no_vocals.wav")

        # Reuse existing separation if already cached
        if os.path.isfile(std_vocals) and os.path.isfile(std_instrumental):
            return std_vocals, std_instrumental

        import logging
        model_dir = os.path.expanduser("~/.cache/audio-separator-models")
        os.makedirs(model_dir, exist_ok=True)

        separator = Separator(
            output_dir=out_dir,
            output_format="WAV",
            model_file_dir=model_dir,
            log_level=logging.WARNING,
        )
        if device == "cpu":
            separator.torch_device = "cpu"

        separator.load_model(model)
        output_files = separator.separate(input_audio_path)

        # Identify vocals vs instrumental from generated filenames
        v_path = None
        i_path = None
        for f in output_files:
            full_path = os.path.join(out_dir, os.path.basename(f))
            fname_lower = os.path.basename(f).lower()
            if "vocal" in fname_lower and "instrumental" not in fname_lower and "no_vocal" not in fname_lower:
                v_path = full_path
            else:
                i_path = full_path

        if v_path and os.path.isfile(v_path):
            if v_path != std_vocals:
                shutil.copy2(v_path, std_vocals)
        if i_path and os.path.isfile(i_path):
            if i_path != std_instrumental:
                shutil.copy2(i_path, std_instrumental)

        # Free VRAM immediately after separation
        del separator
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        return std_vocals, std_instrumental
