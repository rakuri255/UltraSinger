"""Hybrid Transformer Demucs (HTDemucs) Multi-stem Separation Strategy."""
import os
from modules.console_colors import (
    ULTRASINGER_HEAD,
    blue_highlighted,
    red_highlighted,
    gold_highlighted,
)
from modules.Audio.Separators.base_separator import BaseVocalSeparator
from modules.Audio.Separators.demucs_separator import DemucsSeparator


class HTDemucsSeparator(BaseVocalSeparator):
    """Concrete separator implementing Meta's HTDemucs fine-tuned / 6-stem models."""

    DEFAULT_MODEL = "htdemucs_ft"
    SIX_STEM_MODEL = "htdemucs_6s"

    def separate(
        self,
        input_audio_path: str,
        cache_dir: str,
        device: str = "cpu",
        model_name: str = None,
    ) -> tuple[str, str]:
        model = (model_name or self.DEFAULT_MODEL).strip()
        print(
            f"{ULTRASINGER_HEAD} Separating vocals with SOTA {blue_highlighted('HTDemucs')} "
            f"(multi-stem, model: {blue_highlighted(model)}, worker: {red_highlighted(device)})"
        )

        demucs_sep = DemucsSeparator()
        try:
            return demucs_sep.separate(
                input_audio_path=input_audio_path,
                cache_dir=cache_dir,
                device=device,
                model_name=model,
            )
        except Exception as exc:
            print(
                f"{ULTRASINGER_HEAD} {gold_highlighted('HTDemucs fallback:')} "
                f"Model '{model}' failed ({exc}). Retrying with base 'htdemucs'..."
            )
            return demucs_sep.separate(
                input_audio_path=input_audio_path,
                cache_dir=cache_dir,
                device=device,
                model_name="htdemucs",
            )

