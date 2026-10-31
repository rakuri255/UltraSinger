"""SCNet (Sparse Compression Network) Vocal Separation Strategy."""
import os
from modules.console_colors import (
    ULTRASINGER_HEAD,
    blue_highlighted,
    red_highlighted,
    gold_highlighted,
)
from modules.Audio.Separators.base_separator import BaseVocalSeparator
from modules.Audio.Separators.bs_roformer_separator import BSRoformerSeparator


class SCNetSeparator(BaseVocalSeparator):
    """Concrete separator implementing SCNet (low VRAM and high inference speed)."""

    DEFAULT_SCNET_MODEL = "SCNet-Large"
    YAML_SCNET_MODEL = "scnet_large.yaml"

    def separate(
        self,
        input_audio_path: str,
        cache_dir: str,
        device: str = "cpu",
        model_name: str = None,
    ) -> tuple[str, str]:
        target_model = model_name or self.DEFAULT_SCNET_MODEL
        print(
            f"{ULTRASINGER_HEAD} Separating vocals with {blue_highlighted('SCNet-Large')} "
            f"(low VRAM, fast inference, model: {blue_highlighted(target_model)}, worker: {red_highlighted(device)})"
        )

        bs_sep = BSRoformerSeparator()
        try:
            return bs_sep.separate(
                input_audio_path=input_audio_path,
                cache_dir=cache_dir,
                device=device,
                model_name=target_model,
            )
        except Exception as exc:
            print(
                f"{ULTRASINGER_HEAD} {gold_highlighted('SCNet model fallback:')} "
                f"Could not initialize SCNet '{target_model}' ({exc}). Retrying with '{self.YAML_SCNET_MODEL}'..."
            )
            try:
                return bs_sep.separate(
                    input_audio_path=input_audio_path,
                    cache_dir=cache_dir,
                    device=device,
                    model_name=self.YAML_SCNET_MODEL,
                )
            except Exception as inner_exc:
                print(
                    f"{ULTRASINGER_HEAD} {gold_highlighted('Falling back to default BS-Roformer:')} ({inner_exc})"
                )
                return bs_sep.separate(
                    input_audio_path=input_audio_path,
                    cache_dir=cache_dir,
                    device=device,
                    model_name=BSRoformerSeparator.DEFAULT_MODEL,
                )

