"""Mel-Band Roformer Vocal Separation Strategy (Psychoacoustic Mel-Scale)."""
import os
from modules.console_colors import (
    ULTRASINGER_HEAD,
    blue_highlighted,
    red_highlighted,
    gold_highlighted,
)
from modules.Audio.Separators.base_separator import BaseVocalSeparator
from modules.Audio.Separators.bs_roformer_separator import BSRoformerSeparator


class MelRoformerSeparator(BSRoformerSeparator):
    """Concrete separator implementing Mel-Band Roformer for lead vocal isolation."""

    DEFAULT_MEL_MODEL = "mel_band_roformer_karaoke_aufr33_viperx.yaml"
    FALLBACK_MEL_MODEL = "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt"

    def separate(
        self,
        input_audio_path: str,
        cache_dir: str,
        device: str = "cpu",
        model_name: str = None,
    ) -> tuple[str, str]:
        target_model = model_name or self.DEFAULT_MEL_MODEL
        print(
            f"{ULTRASINGER_HEAD} Separating vocals with SOTA {blue_highlighted('Mel-Band Roformer')} "
            f"(psychoacoustic mel-scale, model: {blue_highlighted(target_model)}, worker: {red_highlighted(device)})"
        )

        song_stem = os.path.splitext(os.path.basename(input_audio_path))[0]
        out_dir = os.path.join(cache_dir, "separated", "mel_roformer", song_stem)
        os.makedirs(out_dir, exist_ok=True)

        std_vocals = os.path.join(out_dir, "vocals.wav")
        std_instrumental = os.path.join(out_dir, "no_vocals.wav")

        if os.path.isfile(std_vocals) and os.path.isfile(std_instrumental):
            return std_vocals, std_instrumental

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
                f"{ULTRASINGER_HEAD} {gold_highlighted('Mel-Band model fallback:')} "
                f"Failed to load '{target_model}' ({exc}). Retrying with checkpoint '{self.FALLBACK_MEL_MODEL}'..."
            )
            try:
                return bs_sep.separate(
                    input_audio_path=input_audio_path,
                    cache_dir=cache_dir,
                    device=device,
                    model_name=self.FALLBACK_MEL_MODEL,
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

