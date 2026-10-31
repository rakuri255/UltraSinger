"""Demucs Vocal Separation Strategy."""
import os
import demucs.separate

from modules.console_colors import (
    ULTRASINGER_HEAD,
    blue_highlighted,
    red_highlighted,
)
from modules.Audio.Separators.base_separator import BaseVocalSeparator


class DemucsSeparator(BaseVocalSeparator):
    """Concrete separator implementing Meta's Demucs (htdemucs / mdx)."""

    def separate(
        self,
        input_audio_path: str,
        cache_dir: str,
        device: str = "cpu",
        model_name: str = "htdemucs",
    ) -> tuple[str, str]:
        model = model_name or "htdemucs"
        print(
            f"{ULTRASINGER_HEAD} Separating vocals with {blue_highlighted('Demucs')} "
            f"(model: {blue_highlighted(model)}, worker: {red_highlighted(device)})"
        )

        demucs_out_dir = os.path.join(cache_dir, "separated")
        demucs.separate.main(
            [
                "--two-stems",
                "vocals",
                "-d",
                str(device),
                "--float32",
                "-n",
                str(model),
                "--out",
                demucs_out_dir,
                input_audio_path,
            ]
        )

        song_stem = os.path.splitext(os.path.basename(input_audio_path))[0]
        result_dir = os.path.join(demucs_out_dir, model, song_stem)
        vocals_path = os.path.join(result_dir, "vocals.wav")
        instrumental_path = os.path.join(result_dir, "no_vocals.wav")

        return vocals_path, instrumental_path
