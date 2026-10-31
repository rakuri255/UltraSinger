"""Separate vocals from audio"""
import os
from enum import Enum

import demucs.separate

from modules.console_colors import (
    ULTRASINGER_HEAD,
    blue_highlighted,
    red_highlighted, green_highlighted,
)
from modules.os_helper import check_file_exists

class DemucsModel(Enum):
    HTDEMUCS = "htdemucs"           # first version of Hybrid Transformer Demucs. Trained on MusDB + 800 songs. Default model.
    HTDEMUCS_FT = "htdemucs_ft"     # fine-tuned version of htdemucs, separation will take 4 times more time but might be a bit better. Same training set as htdemucs.
    HTDEMUCS_6S = "htdemucs_6s"     # 6 sources version of htdemucs, with piano and guitar being added as sources. Note that the piano source is not working great at the moment.
    HDEMUCS_MMI = "hdemucs_mmi"     # Hybrid Demucs v3, retrained on MusDB + 800 songs.
    MDX = "mdx"                     # trained only on MusDB HQ, winning model on track A at the MDX challenge.
    MDX_EXTRA = "mdx_extra"         # trained with extra training data (including MusDB test set), ranked 2nd on the track B of the MDX challenge.
    MDX_Q = "mdx_q"                 # quantized version of the previous models. Smaller download and storage but quality can be slightly worse.
    MDX_EXTRA_Q = "mdx_extra_q"     # quantized version of mdx_extra. Smaller download and storage but quality can be slightly worse.
    SIG = "SIG"                     # Placeholder for a single model from the model zoo.

def separate_audio(input_file_path: str, output_folder: str, model: DemucsModel, device="cpu") -> None:
    """Separate vocals from audio with demucs."""

    print(
        f"{ULTRASINGER_HEAD} Separating vocals from audio with {blue_highlighted('demucs')} with model {blue_highlighted(model.value)} and {red_highlighted(device)} as worker."
    )

    demucs.separate.main(
        [
            "--two-stems", "vocals",
            "-d", f"{device}",
            "--float32",
            "-n",
            model.value,
            "--out", f"{os.path.join(output_folder, 'separated')}",
            f"{input_file_path}",
        ]
    )

def separate_vocal_from_audio(cache_folder_path: str,
                              audio_output_file_path: str,
                              use_separated_vocal: bool,
                              create_karaoke: bool,
                              pytorch_device: str,
                              model: DemucsModel,
                              skip_cache: bool = False,
                              separator_name: str = "bs_roformer",
                              bs_roformer_model: str = None,
                              dereverb_engine: str = "none") -> str:
    """Separate vocal from audio using selected separator strategy (BS-Roformer, Mel-Band Roformer, HTDemucs, SCNet, Demucs) with optional de-reverb post-processing."""
    from modules.Audio.Separators import SeparatorFactory, apply_dereverb_filter

    strategy_name = (separator_name or "bs_roformer").lower()
    model_str = bs_roformer_model if strategy_name in ["bs-roformer", "bs_roformer", "roformer", "mel_roformer", "mel-roformer", "mel_band_roformer"] else (model.value if hasattr(model, "value") else str(model))

    song_stem = os.path.splitext(os.path.basename(audio_output_file_path))[0]
    subfolder = strategy_name.replace("-", "_")
    audio_separation_path = os.path.join(cache_folder_path, "separated", subfolder, song_stem)

    vocals_path = os.path.join(audio_separation_path, "vocals.wav")
    instrumental_path = os.path.join(audio_separation_path, "no_vocals.wav")

    if use_separated_vocal or create_karaoke:
        cache_available = check_file_exists(vocals_path) and check_file_exists(instrumental_path)
        if skip_cache or not cache_available:
            separator = SeparatorFactory.get_separator(strategy_name)
            v_path, i_path = separator.separate(
                input_audio_path=audio_output_file_path,
                cache_dir=cache_folder_path,
                device=pytorch_device,
                model_name=model_str,
            )

            # Ensure vocals.wav and no_vocals.wav are in audio_separation_path
            if v_path != vocals_path and os.path.isfile(v_path):
                import shutil
                os.makedirs(audio_separation_path, exist_ok=True)
                shutil.copy2(v_path, vocals_path)
            if i_path != instrumental_path and os.path.isfile(i_path):
                import shutil
                os.makedirs(audio_separation_path, exist_ok=True)
                shutil.copy2(i_path, instrumental_path)

            # Apply surgical De-Reverb / De-Echo filter if requested
            if dereverb_engine and dereverb_engine.lower() not in ("none", "false", "0", ""):
                apply_dereverb_filter(
                    input_vocal_path=vocals_path,
                    output_vocal_path=vocals_path,
                    dereverb_engine=dereverb_engine,
                    device=pytorch_device,
                )
        else:
            print(f"{ULTRASINGER_HEAD} {green_highlighted('cache')} reusing cached separated vocals")

    return audio_separation_path
