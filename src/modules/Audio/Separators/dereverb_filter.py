"""Surgical vocal de-reverberation and de-echoing filter module."""
import os
import shutil
import gc
from modules.console_colors import (
    ULTRASINGER_HEAD,
    blue_highlighted,
    red_highlighted,
    gold_highlighted,
)


def apply_dereverb_filter(
    input_vocal_path: str,
    output_vocal_path: str,
    dereverb_engine: str = "uvr_dereverb",
    device: str = "cpu",
) -> str:
    """Applies surgical post-processing de-echo / de-reverb to extracted vocal track.

    Args:
        input_vocal_path: Path to the isolated vocal WAV.
        output_vocal_path: Desired output path for the de-reverbed vocal WAV.
        dereverb_engine: Model name ('uvr_dereverb' for UVR-DeEcho-DeReverb.onnx or 'kim_vocal_2' for Kim_Vocal_2.onnx).
        device: 'cuda' or 'cpu'.

    Returns:
        Path to the resulting vocal file (either processed or original fallback).
    """
    engine_key = (dereverb_engine or "").lower().strip()
    if engine_key in ("none", "", "false", "0"):
        if input_vocal_path != output_vocal_path and os.path.isfile(input_vocal_path):
            shutil.copy2(input_vocal_path, output_vocal_path)
        return output_vocal_path

    model_map = {
        "uvr_dereverb": "UVR-DeEcho-DeReverb.onnx",
        "deecho": "UVR-DeEcho-DeReverb.onnx",
        "dereverb": "UVR-DeEcho-DeReverb.onnx",
        "kim_vocal_2": "Kim_Vocal_2.onnx",
        "kim": "Kim_Vocal_2.onnx",
        "kim_vocal": "Kim_Vocal_2.onnx",
    }
    model_name = model_map.get(engine_key, "UVR-DeEcho-DeReverb.onnx")

    print(
        f"{ULTRASINGER_HEAD} Applying surgical De-Reverb filter with "
        f"{blue_highlighted(model_name)} (worker: {red_highlighted(device)})"
    )

    try:
        from audio_separator.separator import Separator
        import logging

        work_dir = os.path.dirname(output_vocal_path) or "/tmp"
        os.makedirs(work_dir, exist_ok=True)
        model_dir = os.path.expanduser("~/.cache/audio-separator-models")
        os.makedirs(model_dir, exist_ok=True)

        separator = Separator(
            output_dir=work_dir,
            output_format="WAV",
            model_file_dir=model_dir,
            log_level=logging.WARNING,
        )
        if device == "cpu":
            separator.torch_device = "cpu"

        separator.load_model(model_name)
        separated_files = separator.separate(input_vocal_path)

        # Look for de-reverbed output (often tagged 'dry', 'no_echo', or the primary vocal stem)
        chosen_file = None
        for f in separated_files:
            full_f = os.path.join(work_dir, os.path.basename(f))
            fname = os.path.basename(f).lower()
            if "dry" in fname or "deecho" in fname or "dereverb" in fname or "vocal" in fname:
                chosen_file = full_f
                break

        if not chosen_file and separated_files:
            chosen_file = os.path.join(work_dir, os.path.basename(separated_files[0]))

        if chosen_file and os.path.isfile(chosen_file):
            if chosen_file != output_vocal_path:
                shutil.copy2(chosen_file, output_vocal_path)
            print(f"{ULTRASINGER_HEAD} {gold_highlighted('De-Reverb complete:')} Dry vocal saved.")
        else:
            if input_vocal_path != output_vocal_path:
                shutil.copy2(input_vocal_path, output_vocal_path)

        del separator
    except Exception as exc:
        print(
            f"{ULTRASINGER_HEAD} {gold_highlighted('De-Reverb warning:')} "
            f"Could not complete de-reverberation ({exc}). Retaining pristine separated vocals."
        )
        if input_vocal_path != output_vocal_path and os.path.isfile(input_vocal_path):
            shutil.copy2(input_vocal_path, output_vocal_path)
    finally:
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    return output_vocal_path

