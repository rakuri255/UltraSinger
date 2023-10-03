"""Separate vocals from audio"""
import os
import shlex
import subprocess

import demucs.separate

from modules.console_colors import (
    ULTRASINGER_HEAD,
    blue_highlighted,
    red_highlighted,
)
from modules.os_helper import current_executor_path, move, path_join


def separate_audio(input_file_path: str, output_folder: str, device="cpu") -> None:
    """Separate vocals from audio with demucs."""

    print(
        f"{ULTRASINGER_HEAD} Separating vocals from audio with {blue_highlighted('demucs')} and {red_highlighted(device)} as worker."
    )

    demucs.separate.main(shlex.split(f'--two-stems vocals -d {device} --out "{os.path.join(output_folder, "separated")}" "{input_file_path}"'))
    # Model selection?
    # -n mdx_q
    # -n htdemucs_ft
    # subprocess.run(
    #     ["demucs", "-d", device, "--two-stems=vocals", input_file_path.replace("\\", "/")]
    # )