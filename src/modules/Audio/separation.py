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

    # Model selection?
    # -n htdemucs_ft
    # subprocess.run(
    #     ["demucs", "-d", device, "--two-stems=vocals", "--float32", input_file_path]
    # )
    # separated_folder = path_join(current_executor_path(), "separated")
    # move(separated_folder, output_file)

    # fixme "--float32" is missing
    demucs.separate.main(shlex.split(f'--two-stems vocals -d {device} --out "{os.path.join(output_folder, "separated")}" "{input_file_path}"'))
