"""Docstring"""

import subprocess

from modules.console_colors import (
    ULTRASINGER_HEAD,
    blue_highlighted,
    red_highlighted,
)
from modules.os_helper import current_executor_path, move, path_join


def separate_audio(input_file_path, output_file, device="cpu"):
    """Docstring"""

    print(
        f"{ULTRASINGER_HEAD} Separating vocals from audio with {blue_highlighted('demucs')} and {red_highlighted(device)} as worker."
    )
    # Model selection?
    # -n mdx_q
    # -n htdemucs_ft
    # todo: -d cpu otherwise it automatically uses gpu
    subprocess.run(
        ["demucs", "--two-stems=vocals", input_file_path], check=True
    )
    separated_folder = path_join(current_executor_path(), "separated")
    move(separated_folder, output_file)
