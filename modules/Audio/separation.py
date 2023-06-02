"""Docstring"""

import subprocess

from modules.Log import (
    PRINT_ULTRASTAR,
    print_blue_highlighted_text,
    print_red_highlighted_text,
)
from modules.os_helper import current_executor_path, move, path_join


def separate_audio(input_file_path, output_file, device="cpu"):
    """Docstring"""

    print(
        f"{PRINT_ULTRASTAR} Separating vocals from audio with {print_blue_highlighted_text('demucs')} and {print_red_highlighted_text(device)} as worker."
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
