"""Docstring"""

import torch

from modules.console_colors import ULTRASINGER_HEAD, red_highlighted


def get_available_device():
    """Docstring"""
    isCuda = torch.cuda.is_available()

    if not isCuda:
        print(
            f"{ULTRASINGER_HEAD} There are no {red_highlighted('cuda')} devices available. -> Using {red_highlighted('cpu')}."
        )

        return "cpu"
    else:
        print(f"{ULTRASINGER_HEAD} Using {red_highlighted('cuda')} GPU.")

    return "cuda"