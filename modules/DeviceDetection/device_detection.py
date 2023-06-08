"""Device detection module."""

import torch

from modules.console_colors import ULTRASINGER_HEAD, red_highlighted


def get_available_device() -> str:
    """Get available worker device e.g cuda or cpu"""
    is_cuda = torch.cuda.is_available()

    if not is_cuda:
        print(
            f"{ULTRASINGER_HEAD} There are no {red_highlighted('cuda')} devices available. -> Using {red_highlighted('cpu')}."
        )
        return "cpu"
    print(f"{ULTRASINGER_HEAD} Using {red_highlighted('cuda')} GPU.")

    return "cuda"
