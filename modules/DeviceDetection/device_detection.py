"""Docstring"""

import tensorflow as tf

from modules.Log import PRINT_ULTRASTAR, print_red_highlighted_text


def get_available_device():
    """Docstring"""
    gpus = tf.config.list_physical_devices("GPU")

    if not gpus:
        print(
            f"{PRINT_ULTRASTAR} There are no GPUs available. Using {print_red_highlighted_text('cpu')}."
        )
        return "cpu"

    print(f"{PRINT_ULTRASTAR} Found available GPUs:")
    for gpu in gpus:
        print(f"Name: {gpu.name}, Type: {gpu.device_type}")

    # Todo: Finish this
    print(
        f"{PRINT_ULTRASTAR} GPU usage are currently in development. Using {print_red_highlighted_text('cpu')}"
    )
    return "cpu"
