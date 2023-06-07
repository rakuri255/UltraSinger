"""Docstring"""

import tensorflow as tf

from modules.console_colors import ULTRASINGER_HEAD, red_highlighted


def get_available_device():
    """Docstring"""
    gpus = tf.config.list_physical_devices("GPU")

    if not gpus:
        print(
            f"{ULTRASINGER_HEAD} There are no GPUs available. Using {red_highlighted('cpu')}."
        )
        return "cpu"

    print(f"{ULTRASINGER_HEAD} Found available GPUs:")
    for gpu in gpus:
        print(f"Name: {gpu.name}, Type: {gpu.device_type}")

    # Todo: Finish this
    print(
        f"{ULTRASINGER_HEAD} GPU usage are currently in development. Using {red_highlighted('cpu')}"
    )
    return "cpu"
