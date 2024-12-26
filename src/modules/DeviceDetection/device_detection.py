"""Device detection module."""

import torch
import os
import tensorflow as tf

from modules.console_colors import ULTRASINGER_HEAD, red_highlighted, blue_highlighted

tensorflow_gpu_supported = False
pytorch_gpu_supported = False

def check_gpu_support() -> tuple[bool, bool]:
    """Check worker device (e.g cuda or cpu) supported by tensorflow and pytorch"""

    print(f"{ULTRASINGER_HEAD} Checking GPU support.")

    pytorch_gpu_supported = __check_pytorch_support()
    tensorflow_gpu_supported = __check_tensorflow_support()

    return 'cuda' if tensorflow_gpu_supported else 'cpu', 'cuda' if pytorch_gpu_supported else 'cpu'


def __check_tensorflow_support():
    tensorflow_gpu_supported = False
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tensorflow_gpu_supported = True
        print(f"{ULTRASINGER_HEAD} {blue_highlighted('tensorflow')} - using {red_highlighted('cuda')} gpu.")
    else:
        print(
            f"{ULTRASINGER_HEAD} {blue_highlighted('tensorflow')} - there are no {red_highlighted('cuda')} devices available -> Using {red_highlighted('cpu')}.")
        if os.name == 'nt':
            print(
                f"{ULTRASINGER_HEAD} {blue_highlighted('tensorflow')} - versions above 2.10 dropped GPU support for Windows, refer to the readme for possible solutions.")
    return tensorflow_gpu_supported


def __check_pytorch_support():
    pytorch_gpu_supported = torch.cuda.is_available()
    if not pytorch_gpu_supported:
        print(
            f"{ULTRASINGER_HEAD} {blue_highlighted('pytorch')} - there are no {red_highlighted('cuda')} devices available -> Using {red_highlighted('cpu')}."
        )
    else:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_properties = torch.cuda.get_device_properties(0)
        gpu_vram = round(gpu_properties.total_memory / 1024 ** 3, 2)  # Convert bytes to GB and round to 2 decimal places
        print(f"{ULTRASINGER_HEAD} Found GPU: {blue_highlighted(gpu_name)} VRAM: {blue_highlighted(gpu_vram)} GB.")
        if gpu_vram < 6:
            print(
                f"{ULTRASINGER_HEAD} {red_highlighted('GPU VRAM is less than 6GB. Program may crash due to insufficient memory.')}")
        print(f"{ULTRASINGER_HEAD} {blue_highlighted('pytorch')} - using {red_highlighted('cuda')} gpu.")
    return pytorch_gpu_supported
