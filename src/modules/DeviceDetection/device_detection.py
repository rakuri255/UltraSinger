"""Device detection module."""

import torch
import os
import tensorflow as tf

from modules.console_colors import ULTRASINGER_HEAD, red_highlighted, blue_highlighted

tensorflow_gpu_supported = False
pytorch_gpu_supported = False

def check_gpu_support() -> tuple[bool, bool]:
    """Check worker device (e.g cuda or cpu) supported by tensorflow and pytorch"""

    print(f"{ULTRASINGER_HEAD} Checking GPU support for {blue_highlighted('tensorflow')} and {blue_highlighted('pytorch')}.")

    tensorflow_gpu_supported = False
    pytorch_gpu_supported = False

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tensorflow_gpu_supported = True
        print(f"{ULTRASINGER_HEAD} {blue_highlighted('tensorflow')} - using {red_highlighted('cuda')} gpu.")
    else:
        print(f"{ULTRASINGER_HEAD} {blue_highlighted('tensorflow')} - there are no {red_highlighted('cuda')} devices available -> Using {red_highlighted('cpu')}.")
        if os.name == 'nt':
            print(f"{ULTRASINGER_HEAD} {blue_highlighted('tensorflow')} - versions above 2.10 dropped GPU support for Windows, refer to the readme for possible solutions.")

    pytorch_gpu_supported = torch.cuda.is_available()
    if not pytorch_gpu_supported:
        print(
            f"{ULTRASINGER_HEAD} {blue_highlighted('pytorch')} - there are no {red_highlighted('cuda')} devices available -> Using {red_highlighted('cpu')}."
        )
    else:
        print(f"{ULTRASINGER_HEAD} {blue_highlighted('pytorch')} - using {red_highlighted('cuda')} gpu.")

    return 'cuda' if tensorflow_gpu_supported else 'cpu', 'cuda' if pytorch_gpu_supported else 'cpu'
