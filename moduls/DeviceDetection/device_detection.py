import tensorflow as tf

from moduls.Log import PRINT_ULTRASTAR, print_red_highlighted_text


def get_available_device():
    gpus = tf.config.list_physical_devices('GPU')

    if not gpus:
        print("{} Thera are no GPUs available. Using {}.".format(PRINT_ULTRASTAR, print_red_highlighted_text("cpu")))
        return "cpu"

    print("{} Found available GPUs:".format(PRINT_ULTRASTAR))
    for gpu in gpus:
        print("Name:", gpu.name, "  Type:", gpu.device_type)

    # Todo: Finish this
    print("{} GPU usage are currently in development. Using {}".format(PRINT_ULTRASTAR,
                                                                       print_red_highlighted_text("cpu")))
    return "cpu"
