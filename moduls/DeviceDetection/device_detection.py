from moduls.Log import PRINT_ULTRASTAR, print_red_highlighted_text
import torch

def get_available_device():
    isCuda = torch.cuda.is_available()

    if not isCuda:
        print("{} There are no {} devices available. -> Using {}.".format(PRINT_ULTRASTAR, print_red_highlighted_text("cuda"), print_red_highlighted_text("cpu")))
        return "cpu"
    else:
        print("{} Using {} GPU.".format(PRINT_ULTRASTAR, print_red_highlighted_text("cuda")))

    return "cuda"
