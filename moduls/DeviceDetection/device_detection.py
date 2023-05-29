from moduls.Log import PRINT_ULTRASTAR, print_red_highlighted_text
import torch

def get_available_device():
    isCuda = torch.cuda.is_available()

    if not isCuda:
        print(f"{PRINT_ULTRASTAR} There are no {print_red_highlighted_text('cuda')} devices available. -> Using {print_red_highlighted_text('cpu')}.")
        return "cpu"
    else:
        print(f"{PRINT_ULTRASTAR} Using {print_red_highlighted_text('cuda')} GPU.")

    return "cuda"
