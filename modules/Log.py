PRINT_ULTRASTAR = "\033[92m[UltraSinger]\033[0m"


def print_blue_highlighted_text(text):
    return Bcolors.BLUE + text + Bcolors.ENDC


def print_gold_highlighted_text(text):
    return Bcolors.Gold + text + Bcolors.ENDC


def print_light_blue_highlighted_text(text):
    return Bcolors.LightBlue + text + Bcolors.ENDC


def print_underlined_text(text):
    return Bcolors.Underline + text + Bcolors.ENDC


def print_red_highlighted_text(text):
    return Bcolors.RED + text + Bcolors.ENDC


def print_cyan_highlighted_text(text):
    return Bcolors.Cyan + text + Bcolors.ENDC


class Bcolors:
    BLUE = '\033[94m'
    RED = '\033[91m'
    LightBlue = '\033[96m'
    Cyan = '\033[36m'
    Gold = '\033[93m'
    Underline = '\033[4m'
    ENDC = '\033[0m'
