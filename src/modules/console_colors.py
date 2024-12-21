"""Colors for the console"""

ULTRASINGER_HEAD = "\033[92m[UltraSinger]\033[0m"


def blue_highlighted(text: str) -> str:
    """Returns a blue highlighted text"""
    return f"{Bcolors.blue}{text}{Bcolors.endc}"


def green_highlighted(text: str) -> str:
    """Returns a blue highlighted text"""
    return f"{Bcolors.dark_green}{text}{Bcolors.endc}"


def gold_highlighted(text: str) -> str:
    """Returns a gold highlighted text"""
    return f"{Bcolors.gold}{text}{Bcolors.endc}"


def light_blue_highlighted(text: str) -> str:
    """Returns a light blue highlighted text"""
    return f"{Bcolors.light_blue}{text}{Bcolors.endc}"


def underlined(text: str) -> str:
    """Returns an underlined text"""
    return f"{Bcolors.underline}{text}{Bcolors.endc}"


def red_highlighted(text: str) -> str:
    """Returns a red highlighted text"""
    return f"{Bcolors.red}{text}{Bcolors.endc}"

def cyan_highlighted(text: str) -> str:
    """Returns a cyan highlighted text"""
    return f"{Bcolors.cyan}{text}{Bcolors.endc}"

def bright_green_highlighted(text: str) -> str:
    """Returns a cyan highlighted text"""
    return f"{Bcolors.bright_green}{text}{Bcolors.endc}"

class Bcolors:
    """Colors for the console"""

    blue = "\033[94m"
    dark_green = "\033[32m"
    red = "\033[91m"
    light_blue = "\033[96m"
    cyan = "\033[36m"
    gold = "\033[93m"
    underline = "\033[4m"
    endc = "\033[0m"
    bright_green = "\033[38;2;204;255;204m"
