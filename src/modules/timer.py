import atexit
from functools import reduce
from time import process_time

from modules.console_colors import ULTRASINGER_HEAD


def seconds_to_str(t):
    """Format seconds to string"""
    return "%d:%02d:%02d.%03d" % reduce(
        lambda ll, b: divmod(ll[0], b) + ll[1:], [(t * 1000,), 1000, 60, 60]
    )


def log(s):
    """Log line with optional time elapsed"""
    print(f"{ULTRASINGER_HEAD} {seconds_to_str(process_time())} - {s}")


def end_log():
    """Log at program end"""
    log("End Program")


atexit.register(end_log)
log("Initialized...")
