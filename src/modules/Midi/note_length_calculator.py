def get_thirtytwo_note_second(real_bpm: float):
    """Converts a beat to a 1/32 note in second"""
    return 60 / real_bpm / 8


def get_sixteenth_note_second(real_bpm: float):
    """Converts a beat to a 1/16 note in second"""
    return 60 / real_bpm / 4


def get_eighth_note_second(real_bpm: float):
    """Converts a beat to a 1/8 note in second"""
    return 60 / real_bpm / 2


def get_quarter_note_second(real_bpm: float):
    """Converts a beat to a 1/4 note in second"""
    return 60 / real_bpm


def get_half_note_second(real_bpm: float):
    """Converts a beat to a 1/2 note in second"""
    return 60 / real_bpm * 2


def get_whole_note_second(real_bpm: float):
    """Converts a beat to a 1/1 note in second"""
    return 60 / real_bpm * 4
