"""Docstring"""


def real_bpm_to_ultrastar_bpm(real_bpm):
    """Docstring"""
    # The UltraStar BPM info is a fifth beat of the real BPM
    ultrastar_bpm = real_bpm / 4
    return ultrastar_bpm


def ultrastar_bpm_to_real_bpm(ultrastar_bpm):
    """Docstring"""
    # The UltraStar BPM info is a fifth beat of the real BPM
    bpm = ultrastar_bpm * 4
    return bpm


def second_to_beat(seconds, real_bpm):
    """Docstring"""
    # BPM = 60 * beat / T
    # T * BPM = 60 * beat
    # beat = T * BPM / 60
    beat = seconds * real_bpm / 60
    return beat


def beat_to_second(beat, real_bpm):
    """Docstring"""

    seconds = beat * 60 / real_bpm
    return seconds


def midi_note_to_ultrastar_note(midi_note):
    """Docstring"""

    # C4 == 48 in Midi
    ultrastar_note = midi_note - 48
    return ultrastar_note


def ultrastar_note_to_midi_note(ultrastar_note):
    """Docstring"""

    # C4 == 48 in Midi
    midi_note = ultrastar_note + 48
    return midi_note


def get_start_time_from_ultrastar(ultrastar_class, pos):
    """Docstring"""

    gap = int(ultrastar_class.gap) / 1000
    real_bpm = ultrastar_bpm_to_real_bpm(
        float(ultrastar_class.bpm.replace(",", "."))
    )
    start_time = (
        beat_to_second(int(ultrastar_class.startBeat[pos]), real_bpm) + gap
    )
    return start_time


def get_end_time_from_ultrastar(ultrastar_class, pos):
    """Docstring"""

    gap = int(ultrastar_class.gap) / 1000
    real_bpm = ultrastar_bpm_to_real_bpm(
        float(ultrastar_class.bpm.replace(",", "."))
    )
    end_time = (
        beat_to_second(
            int(ultrastar_class.startBeat[pos])
            + int(ultrastar_class.durations[pos]),
            real_bpm,
        )
        + gap
    )
    return end_time
