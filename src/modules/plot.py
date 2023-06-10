import os

import librosa
from matplotlib import pyplot as plt

from src.modules.Pitcher.pitched_data import PitchedData
from src.modules.Speech_Recognition.TranscribedData import TranscribedData


def plot(pitched_data: PitchedData, transcribed_data: list[TranscribedData], midi_notes: list[str], output_path: str) -> None:
    """Plot transcribed data"""

    conf_t, conf_f, conf_c = get_confidence(pitched_data, 0.4)

    plt.ylim(0, 600)
    plt.xlim(0, 50)
    plt.plot(conf_t, conf_f, linewidth=0.1)
    plt.savefig(os.path.join(output_path, "crepe_0.4.png"))

    for i, data in enumerate(transcribed_data):
        note_frequency = librosa.note_to_hz(midi_notes[i])
        plt.plot(
            [data.start, data.end],
            [note_frequency, note_frequency],
            linewidth=1,
            alpha=0.5,
        )
    plt.savefig(os.path.join(output_path, "pit.png"), dpi=2000)


def get_confidence(pitched_data: PitchedData, threshold: float) -> tuple[list[float], list[float], list[float]]:
    """Get high confidence data"""
    # todo: replace get_frequency_with_high_conf from pitcher
    conf_t = []
    conf_f = []
    conf_c = []
    for i in enumerate(pitched_data.times):
        pos = i[0]
        if pitched_data.confidence[pos] > threshold:
            conf_t.append(pitched_data.times[pos])
            conf_f.append(pitched_data.frequencies[pos])
            conf_c.append(pitched_data.confidence[pos])
    return conf_t, conf_f, conf_c
