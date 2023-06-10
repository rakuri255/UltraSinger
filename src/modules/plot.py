"""Plot transcribed data"""

import os
import librosa
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from src.modules.Pitcher.pitched_data import PitchedData
from src.modules.Speech_Recognition.TranscribedData import TranscribedData
from src.modules.console_colors import ULTRASINGER_HEAD


def get_frequency_range(midi_note: str) -> float:
    midi = librosa.note_to_midi(midi_note)
    frequency_range = librosa.midi_to_hz(midi + 1) - librosa.midi_to_hz(midi)
    return frequency_range


def plot(pitched_data: PitchedData,
         transcribed_data: list[TranscribedData],
         midi_notes: list[str],
         output_path: str
         ) -> None:
    """Plot transcribed data"""

    print(f"{ULTRASINGER_HEAD} Creating plot")

    conf_t, conf_f, conf_c = get_confidence(pitched_data, 0.4)

    plt.ylim(0, 600)
    plt.xlim(0, 50)
    plt.plot(conf_t, conf_f, linewidth=0.1)

    for i, data in enumerate(transcribed_data):
        note_frequency = librosa.note_to_hz(midi_notes[i])
        frequency_range = get_frequency_range(midi_notes[i])
        xy_start_pos = (data.start, note_frequency - frequency_range / 2)
        width = data.end - data.start
        height = frequency_range
        rect = Rectangle(
            xy_start_pos,
            width,
            height,
            edgecolor='none',
            facecolor='red',
            alpha=0.5)
        plt.gca().add_patch(rect)
    plt.savefig(os.path.join(output_path, "plot.png"), dpi=3000)


def get_confidence(
        pitched_data: PitchedData,
        threshold: float
) -> tuple[list[float], list[float], list[float]]:
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
