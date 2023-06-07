"""Docstring"""

import csv

import crepe
from scipy.io import wavfile

from modules.console_colors import (
    ULTRASINGER_HEAD,
    blue_highlighted,
    red_highlighted,
)
from modules.Pitcher.pitched_data import PitchedData


def get_pitch_with_crepe_file(filename, step_size, model_capacity):
    """Docstring"""
    print(
        f"{ULTRASINGER_HEAD} Pitching with {blue_highlighted('crepe')} and model {blue_highlighted(model_capacity)} and {red_highlighted('cpu')} as worker"
    )
    # Todo: add GPU support by using torchcrepe
    sample_rate, audio = wavfile.read(filename)

    pitched_data = PitchedData()
    (
        pitched_data.times,
        pitched_data.frequencies,
        pitched_data.confidence,
        activation,
    ) = crepe.predict(
        audio, sample_rate, model_capacity, step_size=step_size, viterbi=True
    )
    return pitched_data


def get_pitch_with_crepe(audio, sample_rate, step_size, model_capacity):
    """Docstring"""
    pitched_data = PitchedData()
    (
        pitched_data.times,
        pitched_data.frequencies,
        pitched_data.confidence,
        activation,
    ) = crepe.predict(
        audio, sample_rate, model_capacity, step_size=step_size, viterbi=True
    )
    return pitched_data


def write_lists_to_csv(times, frequencies, confidences, filename):
    """Docstring"""
    with open(filename, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        header = ["time", "frequency", "confidence"]
        writer.writerow(header)
        for i in enumerate(times):
            pos = i[0]
            writer.writerow([times[pos], frequencies[pos], confidences[pos]])


def read_data_from_csv(filename):
    """Docstring"""
    csv_data = []
    with open(filename, "r", encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file)
        for line in csv_reader:
            csv_data.append(line)
    headless_data = csv_data[1:]
    return headless_data


def get_frequency_with_high_confidence(freqs, confs, threshold=0.4):
    """Docstring"""
    conf_f = []
    for i, conf in enumerate(confs):
        if conf > threshold:
            conf_f.append(freqs[i])
    if not conf_f:
        conf_f = freqs
    return conf_f


class Pitcher:
    """Docstring"""
