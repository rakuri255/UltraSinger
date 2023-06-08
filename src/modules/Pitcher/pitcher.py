"""Pitcher module"""

import crepe
from scipy.io import wavfile

from src.modules.console_colors import (
    ULTRASINGER_HEAD,
    blue_highlighted,
    red_highlighted,
)
from src.modules.Pitcher.pitched_data import PitchedData


def get_pitch_with_crepe_file(filename: str, step_size: int, model_capacity: str) -> PitchedData:
    """Pitch with crepe"""
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


def get_pitch_with_crepe(audio, sample_rate: int, step_size: int, model_capacity: str) -> PitchedData:
    """Pitch with crepe"""
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


def get_frequency_with_high_confidence(frequencies: list[float], confidences: list[float], threshold=0.4) -> list[float]:
    """Get frequency with high confidence"""
    conf_f = []
    for i, conf in enumerate(confidences):
        if conf > threshold:
            conf_f.append(frequencies[i])
    if not conf_f:
        conf_f = frequencies
    return conf_f


class Pitcher:
    """Docstring"""
