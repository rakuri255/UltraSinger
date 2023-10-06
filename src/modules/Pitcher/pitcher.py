"""Pitcher module"""

import crepe
import librosa

from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted, red_highlighted
from modules.Pitcher.core import CREPE_MODEL_SAMPLE_RATE
from modules.Pitcher.loudness import set_confidence_to_zero_in_silent_regions
from modules.Pitcher.pitched_data import PitchedData
import modules.timer as timer


def get_pitch_with_crepe_file(
    filename: str, model_capacity: str, step_size: int = 10, device: str = "cpu"
) -> PitchedData:
    """Pitch with crepe"""

    print(
        f"{ULTRASINGER_HEAD} Pitching with {blue_highlighted('crepe')} and model {blue_highlighted(model_capacity)} and {red_highlighted(device)} as worker"
    )
    timer.log('Load file for pitch detection start')
    audio, sample_rate = librosa.load(filename)
    timer.log('Load file for pitch detection end')

    return get_pitch_with_crepe(audio, sample_rate, model_capacity, step_size)


def get_pitch_with_crepe(audio, sample_rate: int, model_capacity: str, step_size: int = 10) -> PitchedData:
    """Pitch with crepe"""

    if sample_rate != CREPE_MODEL_SAMPLE_RATE:
        from resampy import resample
        audio = resample(audio, sample_rate, CREPE_MODEL_SAMPLE_RATE)
        sample_rate = CREPE_MODEL_SAMPLE_RATE

    timer.log('Crepe pitch detection start')
    times, frequencies, confidence, activation = crepe.predict(audio, sample_rate, model_capacity, step_size=step_size, viterbi=True)
    timer.log('Crepe pitch detection end')

    timer.log('Computing loudness start')
    confidence, perceived_loudness = set_confidence_to_zero_in_silent_regions(confidence, audio, step_size=step_size)
    timer.log('Computing loudness end')

    # convert to native float for serialization
    confidence = [float(x) for x in confidence]

    return PitchedData(times, frequencies, confidence, perceived_loudness)


def get_pitched_data_with_high_confidence(
    pitched_data: PitchedData, threshold=0.4
) -> PitchedData:
    """Get frequency with high confidence"""
    new_pitched_data = PitchedData([], [], [], [])
    for i, conf in enumerate(pitched_data.confidence):
        if conf > threshold:
            new_pitched_data.times.append(pitched_data.times[i])
            new_pitched_data.frequencies.append(pitched_data.frequencies[i])
            new_pitched_data.confidence.append(pitched_data.confidence[i])

    return new_pitched_data


def get_frequencies_with_high_confidence(
    frequencies: list[float], confidences: list[float], threshold=0.4
) -> list[float]:
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
