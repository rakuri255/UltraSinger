"""Pitcher module"""
import os

import crepe
from scipy.io import wavfile

from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted, red_highlighted
from modules.Midi.midi_creator import convert_frequencies_to_notes, most_frequent
from modules.Pitcher.pitched_data import PitchedData
from modules.Pitcher.pitched_data_helper import get_frequencies_with_high_confidence


def get_pitch_with_crepe_file(
    filename: str, model_capacity: str, step_size: int = 10, device: str = "cpu"
) -> PitchedData:
    """Pitch with crepe"""

    print(
        f"{ULTRASINGER_HEAD} Pitching with {blue_highlighted('crepe')} and model {blue_highlighted(model_capacity)} and {red_highlighted(device)} as worker"
    )
    sample_rate, audio = wavfile.read(filename)

    return get_pitch_with_crepe(audio, sample_rate, model_capacity, step_size)


def get_pitch_with_crepe(
    audio, sample_rate: int, model_capacity: str, step_size: int = 10
) -> PitchedData:
    """Pitch with crepe"""

    # Info: The model is trained on 16 kHz audio, so if the input audio has a different sample rate, it will be first resampled to 16 kHz using resampy inside crepe.

    times, frequencies, confidence, activation = crepe.predict(
        audio, sample_rate, model_capacity, step_size=step_size, viterbi=True
    )

    # convert to native float for serialization
    confidence = [float(x) for x in confidence]

    return PitchedData(times, frequencies, confidence)


def get_pitched_data_with_high_confidence(
    pitched_data: PitchedData, threshold=0.4
) -> PitchedData:
    """Get frequency with high confidence"""
    new_pitched_data = PitchedData([], [], [])
    for i, conf in enumerate(pitched_data.confidence):
        if conf > threshold:
            new_pitched_data.times.append(pitched_data.times[i])
            new_pitched_data.frequencies.append(pitched_data.frequencies[i])
            new_pitched_data.confidence.append(pitched_data.confidence[i])

    return new_pitched_data

# Todo: Unused
def pitch_each_chunk_with_crepe(directory: str,
                                crepe_model_capacity: str,
                                crepe_step_size: int,
                                tensorflow_device: str) -> list[str]:
    """Pitch each chunk with crepe and return midi notes"""
    print(f"{ULTRASINGER_HEAD} Pitching each chunk with {blue_highlighted('crepe')}")

    midi_notes = []
    for filename in sorted(
            [f for f in os.listdir(directory) if f.endswith(".wav")],
            key=lambda x: int(x.split("_")[1]),
    ):
        filepath = os.path.join(directory, filename)
        # todo: stepsize = duration? then when shorter than "it" it should take the duration. Otherwise there a more notes
        pitched_data = get_pitch_with_crepe_file(
            filepath,
            crepe_model_capacity,
            crepe_step_size,
            tensorflow_device,
        )
        conf_f = get_frequencies_with_high_confidence(
            pitched_data.frequencies, pitched_data.confidence
        )

        notes = convert_frequencies_to_notes(conf_f)
        note = most_frequent(notes)[0][0]

        midi_notes.append(note)
        # todo: Progress?
        # print(filename + " f: " + str(mean))

    return midi_notes

class Pitcher:
    """Docstring"""
