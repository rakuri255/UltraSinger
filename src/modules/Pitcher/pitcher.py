"""Pitcher module"""
import os

import torchcrepe
import torch
import torchaudio

from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted, red_highlighted
from modules.Midi.midi_creator import convert_frequencies_to_notes, most_frequent
from modules.Pitcher.pitched_data import PitchedData
from modules.Pitcher.pitched_data_helper import get_frequencies_with_high_confidence


def get_pitch_with_torchcrepe_file(
    filename: str, model_capacity: str, step_size: int = 10, device: str = "cpu"
) -> PitchedData:
    """Pitch with torchcrepe"""

    print(
        f"{ULTRASINGER_HEAD} Pitching with {blue_highlighted('torchcrepe')} and model {blue_highlighted(model_capacity)} and {red_highlighted(device)} as worker"
    )
    # Load audio
    audio, sample_rate = torchaudio.load(filename)
    # Resample audio if necessary (torchcrepe expects 16kHz)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        audio = resampler(audio)
        sample_rate = 16000
    # Ensure audio is mono
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    # Ensure audio is on the correct device
    audio = audio.to(device)

    return get_pitch_with_torchcrepe(audio, sample_rate, model_capacity, step_size, device)


def get_pitch_with_torchcrepe(
    audio: torch.Tensor, sample_rate: int, model_capacity: str, step_size: int = 10, device: str = "cpu"
) -> PitchedData:
    """Pitch with torchcrepe"""

    # Calculate hop length
    hop_length = int(sample_rate * step_size / 1000)

    # Pitch prediction
    # Note: torchcrepe expects audio tensor on the specified device
    frequency, confidence = torchcrepe.predict(
        audio,
        sample_rate,
        hop_length,
        model=model_capacity,
        batch_size=512,  # Adjust batch size as needed
        device=device,
        return_periodicity=True  # Get confidence scores
    )

    # Generate timestamps
    times = torch.arange(len(frequency[0])) * hop_length / sample_rate
    times = times.cpu().numpy().tolist()

    # Ensure tensors are properly shaped and convert to lists of native floats for serialization
    frequencies = frequency.squeeze().cpu().numpy().tolist()
    confidence = confidence.squeeze().cpu().numpy().tolist()

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
# Todo: Unused, consider removing or updating if needed
def pitch_each_chunk_with_torchcrepe(directory: str,
                                   torchcrepe_model_capacity: str,
                                   torchcrepe_step_size: int,
                                   device: str) -> list[str]:
    """Pitch each chunk with torchcrepe and return midi notes"""
    print(f"{ULTRASINGER_HEAD} Pitching each chunk with {blue_highlighted('torchcrepe')}")

    midi_notes = []
    for filename in sorted(
            [f for f in os.listdir(directory) if f.endswith(".wav")],
            key=lambda x: int(x.split("_")[1]),
    ):
        filepath = os.path.join(directory, filename)
        # todo: stepsize = duration? then when shorter than "it" it should take the duration. Otherwise there a more notes
        pitched_data = get_pitch_with_torchcrepe_file(
            filepath,
            torchcrepe_model_capacity,
            torchcrepe_step_size,
            device,
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
