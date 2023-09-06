"""Pitcher module"""

import torchcrepe
from torch.cuda import OutOfMemoryError

from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted, red_highlighted
from modules.Pitcher.pitched_data import PitchedData

DEFAULT_MODEL: str = "full"
SUPPORTED_MODELS: list[str] = ["tiny", "full"]
TIMES_DECIMAL_PLACES: int = 3


def get_pitch_with_crepe_file(
    filename: str,
    model: str,
    device: str = "cpu",
    batch_size: int = None,
    step_size: int = 10,
) -> PitchedData:
    """Pitch with crepe"""

    # Load audio
    audio, sample_rate = torchcrepe.load.audio(filename)

    return get_pitch_with_crepe(
        audio, sample_rate, model, device, batch_size, step_size
    )


def get_pitch_with_crepe(
    audio,
    sample_rate: int,
    model: str,
    device: str = "cpu",
    batch_size: int = None,
    step_size: int = 10,
) -> PitchedData:
    """Pitch with crepe"""

    if model not in SUPPORTED_MODELS:
        print(
            f"{ULTRASINGER_HEAD} {blue_highlighted('crepe')} model {blue_highlighted(model)} is not supported, defaulting to {blue_highlighted(DEFAULT_MODEL)}"
        )
        model = DEFAULT_MODEL

    print(
        f"{ULTRASINGER_HEAD} Pitching using {blue_highlighted('crepe')} with model {blue_highlighted(model)} and {red_highlighted(device)} as worker"
    )

    hop_length = None
    if step_size:
        step_size_seconds = round(step_size / 1000, TIMES_DECIMAL_PLACES)
        steps_per_second = 1 / step_size_seconds
        hop_length = sample_rate // steps_per_second
    else:
        step_size_seconds = round(step_size / 1000, TIMES_DECIMAL_PLACES)
        steps_per_second = 1 / step_size_seconds
        hop_length = sample_rate // steps_per_second

    # Provide a sensible frequency range for your domain (upper limit is 2006 Hz)
    # TODO: determine appropriate range for vocals
    fmin = 50
    fmax = 2006

    try:
        frequencies, confidence = torchcrepe.predict(
            audio,
            sample_rate,
            hop_length,
            fmin,
            fmax,
            model,
            return_periodicity=True,
            batch_size=batch_size,
            device=device,
        )
        frequencies = frequencies.detach().cpu().numpy().squeeze(0)
        confidence = confidence.detach().cpu().numpy().squeeze(0)
    except OutOfMemoryError as oom_exception:
        print(
            f"{ULTRASINGER_HEAD} {blue_highlighted('crepe')} ran out of GPU memory; reduce --crepe_batch_size or force cpu with --force_crepe_cpu"
        )
        raise oom_exception

    times = [
        round(i * step_size_seconds, TIMES_DECIMAL_PLACES)
        for i, x in enumerate(confidence)
    ]

    return PitchedData(times, frequencies, confidence)


def get_frequency_with_high_confidence(
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
