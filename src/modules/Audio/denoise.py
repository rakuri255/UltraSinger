"""Reduce noise from audio"""

import ffmpeg

from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted, green_highlighted
from modules.os_helper import check_file_exists


def __ffmpeg_reduce_noise(input_file_path: str, output_file: str,
                          noise_reduction: float = 20,
                          noise_floor: float = -80,
                          track_noise: bool = True) -> None:
    """Reduce noise from vocal audio with ffmpeg.

    Uses the afftdn (FFT-based denoising) filter.

    Args:
        input_file_path: Path to input audio file.
        output_file: Path for denoised output file.
        noise_reduction: Noise reduction in dB (0.01-97). Default: 20.
            Lower values preserve more vocal detail (consonants, sibilants).
            Higher values remove more noise but risk destroying vocal nuances.
        noise_floor: Noise floor in dB (-80 to -20). Default: -80.
        track_noise: Enable adaptive noise floor tracking. Default: True.
    """

    tn_flag = 1 if track_noise else 0
    af_filter = f"afftdn=nr={noise_reduction}:nf={noise_floor}:tn={tn_flag}"

    print(
        f"{ULTRASINGER_HEAD} Reduce noise from vocal audio with {blue_highlighted('ffmpeg')} (nr={noise_reduction}dB, nf={noise_floor}dB)."
    )
    try:
        (
            ffmpeg.input(input_file_path)
            .output(output_file, af=af_filter)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as ffmpeg_exception:
        print("ffmpeg stdout:", ffmpeg_exception.stdout.decode("utf8"))
        print("ffmpeg stderr:", ffmpeg_exception.stderr.decode("utf8"))
        raise ffmpeg_exception


def denoise_vocal_audio(input_path: str, output_path: str,
                        skip_cache: bool = False,
                        noise_reduction: float = 20,
                        noise_floor: float = -80,
                        track_noise: bool = True) -> None:
    """Denoise vocal audio"""
    cache_available = check_file_exists(output_path)
    if skip_cache or not cache_available:
        __ffmpeg_reduce_noise(input_path, output_path,
                              noise_reduction=noise_reduction,
                              noise_floor=noise_floor,
                              track_noise=track_noise)
    else:
        print(f"{ULTRASINGER_HEAD} {green_highlighted('cache')} reusing cached denoised audio")
