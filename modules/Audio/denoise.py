import ffmpeg

from modules.Log import PRINT_ULTRASTAR, print_blue_highlighted_text


def ffmpeg_reduce_noise(input_file_path, output_file):
    # Denoise audio samples with FFT.
    # A description of the accepted parameters follows.

    # noise_reduction, nr
    #    Set the noise reduction in dB, allowed range is 0.01 to 97. Default value is 12 dB.
    # noise_floor, nf
    #    Set the noise floor in dB, allowed range is -80 to -20. Default value is -50 dB.
    # track_noise, tn
    #    Enable noise floor tracking. By default is disabled. With this enabled, noise floor is automatically adjusted.

    print(
        f"{PRINT_ULTRASTAR} Reduce noise from vocal audio with {print_blue_highlighted_text('ffmpeg')}."
    )
    (
        ffmpeg.input(input_file_path)
        .output(output_file, af="afftdn=nr=70:nf=-50:tn=1")
        .overwrite_output()
        .run()
    )
