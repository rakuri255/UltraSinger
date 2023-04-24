import ffmpeg
from moduls.Log import PRINT_ULTRASTAR


def ffmpeg_reduce_noise(input_file_path, output_file):
    print(PRINT_ULTRASTAR + " Reduce noise from vocal audio")
    (
        ffmpeg
        .input(input_file_path)
        .output(output_file, af='afftdn=nr=30:nf=-30:tn=1')
        .overwrite_output()
        .run()
    )
