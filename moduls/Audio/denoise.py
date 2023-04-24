import ffmpeg


def ffmpeg_reduce_noise(input_file_path, output_file):
    print("\033[92m[UltraSinger]\033[97m Reduce noise from vocal audio")
    (
        ffmpeg
        .input(input_file_path)
        .output(output_file, af='afftdn=nr=30:nf=-30:tn=1')
        .overwrite_output()
        .run()
    )
