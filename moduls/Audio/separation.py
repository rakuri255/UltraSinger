import subprocess
from moduls.os_helper import move, current_executor_path, path_join
from moduls.Log import PRINT_ULTRASTAR, print_blue_highlighted_text, print_red_highlighted_text


def separate_audio(input_file_path, output_file, device="cpu"):
    print("{} Separating vocals from audio with {} and {} as worker.".format(PRINT_ULTRASTAR, print_blue_highlighted_text("demucs"), print_red_highlighted_text(device)))
    # Model selection?
    # -n mdx_q
    # -n htdemucs_ft
    device = "cpu" if device != "cuda" else "cuda"
    subprocess.run(["demucs", "-d", device, "--two-stems=vocals", input_file_path])

    separated_folder = path_join(current_executor_path(), "separated")
    move(separated_folder, output_file)
