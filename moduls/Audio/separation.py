import subprocess
from moduls.os_helper import move, current_executor_path, path_join


def separate_audio(input_file_path, output_file):
    print("[UltraSinger] Separating audio")
    # Model selection?
    # -n mdx_q
    # -n htdemucs_ft
    subprocess.run(["demucs", "--two-stems=vocals", input_file_path])
    separated_folder = path_join(current_executor_path(), "separated")
    move(separated_folder, output_file)
