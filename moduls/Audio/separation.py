import subprocess
from moduls.os_helper import move, current_executor_path


def separate_audio(input_file_path, output_file):
    print("Separating audio")
    # Model selection?
    # -n mdx_q
    subprocess.run(["demucs", "--two-stems=vocals", input_file_path])
    separated_folder = current_executor_path() + "/separated"
    move(separated_folder, output_file)
