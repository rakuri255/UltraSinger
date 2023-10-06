import copy
import os
from datetime import datetime
from pathlib import Path
from typing import List
import importlib.util

import UltraSinger
from Settings import Settings
from modules.DeviceDetection.device_detection import check_gpu_support
from modules.Research.TestSong import TestSong
from modules.Ultrastar import ultrastar_parser
from modules.console_colors import ULTRASINGER_HEAD, red_highlighted

test_input_folder = os.path.normpath(
    os.path.abspath(__file__ + "/../../test_input")
)
test_output_folder = os.path.normpath(
    os.path.abspath(__file__ + "/../../test_output")
)
test_run_folder = os.path.join(
    test_output_folder, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)


def main() -> None:
    """Main function"""
    Path(test_input_folder).mkdir(parents=True, exist_ok=True)
    Path(test_output_folder).mkdir(parents=True, exist_ok=True)
    Path(test_run_folder).mkdir(parents=True)

    base_settings = initialize_settings()
    base_settings.output_file_path = test_run_folder

    base_settings.test_songs_input_folder = os.path.normpath(
        base_settings.test_songs_input_folder
    )
    if not os.path.isdir(base_settings.test_songs_input_folder):
        print(
            f"{ULTRASINGER_HEAD} {red_highlighted('Error!')} No test songs input folder configured (refer to "
            f"evaluation section in readme)."
        )
        exit(1)

    test_songs: List[TestSong] = []
    for dir_entry in os.listdir(base_settings.test_songs_input_folder):
        song_folder = os.path.join(base_settings.test_songs_input_folder, dir_entry)
        if os.path.isdir(song_folder):
            for song_folder_item in os.listdir(song_folder):
                if song_folder_item.endswith(".txt") and song_folder_item != "license.txt":
                    txt_file = os.path.join(song_folder, song_folder_item)
                    ultrastar_class = ultrastar_parser.parse_ultrastar_txt(txt_file)

                    if ultrastar_class.mp3:
                        test_song = TestSong(txt_file, song_folder, ultrastar_class.mp3, ultrastar_class)
                        test_songs.append(test_song)
                        break
                    else:
                        print(
                            f"{ULTRASINGER_HEAD} {red_highlighted('Warning.')} {base_settings.test_songs_input_folder} contains an UltraStar text file but has no audio referenced in it. Skipping."
                        )

    if len(test_songs) == 0:
        print(
            f"{ULTRASINGER_HEAD} {red_highlighted('Error!')} No test songs found in {base_settings.test_songs_input_folder}."
        )
        exit(1)

    print(f"{ULTRASINGER_HEAD} Running evaluation for {len(test_songs)} songs")

    for index, test_song in enumerate(test_songs):
        print(f"{ULTRASINGER_HEAD} ========================")
        print(
            f"{ULTRASINGER_HEAD} {index+1}/{len(test_songs)}: {os.path.basename(test_song.txt)}"
        )

        # prepare cache directory
        song_cache_path = os.path.join(test_song.folder, "cache")
        Path(song_cache_path).mkdir(parents=True, exist_ok=True)
        test_song_settings = copy.deepcopy(base_settings)
        test_song_settings.input_file_path = test_song.txt
        test_song_settings.cache_override_path = song_cache_path
        UltraSinger.settings = test_song_settings
        UltraSinger.run()


def initialize_settings():
    s = Settings()
    user_config_file = os.path.normpath(
        os.path.join(test_input_folder, "config/local.py")
    )
    if os.path.isfile(user_config_file):
        print(f"{ULTRASINGER_HEAD} Using custom settings found under {user_config_file}")

        spec = importlib.util.spec_from_file_location("custom_settings", user_config_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        s = module.user_settings
    else:
        print(f"{ULTRASINGER_HEAD} No custom settings found under {user_config_file}")

    if not s.force_cpu:
        s.tensorflow_device, s.pytorch_device = check_gpu_support()
    return s


if __name__ == "__main__":
    main()
