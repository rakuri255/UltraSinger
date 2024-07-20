import copy
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import List
import importlib.util

import pandas

import UltraSinger
from modules.DeviceDetection.device_detection import check_gpu_support
from Settings import Settings
from modules.Research.TestRun import TestRun, TestedSong
from modules.Research.TestSong import TestSong
from modules.Ultrastar import ultrastar_parser
from modules.Ultrastar.ultrastar_converter import compare_pitches
from modules.Ultrastar.ultrastar_parser import parse_ultrastar_txt
from modules.Ultrastar.ultrastar_txt import UltrastarTxtValue, FILE_ENCODING
from modules.console_colors import ULTRASINGER_HEAD, red_highlighted

test_input_folder = os.path.normpath(os.path.abspath(__file__ + "/../../test_input"))
test_output_folder = os.path.normpath(os.path.abspath(__file__ + "/../../test_output"))

test_start_time = datetime.now()

test_run_name = test_start_time.strftime("%Y-%m-%d_%H-%M-%S")
test_run_folder = os.path.join(test_output_folder, test_run_name)
test_run_songs_folder = os.path.join(test_run_folder, "songs")


def main() -> None:
    """Main function"""
    Path(test_input_folder).mkdir(parents=True, exist_ok=True)
    Path(test_output_folder).mkdir(parents=True, exist_ok=True)
    Path(test_run_folder).mkdir(parents=True)
    Path(test_run_songs_folder).mkdir(parents=True)

    base_settings = initialize_settings()
    base_settings.output_folder_path = test_run_songs_folder

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
        found_song = find_ultrastar_song(song_folder)
        if found_song is None:
            continue

        test_songs.append(TestSong(found_song[0], song_folder, found_song[1]))

    if len(test_songs) == 0:
        print(
            f"{ULTRASINGER_HEAD} {red_highlighted('Error!')} No test songs found in {base_settings.test_songs_input_folder}."
        )
        exit(1)

    print(f"{ULTRASINGER_HEAD} Running evaluation for {len(test_songs)} songs")

    test_run = TestRun(test_run_name, base_settings, test_start_time)
    for index, test_song in enumerate(test_songs):
        print(f"{ULTRASINGER_HEAD} ========================")
        print(
            f"{ULTRASINGER_HEAD} {index+1}/{len(test_songs)}: {os.path.basename(test_song.input_txt)}"
        )

        # prepare cache directory
        song_cache_path = os.path.join(test_song.input_folder, "cache")
        Path(song_cache_path).mkdir(parents=True, exist_ok=True)

        test_song_settings = copy.deepcopy(base_settings)
        test_song_settings.input_file_path = test_song.input_txt
        test_song_settings.cache_override_path = song_cache_path
        UltraSinger.settings = test_song_settings

        tested_song = TestedSong(test_song.input_txt)
        test_run.tested_songs.append(tested_song)
        try:
            output_txt, _, _ = UltraSinger.run()
        except Exception as error:
            print(
                f"{ULTRASINGER_HEAD} {red_highlighted('Error!')} Failed to process {test_song.input_txt}\n{error}."
            )
            traceback.print_exc()
            continue


        output_folder_name = f"{test_song.input_ultrastar_class.artist} - {test_song.input_ultrastar_class.title}"
        output_folder = os.path.join(test_run_songs_folder, output_folder_name)

        if not os.path.isfile(output_txt):
            print(
                f"{ULTRASINGER_HEAD} {red_highlighted('Error!')} Could not find song txt in '{output_folder}'."
            )
            test_run.tested_songs.append(tested_song)
            continue

        ultrastar_class = parse_ultrastar_txt(output_txt)
        (
            input_match_ratio,
            output_match_ratio,
            input_pitch_shift_match_ratios,
            output_pitch_shift_match_ratios,
            pitch_where_should_be_no_pitch_ratio,
            no_pitch_where_should_be_pitch_ratio,
        ) = compare_pitches(test_song.input_ultrastar_class, ultrastar_class)

        tested_song.output_path = output_txt
        tested_song.success = True
        tested_song.input_match_ratio = input_match_ratio
        tested_song.output_match_ratio = output_match_ratio
        tested_song.input_pitch_shift_match_ratios = input_pitch_shift_match_ratios
        tested_song.output_pitch_shift_match_ratios = output_pitch_shift_match_ratios
        tested_song.pitch_where_should_be_no_pitch_ratio = pitch_where_should_be_no_pitch_ratio
        tested_song.no_pitch_where_should_be_pitch_ratio = no_pitch_where_should_be_pitch_ratio

    test_run.end_time = datetime.now()
    test_run_result_file = os.path.join(test_run_folder, "run.json")
    test_run_json = test_run.to_json()
    with open(test_run_result_file, "w", encoding=FILE_ENCODING) as file:
        file.write(test_run_json)


def find_ultrastar_song(
    song_folder, require_audio: bool = True
) -> tuple[str, UltrastarTxtValue]:
    if os.path.isdir(song_folder):
        for song_folder_item in os.listdir(song_folder):
            if (
                song_folder_item.endswith(".txt")
                and song_folder_item != "license.txt"
                and not song_folder_item.endswith("[Karaoke].txt")
                and not song_folder_item.endswith("[MULTI].txt")
                and not song_folder_item.endswith("[DUET].txt")
                and not song_folder_item.endswith("instrumental.txt")
            ):
                txt_file = os.path.join(song_folder, song_folder_item)
                ultrastar_class = ultrastar_parser.parse_ultrastar_txt(txt_file)

                if ultrastar_class.mp3 != "" or not require_audio:
                    return txt_file, ultrastar_class
                else:
                    print(
                        f"{ULTRASINGER_HEAD} {red_highlighted('Warning.')} {song_folder} contains an UltraStar text file but has no audio referenced in it. Skipping."
                    )


def initialize_settings():
    s = Settings()
    user_config_file = os.path.normpath(
        os.path.join(test_input_folder, "config/local.py")
    )
    if os.path.isfile(user_config_file):
        print(
            f"{ULTRASINGER_HEAD} Using custom settings found under {user_config_file}"
        )

        spec = importlib.util.spec_from_file_location(
            "custom_settings", user_config_file
        )
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
