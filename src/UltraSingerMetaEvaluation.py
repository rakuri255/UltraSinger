import os
from pathlib import Path
from typing import List

import pandas

from modules.Evaluation.TestRun import TestRun
from modules.console_colors import ULTRASINGER_HEAD, red_highlighted

test_output_folder = os.path.normpath(os.path.abspath(__file__ + "/../../evaluation/output"))


def main() -> None:
    """Main function"""
    Path(test_output_folder).mkdir(parents=True, exist_ok=True)

    test_runs: List[TestRun] = []
    for dir_entry in os.listdir(test_output_folder):
        test_run_folder = os.path.join(test_output_folder, dir_entry)
        test_run = find_test_run_result(test_run_folder)
        if test_run is None:
            continue

        test_runs.append(test_run)

    if len(test_runs) == 0:
        print(
            f"{ULTRASINGER_HEAD} {red_highlighted('Error!')} No test runs found in {test_output_folder}."
        )
        exit(1)

    print(f"{ULTRASINGER_HEAD} Running meta evaluation for {len(test_runs)} test runs")

    for test_run in test_runs:
        tested_songs_dicts = []
        for tested_song in [s for s in test_run.tested_songs if s.success]:
            tested_song_dict = tested_song.to_dict()

            best_input_pitch_shift_match_ratio = max(
                tested_song.input_pitch_shift_match_ratios.values()
            )

            # based on the pitch shift of the highest input_pitch_shift_match_ratio picked previously
            # we pick the corresponding value of output_pitch_shift_match_ratios
            matching_input_best_output_pitch_shift_match_ratio = (
                tested_song.output_pitch_shift_match_ratios[
                    list(tested_song.input_pitch_shift_match_ratios.values()).index(
                        best_input_pitch_shift_match_ratio
                    )
                ]
            )

            best_output_pitch_shift_match_ratio = max(
                tested_song.output_pitch_shift_match_ratios.values()
            )

            # based on the pitch shift of the highest output_pitch_shift_match_ratio picked previously
            # we pick the corresponding value of input_pitch_shift_match_ratios
            matching_output_best_input_pitch_shift_match_ratio = (
                tested_song.input_pitch_shift_match_ratios[
                    list(tested_song.output_pitch_shift_match_ratios.values()).index(
                        best_output_pitch_shift_match_ratio
                    )
                ]
            )

            tested_song_dict[
                "best_input_pitch_shift_match_ratio"
            ] = best_input_pitch_shift_match_ratio
            tested_song_dict[
                "matching_input_best_output_pitch_shift_match_ratio"
            ] = matching_input_best_output_pitch_shift_match_ratio
            tested_song_dict[
                "best_output_pitch_shift_match_ratio"
            ] = best_output_pitch_shift_match_ratio
            tested_song_dict[
                "matching_output_best_input_pitch_shift_match_ratio"
            ] = matching_output_best_input_pitch_shift_match_ratio

            tested_songs_dicts.append(tested_song_dict)

        records = pandas.DataFrame.from_records(tested_songs_dicts)
        pandas.options.display.max_columns = records.shape[1]
        pandas.set_option('display.expand_frame_repr', False)
        describe_result = records.describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99])

        print("Test run:", test_run.name)
        print(describe_result)

    print("Done")


def find_test_run_result(test_run_folder) -> TestRun:
    if os.path.isdir(test_run_folder):
        for test_run_folder_item in os.listdir(test_run_folder):
            test_run_folder_item_path = os.path.join(
                test_run_folder, test_run_folder_item
            )
            if (
                os.path.isfile(test_run_folder_item_path)
                and test_run_folder_item == "run.json"
            ):
                test_run = None
                with open(test_run_folder_item_path) as file:
                    json = file.read()
                    test_run = TestRun.from_json(json)
                return test_run


if __name__ == "__main__":
    main()
