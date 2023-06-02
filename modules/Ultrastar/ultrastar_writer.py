"""Docstring"""

import re

import langcodes

from modules.Log import PRINT_ULTRASTAR
from modules.Ultrastar.ultrastar_converter import (
    real_bpm_to_ultrastar_bpm,
    second_to_beat,
)


def get_multiplier(real_bpm):
    """Docstring"""

    if real_bpm == 0:
        raise Exception("BPM is 0")

    multiplier = 1
    result = 0
    while result < 400:
        result = real_bpm * multiplier
        multiplier += 1
    return multiplier - 2


def get_language_name(language):
    """Docstring"""

    return langcodes.Language.make(language=language).display_name()


def create_ultrastar_txt_from_automation(
    transcribed_data,
    note_numbers,
    ultrastar_file_output,
    ultrastar_class,
    bpm=120,
):
    """Docstring"""

    print(
        f"{PRINT_ULTRASTAR} Creating {ultrastar_file_output} from transcription."
    )

    real_bpm = real_bpm_to_ultrastar_bpm(bpm)
    multiplication = get_multiplier(real_bpm)
    ultrastar_bpm = real_bpm * get_multiplier(real_bpm)

    with open(ultrastar_file_output, "w", encoding="utf-8") as file:
        gap = transcribed_data[0].start

        file.write(f"#ARTIST:{ultrastar_class.artist}\n")
        file.write(f"#TITLE:{ultrastar_class.title}\n")
        file.write(f"#CREATOR:{ultrastar_class.creator}\n")
        file.write(f"#FIXER:{ultrastar_class.fixer}\n")
        if ultrastar_class.language is not None:
            file.write(
                f"#LANGUAGE:{get_language_name(ultrastar_class.language)}\n"
            )
        if ultrastar_class.cover is not None:
            file.write(f"#COVER:{ultrastar_class.cover}\n")
        file.write(f"#MP3:{ultrastar_class.mp3}\n")
        file.write(f"#VIDEO:{ultrastar_class.video}\n")
        file.write(f"#BPM:{round(ultrastar_bpm, 2)}\n")  # not the real BPM!
        file.write(f"#GAP:{int(gap * 1000)}\n")
        file.write(f"#COMMENT:{ultrastar_class.comment}\n")

        # Write the singing part
        previous_end_beat = 0
        for i in enumerate(transcribed_data):
            start_time = (transcribed_data[i].start - gap) * multiplication
            end_time = (
                transcribed_data[i].end - transcribed_data[i].start
            ) * multiplication
            start_beat = round(second_to_beat(start_time, bpm))
            duration = round(second_to_beat(end_time, bpm))

            # Fix the round issue, so the beats don’t overlap
            start_beat = max(start_beat, previous_end_beat)
            previous_end_beat = start_beat + duration

            # : 10 10 10 w
            # ':'   start midi part
            # 'n1'  start at real beat
            # 'n2'  duration at real beat
            # 'n3'  pitch where 0 == C4
            # 'w'   lyric
            file.write(": ")
            file.write(str(start_beat) + " ")
            file.write(str(duration) + " ")
            file.write(str(note_numbers[i]) + " ")
            file.write(transcribed_data[i].word)
            file.write("\n")

            # detect silence between words
            if i < len(transcribed_data) - 1:
                silence = (
                    transcribed_data[i + 1].start - transcribed_data[i].end
                )
            else:
                silence = 0

            if silence > 0.3 and i != len(transcribed_data) - 1:
                # - 10
                # '-' end of current sing part
                # 'n1' show next at time in real beat
                file.write("- ")
                show_next = (
                    second_to_beat(transcribed_data[i].end - gap, bpm)
                    * multiplication
                )
                file.write(str(round(show_next)))
                file.write("\n")
        file.write("E")


def create_repitched_txt_from_ultrastar_data(
    input_file, note_numbers, output_repitched_ultrastar
):
    """Docstring"""
    # todo: just add '_repitched' to input_file
    print(
        "{PRINT_ULTRASTAR} Creating repitched ultrastar txt -> {input_file}_repitch.txt"
    )

    # todo: to reader
    with open(input_file, "r", encoding="utf-8") as file:
        txt = file.readlines()

    i = 0
    # todo: just add '_repitched' to input_file
    with open(output_repitched_ultrastar, "w", encoding="utf-8") as file:
        for line in txt:
            if line.startswith(":"):
                parts = re.findall(r"\S+|\s+", line)
                # between are whitespaces
                # [0] :
                # [2] start beat
                # [4] duration
                # [6] pitch
                # [8] word
                parts[6] = str(note_numbers[i])
                delimiter = ""
                file.write(delimiter.join(parts))
                i += 1
            else:
                file.write(line)


def add_score_to_ultrastar_txt(ultrastar_file_output, score):
    """Docstring"""
    with open(ultrastar_file_output, "r", encoding="utf-8") as file:
        text = file.read()
    text = text.split("\n")

    for i in enumerate(text):
        if text[i].startswith("#COMMENT:"):
            text[
                i
            ] = f"{text[i]} | Score: total: {score.score}, notes: {score.notes} line: {score.line_bonus}, golden: {score.golden}"
            break

        if text[i].startswith(("F ", ": ", "* ", "R ", "G ")):
            text.insert(
                i,
                f"#COMMENT: UltraSinger [GitHub] | Score: total: {score.score}, notes: {score.notes} line: {score.line_bonus}, golden: {score.golden}",
            )
            break

    text = "\n".join(text)

    with open(ultrastar_file_output, "w", encoding="utf-8") as file:
        file.write(text)


class UltraStarWriter:
    """Docstring"""
