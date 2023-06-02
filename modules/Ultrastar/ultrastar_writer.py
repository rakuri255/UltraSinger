import re

import langcodes

from modules.Log import PRINT_ULTRASTAR
from modules.Ultrastar.ultrastar_converter import (
    real_bpm_to_ultrastar_bpm,
    second_to_beat,
)


def get_multiplier(real_bpm):
    if real_bpm == 0:
        raise Exception("BPM is 0")

    multiplier = 1
    result = 0
    while result < 400:
        result = real_bpm * multiplier
        multiplier += 1
    return multiplier - 2


def get_language_name(language):
    return langcodes.Language.make(language=language).display_name()


def create_ultrastar_txt_from_automation(
    transcribed_data,
    note_numbers,
    ultrastar_file_output,
    ultrastar_class,
    bpm=120,
):
    print(
        f"{PRINT_ULTRASTAR} Creating {ultrastar_file_output} from transcription."
    )

    real_bpm = real_bpm_to_ultrastar_bpm(bpm)
    multiplication = get_multiplier(real_bpm)
    ultrastar_bpm = real_bpm * get_multiplier(real_bpm)

    with open(ultrastar_file_output, "w", encoding="utf8") as f:
        gap = transcribed_data[0].start

        f.write(f"#ARTIST:{ultrastar_class.artist}\n")
        f.write(f"#TITLE:{ultrastar_class.title}\n")
        f.write(f"#CREATOR:{ultrastar_class.creator}\n")
        f.write(f"#FIXER:{ultrastar_class.fixer}\n")
        if ultrastar_class.language is not None:
            f.write(
                f"#LANGUAGE:{get_language_name(ultrastar_class.language)}\n"
            )
        if ultrastar_class.cover is not None:
            f.write(f"#COVER:{ultrastar_class.cover}\n")
        f.write(f"#MP3:{ultrastar_class.mp3}\n")
        f.write(f"#VIDEO:{ultrastar_class.video}\n")
        f.write(
            f"#BPM:" + str(round(ultrastar_bpm, 2)) + "\n"
        )  # not the real BPM!
        f.write(f"#GAP:" + str(int(gap * 1000)) + "\n")
        f.write(f"#COMMENT:{ultrastar_class.comment}\n")

        # Write the singing part
        previous_end_beat = 0
        for i in range(len(transcribed_data)):
            start_time = (transcribed_data[i].start - gap) * multiplication
            end_time = (
                transcribed_data[i].end - transcribed_data[i].start
            ) * multiplication
            start_beat = round(second_to_beat(start_time, bpm))
            duration = round(second_to_beat(end_time, bpm))

            # Fix the round issue, so the beats dont overlap
            if start_beat < previous_end_beat:
                start_beat = previous_end_beat
            previous_end_beat = start_beat + duration

            # : 10 10 10 w
            # ':'   start midi part
            # 'n1'  start at real beat
            # 'n2'  duration at real beat
            # 'n3'  pitch where 0 == C4
            # 'w'   lyric
            f.write(": ")
            f.write(str(start_beat) + " ")
            f.write(str(duration) + " ")
            f.write(str(note_numbers[i]) + " ")
            f.write(transcribed_data[i].word)
            f.write("\n")

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
                f.write("- ")
                show_next = (
                    second_to_beat(transcribed_data[i].end - gap, bpm)
                    * multiplication
                )
                f.write(str(round(show_next)))
                f.write("\n")
        f.write("E")


def create_repitched_txt_from_ultrastar_data(
    input_file, note_numbers, output_repitched_ultrastar
):
    # todo: just add '_repitched' to input_file
    print(
        "{PRINT_ULTRASTAR} Creating repitched ultrastar txt -> {input_file}_repitch.txt"
    )

    # todo: to reader
    file = open(input_file, "r")
    txt = file.readlines()

    i = 0
    # todo: just add '_repitched' to input_file
    with open(output_repitched_ultrastar, "w", encoding="utf8") as f:
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
                f.write(delimiter.join(parts))
                i += 1
            else:
                f.write(line)


def add_score_to_ultrastar_txt(ultrastar_file_output, score):
    with open(ultrastar_file_output, "r", encoding="utf8") as f:
        text = f.read()
    text = text.split("\n")

    for i in range(len(text)):
        if text[i].startswith("#COMMENT:"):
            text[
                i
            ] = f"{text[i]} | Score: total: {score.score}, notes: {score.notes} line: {score.line_bonus}, golden: {score.golden}"
            break
        elif text[i].startswith(("F ", ": ", "* ", "R ", "G ")):
            text.insert(
                i,
                f"#COMMENT: UltraSinger [GitHub] | Score: total: {score.score}, notes: {score.notes} line: {score.line_bonus}, golden: {score.golden}",
            )
            break

    text = "\n".join(text)

    with open(ultrastar_file_output, "w", encoding="utf8") as f:
        f.write(text)


class UltraStarWriter:
    pass
