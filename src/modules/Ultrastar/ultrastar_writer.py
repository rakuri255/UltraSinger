"""Ultrastar writer module"""

import re
import langcodes
from packaging import version

from modules.console_colors import ULTRASINGER_HEAD
from modules.Ultrastar.ultrastar_converter import (
    real_bpm_to_ultrastar_bpm,
    second_to_beat,
    beat_to_second,
)
from modules.Ultrastar.ultrastar_txt import UltrastarTxtValue, UltrastarTxtTag, UltrastarTxtNoteTypeTag, \
    FILE_ENCODING
from modules.Speech_Recognition.TranscribedData import TranscribedData
from modules.Ultrastar.ultrastar_score_calculator import Score


def get_thirtytwo_note_second(real_bpm: float):
    """Converts a beat to a 1/32 note in second"""
    return 60 / real_bpm / 8


def get_sixteenth_note_second(real_bpm: float):
    """Converts a beat to a 1/16 note in second"""
    return 60 / real_bpm / 4


def get_eighth_note_second(real_bpm: float):
    """Converts a beat to a 1/8 note in second"""
    return 60 / real_bpm / 2


def get_quarter_note_second(real_bpm: float):
    """Converts a beat to a 1/4 note in second"""
    return 60 / real_bpm


def get_half_note_second(real_bpm: float):
    """Converts a beat to a 1/2 note in second"""
    return 60 / real_bpm * 2


def get_whole_note_second(real_bpm: float):
    """Converts a beat to a 1/1 note in second"""
    return 60 / real_bpm * 4


def get_multiplier(real_bpm: float) -> int:
    """Calculates the multiplier for the BPM"""

    if real_bpm == 0:
        raise Exception("BPM is 0")

    multiplier = 1
    result = 0
    while result < 400:
        result = real_bpm * multiplier
        multiplier += 1
    return multiplier - 2


def get_language_name(language: str) -> str:
    """Creates the language name from the language code"""

    return langcodes.Language.make(language=language).display_name()


def create_ultrastar_txt_from_automation(
        transcribed_data: list[TranscribedData],
        note_numbers: list[int],
        ultrastar_file_output: str,
        ultrastar_class: UltrastarTxtValue,
        real_bpm=120,
) -> None:
    """Creates an Ultrastar txt file from the automation data"""

    print(
        f"{ULTRASINGER_HEAD} Creating {ultrastar_file_output} from transcription."
    )

    ultrastar_bpm = real_bpm_to_ultrastar_bpm(real_bpm)
    multiplication = get_multiplier(ultrastar_bpm)
    ultrastar_bpm = ultrastar_bpm * get_multiplier(ultrastar_bpm)
    silence_split_duration = calculate_silent_beat_length(transcribed_data)

    with open(ultrastar_file_output, "w", encoding=FILE_ENCODING) as file:
        gap = transcribed_data[0].start

        if version.parse(ultrastar_class.version) >= version.parse("1.0.0"):
            file.write(f"#{UltrastarTxtTag.VERSION}:{ultrastar_class.version}\n"),
        file.write(f"#{UltrastarTxtTag.ARTIST}:{ultrastar_class.artist}\n")
        file.write(f"#{UltrastarTxtTag.TITLE}:{ultrastar_class.title}\n")
        if ultrastar_class.year is not None:
            file.write(f"#{UltrastarTxtTag.YEAR}:{ultrastar_class.year}\n")
        if ultrastar_class.language is not None:
            file.write(f"#{UltrastarTxtTag.LANGUAGE}:{get_language_name(ultrastar_class.language)}\n")
        if ultrastar_class.genre:
            file.write(f"#{UltrastarTxtTag.GENRE}:{ultrastar_class.genre}\n")
        if ultrastar_class.cover is not None:
            file.write(f"#{UltrastarTxtTag.COVER}:{ultrastar_class.cover}\n")
        file.write(f"#{UltrastarTxtTag.MP3}:{ultrastar_class.mp3}\n")
        if version.parse(ultrastar_class.version) >= version.parse("1.1.0"):
            file.write(f"#{UltrastarTxtTag.AUDIO}:{ultrastar_class.audio}\n")
            if ultrastar_class.vocals is not None:
                file.write(f"#{UltrastarTxtTag.VOCALS}:{ultrastar_class.vocals}\n")
            if ultrastar_class.instrumental is not None:
                file.write(f"#{UltrastarTxtTag.INSTRUMENTAL}:{ultrastar_class.instrumental}\n")
            if ultrastar_class.tags is not None:
                file.write(f"#{UltrastarTxtTag.TAGS}:{ultrastar_class.tags}\n")
        file.write(f"#{UltrastarTxtTag.VIDEO}:{ultrastar_class.video}\n")
        file.write(f"#{UltrastarTxtTag.BPM}:{round(ultrastar_bpm, 2)}\n")  # not the real BPM!
        file.write(f"#{UltrastarTxtTag.GAP}:{int(gap * 1000)}\n")
        file.write(f"#{UltrastarTxtTag.CREATOR}:{ultrastar_class.creator}\n")
        file.write(f"#{UltrastarTxtTag.COMMENT}:{ultrastar_class.comment}\n")

        # Write the singing part
        previous_end_beat = 0
        separated_word_silence = []  # This is a workaround for separated words that get his ends to far away

        for i, data in enumerate(transcribed_data):
            start_time = (data.start - gap) * multiplication
            end_time = (
                               data.end - data.start
                       ) * multiplication
            start_beat = round(second_to_beat(start_time, real_bpm))
            duration = round(second_to_beat(end_time, real_bpm))

            # Fix the round issue, so the beats don’t overlap
            start_beat = max(start_beat, previous_end_beat)
            previous_end_beat = start_beat + duration

            # Calculate the silence between the words
            if i < len(transcribed_data) - 1:
                silence = (transcribed_data[i + 1].start - data.end)
            else:
                silence = 0

            # : 10 10 10 w
            # ':'   start midi part
            # 'n1'  start at real beat
            # 'n2'  duration at real beat
            # 'n3'  pitch where 0 == C4
            # 'w'   lyric
            line = f"{UltrastarTxtNoteTypeTag.NORMAL} " \
                   f"{str(start_beat)} " \
                   f"{str(duration)} " \
                   f"{str(note_numbers[i])} " \
                   f"{data.word}\n"

            file.write(line)

            # detect silence between words
            if not transcribed_data[i].is_word_end:
                separated_word_silence.append(silence)
                continue

            if silence_split_duration != 0 and silence > silence_split_duration or any(
                    s > silence_split_duration for s in separated_word_silence) and i != len(transcribed_data) - 1:
                # - 10
                # '-' end of current sing part
                # 'n1' show next at time in real beat
                show_next = (
                        second_to_beat(data.end - gap, real_bpm)
                        * multiplication
                )
                linebreak = f"{UltrastarTxtTag.LINEBREAK} " \
                            f"{str(round(show_next))}\n"
                file.write(linebreak)
            separated_word_silence = []
        file.write(f"{UltrastarTxtTag.FILE_END}")


def deviation(silence_parts):
    """Calculates the deviation of the silence parts"""

    if len(silence_parts) < 5:
        return 0

    # Remove the longest part so the deviation is not that high
    sorted_parts = sorted(silence_parts)
    filtered_parts = [part for part in sorted_parts if part < sorted_parts[-1]]

    mean = sum(filtered_parts) / len(filtered_parts)
    return mean


def calculate_silent_beat_length(transcribed_data: list[TranscribedData]):
    print(f"{ULTRASINGER_HEAD} Calculating silence parts for linebreaks.")

    silent_parts = []
    for i, data in enumerate(transcribed_data):
        if i < len(transcribed_data) - 1:
            silent_parts.append(transcribed_data[i + 1].start - data.end)

    return deviation(silent_parts)


def create_repitched_txt_from_ultrastar_data(
        input_file: str, note_numbers: list[int], output_repitched_ultrastar: str
) -> None:
    """Creates a repitched ultrastar txt file from the original one"""
    # todo: just add '_repitched' to input_file
    print(
        "{PRINT_ULTRASTAR} Creating repitched ultrastar txt -> {input_file}_repitch.txt"
    )

    # todo: to reader
    with open(input_file, "r", encoding=FILE_ENCODING) as file:
        txt = file.readlines()

    i = 0
    # todo: just add '_repitched' to input_file
    with open(output_repitched_ultrastar, "w", encoding=FILE_ENCODING) as file:
        for line in txt:
            if line.startswith(f"{UltrastarTxtNoteTypeTag.NORMAL} "):
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


def add_score_to_ultrastar_txt(ultrastar_file_output: str, score: Score) -> None:
    """Adds the score to the ultrastar txt file"""
    with open(ultrastar_file_output, "r", encoding=FILE_ENCODING) as file:
        text = file.read()
    text = text.split("\n")

    for i, line in enumerate(text):
        if line.startswith(f"#{UltrastarTxtTag.COMMENT}:"):
            text[
                i
            ] = f"{line} | Score: total: {score.score}, notes: {score.notes} line: {score.line_bonus}, golden: {score.golden}"
            break

        if line.startswith((
                f"{UltrastarTxtNoteTypeTag.FREESTYLE} ",
                f"{UltrastarTxtNoteTypeTag.NORMAL} ",
                f"{UltrastarTxtNoteTypeTag.GOLDEN} ",
                f"{UltrastarTxtNoteTypeTag.RAP} ",
                f"{UltrastarTxtNoteTypeTag.RAP_GOLDEN} ")):
            text.insert(
                i,
                f"#{UltrastarTxtTag.COMMENT}: UltraSinger [GitHub] | Score: total: {score.score}, notes: {score.notes} line: {score.line_bonus}, golden: {score.golden}",
            )
            break

    text = "\n".join(text)

    with open(ultrastar_file_output, "w", encoding=FILE_ENCODING) as file:
        file.write(text)


class UltraStarWriter:
    """Docstring"""
