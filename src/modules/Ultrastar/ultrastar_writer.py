"""Ultrastar writer module"""

import math
import re
import langcodes
import numpy as np
from packaging import version

from modules.console_colors import ULTRASINGER_HEAD
from modules.Ultrastar.coverter.ultrastar_converter import (
    real_bpm_to_ultrastar_bpm,
    second_to_beat, )
from modules.Ultrastar.coverter.ultrastar_midi_converter import convert_midi_note_to_ultrastar_note
from modules.Ultrastar.ultrastar_txt import UltrastarTxtValue, UltrastarTxtTag, UltrastarTxtNoteTypeTag, \
    FILE_ENCODING
from modules.Ultrastar.ultrastar_score_calculator import Score
from modules.Midi.MidiSegment import MidiSegment


def get_multiplier(real_bpm: float) -> int:
    """Calculates the multiplier so that real_bpm * multiplier >= 400.

    A higher multiplier gives finer beat resolution, reducing rounding errors
    when converting seconds to integer beats. The threshold of 400 ensures
    at least ~6.67 beats per second at the lowest BPM.
    """

    if real_bpm <= 0:
        raise Exception("BPM must be positive")

    multiplier = 1
    while real_bpm * multiplier < 400:
        multiplier += 1
    return multiplier


def get_language_name(language: str) -> str:
    """Creates the language name from the language code"""

    return langcodes.Language.make(language=language).display_name()


def create_ultrastar_txt(
        midi_segments: list[MidiSegment],
        ultrastar_file_output: str,
        ultrastar_class: UltrastarTxtValue,
        real_bpm: float) -> None:
    """Creates an Ultrastar txt file from the automation data"""

    print(f"{ULTRASINGER_HEAD} Creating UltraStar file {ultrastar_file_output}")

    ultrastar_bpm = real_bpm_to_ultrastar_bpm(real_bpm)
    multiplication = get_multiplier(ultrastar_bpm)
    ultrastar_bpm = ultrastar_bpm * multiplication
    silence_split_duration = calculate_silent_beat_length(midi_segments)

    with open(ultrastar_file_output, "w", encoding=FILE_ENCODING) as file:
        gap = midi_segments[0].start

        if version.parse(ultrastar_class.version) >= version.parse("1.0.0"):
            file.write(f"#{UltrastarTxtTag.VERSION.value}:{ultrastar_class.version}\n")
        file.write(f"#{UltrastarTxtTag.ARTIST.value}:{ultrastar_class.artist}\n")
        file.write(f"#{UltrastarTxtTag.TITLE.value}:{ultrastar_class.title}\n")
        if ultrastar_class.year is not None:
            file.write(f"#{UltrastarTxtTag.YEAR.value}:{ultrastar_class.year}\n")
        if ultrastar_class.language is not None:
            file.write(f"#{UltrastarTxtTag.LANGUAGE.value}:{get_language_name(ultrastar_class.language)}\n")
        if ultrastar_class.genre:
            file.write(f"#{UltrastarTxtTag.GENRE.value}:{ultrastar_class.genre}\n")
        if ultrastar_class.cover is not None:
            file.write(f"#{UltrastarTxtTag.COVER.value}:{ultrastar_class.cover}\n")
        if version.parse(ultrastar_class.version) >= version.parse("1.2.0"):
            if ultrastar_class.coverUrl is not None:
                file.write(f"#{UltrastarTxtTag.COVERURL.value}:{ultrastar_class.coverUrl}\n")
        if ultrastar_class.background is not None:
            file.write(f"#{UltrastarTxtTag.BACKGROUND.value}:{ultrastar_class.background}\n")
        file.write(f"#{UltrastarTxtTag.MP3.value}:{ultrastar_class.mp3}\n")
        if version.parse(ultrastar_class.version) >= version.parse("1.1.0"):
            file.write(f"#{UltrastarTxtTag.AUDIO.value}:{ultrastar_class.audio}\n")
            if ultrastar_class.vocals is not None:
                file.write(f"#{UltrastarTxtTag.VOCALS.value}:{ultrastar_class.vocals}\n")
            if ultrastar_class.instrumental is not None:
                file.write(f"#{UltrastarTxtTag.INSTRUMENTAL.value}:{ultrastar_class.instrumental}\n")
        if ultrastar_class.video is not None:
            file.write(f"#{UltrastarTxtTag.VIDEO.value}:{ultrastar_class.video}\n")
        if ultrastar_class.videoGap is not None:
            file.write(f"#{UltrastarTxtTag.VIDEOGAP.value}:{ultrastar_class.videoGap}\n")
        if version.parse(ultrastar_class.version) >= version.parse("1.2.0"):
            if ultrastar_class.videoUrl is not None:
                file.write(f"#{UltrastarTxtTag.VIDEOURL.value}:{ultrastar_class.videoUrl}\n")
        file.write(f"#{UltrastarTxtTag.BPM.value}:{round(ultrastar_bpm, 2)}\n")  # not the real BPM!
        file.write(f"#{UltrastarTxtTag.GAP.value}:{int(gap * 1000)}\n")
        if version.parse(ultrastar_class.version) >= version.parse("1.1.0"):
            if ultrastar_class.tags is not None:
                file.write(f"#{UltrastarTxtTag.TAGS.value}:{ultrastar_class.tags}\n")
        file.write(f"#{UltrastarTxtTag.CREATOR.value}:{ultrastar_class.creator}\n")

        # Write the singing part
        previous_end_beat = 0
        separated_word_silence = []  # This is a workaround for separated words that get his ends to far away

        for i, midi_segment in enumerate(midi_segments):
            start_time = (midi_segment.start - gap) * multiplication
            end_time = (midi_segment.end - midi_segment.start) * multiplication

            # Use floor for start (prefer slightly early over late) and
            # max(1, ...) for duration (every note must be at least 1 beat).
            # round() caused ±0.5 beat errors that accumulated over the song
            # because max() below only shifts notes later, never earlier.
            start_beat = math.floor(second_to_beat(start_time, real_bpm))
            duration = max(1, math.ceil(second_to_beat(end_time, real_bpm)))

            # Prevent overlap: shift start to after previous note if needed
            if start_beat < previous_end_beat:
                start_beat = previous_end_beat
            previous_end_beat = start_beat + duration

            # Calculate the silence between the words
            if i < len(midi_segments) - 1:
                silence = (midi_segments[i + 1].start - midi_segment.end)
            else:
                silence = 0

            # : 10 10 10 w
            # ':'   start midi part
            # 'n1'  start at real beat
            # 'n2'  duration at real beat
            # 'n3'  pitch where 0 == C4
            # 'w'   lyric
            line = f"{UltrastarTxtNoteTypeTag.NORMAL.value} " \
                   f"{str(start_beat)} " \
                   f"{str(duration)} " \
                   f"{str(convert_midi_note_to_ultrastar_note(midi_segment))} " \
                   f"{midi_segment.word}\n"

            file.write(line)

            # detect silence between words
            if not midi_segment.word.endswith(" "):
                separated_word_silence.append(silence)
                continue

            if silence_split_duration is not None and (
                    i != len(midi_segments) - 1 and silence > silence_split_duration
                    or any(s > silence_split_duration for s in separated_word_silence)):
                # - 10
                # '-' end of current sing part
                # 'n1' show next at time in real beat
                show_next = (
                        second_to_beat(midi_segment.end - gap, real_bpm)
                        * multiplication
                )
                linebreak = f"{UltrastarTxtTag.LINEBREAK.value} " \
                            f"{str(math.floor(show_next))}\n"
                file.write(linebreak)
            separated_word_silence = []
        file.write(f"{UltrastarTxtTag.FILE_END.value}")


def silence_threshold(
    silence_parts: list[float], percentile: float = 85
) -> float | None:
    """Calculate the silence duration threshold for linebreaks.

    Uses a percentile-based approach: only silences above the given
    percentile of all inter-note gaps trigger a linebreak.  This is
    more robust than a mean-based approach, which either creates too
    many linebreaks (when pauses are evenly distributed) or too few
    (when a single long pause skews the average).

    Args:
        silence_parts: List of silence durations in seconds between
            consecutive notes.
        percentile: The percentile of silence durations to use as the
            threshold (default 85 — only the longest 15 % of gaps
            become linebreaks).  A higher percentile produces fewer
            linebreaks; 85 was chosen empirically to match the density
            of manually-authored UltraStar files.

    Returns:
        The silence duration threshold in seconds, or ``None`` when
        there are fewer than 5 gaps (too few to compute a meaningful
        threshold).  Callers must check for ``None`` before comparing.
    """
    if len(silence_parts) < 5:
        return None

    return float(np.percentile(silence_parts, percentile))


def calculate_silent_beat_length(
    midi_segments: list[MidiSegment], percentile: float = 85
) -> float | None:
    """Extract inter-note silence durations and compute a threshold.

    This is a convenience wrapper around :func:`silence_threshold`
    that collects the gaps between consecutive MIDI segments.

    Args:
        midi_segments: List of MIDI segments with start/end times.
        percentile: Passed to :func:`silence_threshold`.

    Returns:
        The silence duration threshold (see :func:`silence_threshold`).
    """
    print(f"{ULTRASINGER_HEAD} Calculating silence parts for linebreaks.")

    silent_parts = []
    for i, data in enumerate(midi_segments):
        if i < len(midi_segments) - 1:
            silence = max(0, midi_segments[i + 1].start - data.end)
            silent_parts.append(silence)

    return silence_threshold(silent_parts, percentile)


def create_repitched_txt_from_ultrastar_data(
        input_file: str, note_numbers: list[int], output_repitched_ultrastar: str
) -> None:
    """Creates a repitched ultrastar txt file from the original one"""
    # todo: just add '_repitched' to input_file
    print(f"{ULTRASINGER_HEAD} Creating repitched ultrastar txt -> {input_file}_repitch.txt")

    # todo: to reader
    with open(input_file, "r", encoding=FILE_ENCODING) as file:
        txt = file.readlines()

    i = 0
    # todo: just add '_repitched' to input_file
    with open(output_repitched_ultrastar, "w", encoding=FILE_ENCODING) as file:
        for line in txt:
            if line.startswith(f"{UltrastarTxtNoteTypeTag.NORMAL.value} "):
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

    score_suffix = f" | Score: total: {score.score}, notes: {score.notes} line: {score.line_bonus}, golden: {score.golden}"
    for i, line in enumerate(text):
        if line.startswith(f"#{UltrastarTxtTag.CREATOR.value}:"):
            text[i] = f"{line}{score_suffix}"
            break

        if line.startswith((
                f"{UltrastarTxtNoteTypeTag.FREESTYLE.value} ",
                f"{UltrastarTxtNoteTypeTag.NORMAL.value} ",
                f"{UltrastarTxtNoteTypeTag.GOLDEN.value} ",
                f"{UltrastarTxtNoteTypeTag.RAP.value} ",
                f"{UltrastarTxtNoteTypeTag.RAP_GOLDEN.value} ")):
            text.insert(
                i,
                f"#{UltrastarTxtTag.CREATOR.value}:UltraSinger [GitHub]{score_suffix}",
            )
            break

    text = "\n".join(text)

    with open(ultrastar_file_output, "w", encoding=FILE_ENCODING) as file:
        file.write(text)


class UltraStarWriter:
    """Docstring"""


def format_separated_string(data: str) -> str:
    temp = re.sub(r'[;/]', ',', data)
    words = temp.split(',')
    words = [s for s in words if s.strip()]

    for i, word in enumerate(words):
        if "-" not in word:
            words[i] = word.strip().capitalize() + ', '
        else:
            dash_words = word.split('-')
            capitalized_dash_words = [dash_word.strip().capitalize() for dash_word in dash_words]
            formatted_dash_word = '-'.join(capitalized_dash_words) + ', '
            words[i] = formatted_dash_word

    formatted_string = ''.join(words)

    if formatted_string.endswith(', '):
        formatted_string = formatted_string[:-2]

    return formatted_string
