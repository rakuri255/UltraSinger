import librosa

from moduls.Log import (
    PRINT_ULTRASTAR,
    print_blue_highlighted_text,
    print_cyan_highlighted_text,
    print_gold_highlighted_text,
    print_light_blue_highlighted_text,
    print_underlined_text,
)
from moduls.Midi.midi_creator import create_midi_note_from_pitched_data
from moduls.Ultrastar.ultrastar_converter import (
    get_end_time_from_ultrastar,
    get_start_time_from_ultrastar,
    ultrastar_note_to_midi_note,
)

MAX_SONG_SCORE = 10000
MAX_SONG_LINE_BONUS = 1000


class Points:
    notes = 0
    golden_notes = 0
    rap = 0
    golden_rap = 0
    line_bonus = 0
    parts = 0


def add_point(noteType, points):
    if noteType == ":":
        points.notes += 1
    elif noteType == "*":
        points.golden_notes += 2
    elif noteType == "R":
        points.rap += 1
    elif noteType == "G":
        points.golden_rap += 2
    return points


class Score:
    max_score = 0
    notes = 0
    golden = 0
    line_bonus = 0
    score = 0


def get_score(points):
    score = Score()
    score.max_score = (
        MAX_SONG_SCORE
        if points.line_bonus == 0
        else MAX_SONG_SCORE - MAX_SONG_LINE_BONUS
    )
    score.notes = round(score.max_score * (points.notes + points.rap) / points.parts)
    score.golden = round(points.golden_notes + points.golden_rap)
    score.score = round(score.notes + points.line_bonus + score.golden)
    score.line_bonus = round(points.line_bonus)
    return score


def print_score(score):
    print(
        f"{PRINT_ULTRASTAR} Total: {print_cyan_highlighted_text(str(score.score))}, notes: {print_blue_highlighted_text(str(score.notes))}, line bonus: {print_light_blue_highlighted_text(str(score.line_bonus))}, golden notes: {print_gold_highlighted_text(str(score.golden))}"
    )


def calculate_score(pitched_data, ultrastar_class):
    print(PRINT_ULTRASTAR + " Calculating Ultrastar Points")

    simple_points = Points()
    accurate_points = Points()

    reachable_line_bonus_per_word = MAX_SONG_LINE_BONUS / len(ultrastar_class.words)

    for i in range(len(ultrastar_class.words)):
        if ultrastar_class.words == "":
            continue

        if ultrastar_class.noteType[i] == "F":
            continue

        start_time = get_start_time_from_ultrastar(ultrastar_class, i)
        end_time = get_end_time_from_ultrastar(ultrastar_class, i)
        duration = end_time - start_time
        step_size = 0.01  # todo: should be beat length ?
        parts = int(duration / step_size)
        parts = 1 if parts == 0 else parts

        accurate_part_line_bonus_points = 0
        simple_part_line_bonus_points = 0

        ultrastar_midi_note = ultrastar_note_to_midi_note(
            int(ultrastar_class.pitches[i])
        )
        ultrastar_note = librosa.midi_to_note(ultrastar_midi_note)

        for p in range(parts):
            st = start_time + step_size * p
            end = st + step_size
            pitch_note = create_midi_note_from_pitched_data(st, end, pitched_data)

            if pitch_note[:-1] == ultrastar_note[:-1]:
                # Ignore octave high
                simple_points = add_point(ultrastar_class.noteType[i], simple_points)
                simple_part_line_bonus_points += 1

            if pitch_note == ultrastar_note:
                # Octave high must be the same
                accurate_points = add_point(
                    ultrastar_class.noteType[i], accurate_points
                )
                accurate_part_line_bonus_points += 1

            accurate_points.parts += 1
            simple_points.parts += 1

        if accurate_part_line_bonus_points >= parts:
            accurate_points.line_bonus += reachable_line_bonus_per_word

        if simple_part_line_bonus_points >= parts:
            simple_points.line_bonus += reachable_line_bonus_per_word

    return get_score(simple_points), get_score(accurate_points)


def print_score_calculation(simple_points, accurate_points):
    print(
        f"{PRINT_ULTRASTAR} {print_underlined_text('Simple (octave high ignored)')} points"
    )
    print_score(simple_points)

    print(
        f"{PRINT_ULTRASTAR} {print_underlined_text('Accurate (octave high matches)')} points:"
    )
    print_score(accurate_points)
