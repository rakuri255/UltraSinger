from moduls.Midi.midi_creator import create_midi_note_from_pitched_data
from moduls.Ultrastar.ultrastar_converter import get_start_time_from_ultrastar, get_end_time_from_ultrastar, \
    ultrastar_note_to_midi_note
from moduls.Log import PRINT_ULTRASTAR, print_blue_highlighted_text, print_gold_highlighted_text, print_light_blue_highlighted_text, print_underlined_text, print_cyan_highlighted_text
import librosa

# Todo: LineBonus
# if (Ini.LineBonus > 0) then
# MaxSongPoints := MAX_SONG_SCORE - MAX_SONG_LINE_BONUS
# else
# MaxSongPoints := MAX_SONG_SCORE;

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
    if noteType == ':':
        points.notes += 1
    elif noteType == '*':
        points.golden_notes += 2
    elif noteType == 'R':
        points.rap += 1
    elif noteType == 'G':
        points.golden_rap += 2
    return points


def print_score(points):
    notes = MAX_SONG_SCORE * (points.notes + points.rap) / points.parts
    line_bonus = 0
    golden = points.golden_notes + points.golden_rap
    score = notes + line_bonus + golden
    print(PRINT_ULTRASTAR + " Total: {}, notes: {}, line bonus: {}, golden notes: {}".format(print_cyan_highlighted_text(str(round(score))), print_blue_highlighted_text(str(round(notes))), print_light_blue_highlighted_text(str(round(line_bonus))), print_gold_highlighted_text(str(round(golden)))))


def print_score_calculation(pitched_data, ultrastar_class):
    print(PRINT_ULTRASTAR + " Calculating Ultrastar Points")

    simple_points = Points()
    accurate_points = Points()

    for i in range(len(ultrastar_class.words)):
        if ultrastar_class.words == "":
            continue

        if ultrastar_class.noteType[i] == 'F':
            continue

        start_time = get_start_time_from_ultrastar(ultrastar_class, i)
        end_time = get_end_time_from_ultrastar(ultrastar_class, i)
        duration = end_time - start_time
        parts_len = 0.01
        parts = duration / parts_len

        ultrastar_midi_note = ultrastar_note_to_midi_note(int(ultrastar_class.pitches[i]))
        ultrastar_note = librosa.midi_to_note(ultrastar_midi_note)

        for p in range(int(parts)):
            st = start_time + parts_len * p
            end = st + parts_len
            pitch_note = create_midi_note_from_pitched_data(st, end, pitched_data)

            if pitch_note[:-1] == ultrastar_note[:-1]:
                accurate_points = add_point(ultrastar_class.noteType[i], accurate_points)

            if pitch_note == ultrastar_note:
                simple_points = add_point(ultrastar_class.noteType[i], simple_points)

            accurate_points.parts += 1
            simple_points.parts += 1

    # todo: line_bonus

    print("{} {} points".format(PRINT_ULTRASTAR, print_underlined_text("Simple")))
    print_score(simple_points)

    print("{} {} points:".format(PRINT_ULTRASTAR, print_underlined_text("Accurate")))
    print_score(accurate_points)
