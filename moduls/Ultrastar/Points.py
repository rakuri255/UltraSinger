from moduls.Midi.midi_creator import create_midi_note_from_pitched_data
from moduls.Ultrastar.ultrastar_converter import get_start_time_from_ultrastar, get_end_time_from_ultrastar, ultrastar_note_to_midi_note
import librosa

# UNote :491

# if (Ini.LineBonus > 0) then
# MaxSongPoints := MAX_SONG_SCORE - MAX_SONG_LINE_BONUS
# else
# MaxSongPoints := MAX_SONG_SCORE;

# 'F': Note[HighNote].NoteType := ntFreestyle;
# ':': Note[HighNote].NoteType := ntNormal;
# '*': Note[HighNote].NoteType := ntGolden;
# 'R': Note[HighNote].NoteType := ntRap;
# 'G': Note[HighNote].NoteType := ntRapGolden;

# ntFreestyle, ntNormal, ntGolden, ntRap, ntRapGolden
# ScoreFactor = (0, 1, 2, 1, 2);

MAX_SONG_SCORE = 10000
MAX_SONG_LINE_BONUS = 1000

def calc(pitched_data, ultrastar_class, real_bpm):
    print("Calculating Ultrastar Points")
    normal_points = 0
    normal_no_pitch_points = 0
    parts_counter = 0  # parts for max points

    for i in range(len(ultrastar_class.words)):
        if ultrastar_class.words == "":
            continue

        start_time = get_start_time_from_ultrastar(ultrastar_class, i)
        end_time = get_end_time_from_ultrastar(ultrastar_class, i)
        duration = end_time - start_time
        parts_len = 0.01 #duration / real_bpm * 60
        parts = duration / parts_len

        ultrastar_midi_note = ultrastar_note_to_midi_note(int(ultrastar_class.pitches[i]))
        ultrastar_note = librosa.midi_to_note(ultrastar_midi_note)

        for p in range(int(parts)):
            st = start_time + parts_len * p
            end = st + parts_len
            pitch_note = create_midi_note_from_pitched_data(st, end, pitched_data)

            if pitch_note[:-1] == ultrastar_note[:-1]:
                normal_no_pitch_points += 1

            if pitch_note == ultrastar_note:
                normal_points += 1

            # todo: Pharsen bonus?

            parts_counter += 1

    accurate_points = MAX_SONG_SCORE * normal_points / parts_counter
    sing_points = MAX_SONG_SCORE * normal_no_pitch_points / parts_counter
    print("Points: {}, Accurate Points: {}".format(sing_points, accurate_points))
    return normal_points

