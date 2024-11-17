"""Ultrastar Converter"""
from typing import Tuple
import librosa
from modules.Ultrastar.ultrastar_txt import UltrastarTxtValue
from modules.Ultrastar.ultrastar_txt import UltrastarTxtNoteTypeTag
from modules.Midi.MidiSegment import MidiSegment
import numpy

NO_PITCH = -1000
FREESTYLE = -1001


def real_bpm_to_ultrastar_bpm(real_bpm: float) -> float:
    """Converts real BPM to UltraStar BPM"""
    # The UltraStar BPM info is a fourth beat of the real BPM
    ultrastar_bpm = real_bpm / 4
    return ultrastar_bpm


def ultrastar_bpm_to_real_bpm(ultrastar_bpm: float) -> float:
    """Converts UltraStar BPM to real BPM"""
    # The UltraStar BPM info is a fourth beat of the real BPM
    bpm = ultrastar_bpm * 4
    return bpm


def second_to_beat(seconds: float, real_bpm: float) -> float:
    """Converts seconds to beat"""
    # BPM = 60 * beat / T
    # T * BPM = 60 * beat
    # beat = T * BPM / 60
    beat = seconds * real_bpm / 60
    return beat


def beat_to_second(beat: float, real_bpm: float) -> float:
    """Converts beat to seconds"""

    seconds = beat * 60 / real_bpm
    return seconds


def midi_note_to_ultrastar_note(midi_note: int) -> int:
    """Converts Midi note to UltraStar note"""

    # C4 == 48 in Midi
    ultrastar_note = midi_note - 48
    return ultrastar_note


def ultrastar_note_to_midi_note(ultrastar_note: int) -> int:
    """Converts UltraStar note to Midi note"""

    # C4 == 48 in Midi
    midi_note = ultrastar_note + 48
    return midi_note


def get_start_time_from_ultrastar(ultrastar_class: UltrastarTxtValue, pos: int) -> float:
    """Calculates the start time from the Ultrastar txt"""

    start_time = get_start_time(ultrastar_class.gap, ultrastar_class.bpm,
                                ultrastar_class.UltrastarNoteLines[pos].startBeat)
    return start_time


def get_start_time(gap: str, ultrastar_bpm: str, startBeat: float) -> float:
    """Calculates the start time from the Ultrastar txt"""

    gap = __convert_gap(gap)
    real_bpm = __convert_bpm(ultrastar_bpm)
    start_time = beat_to_second(int(startBeat), real_bpm) + gap
    return start_time


def get_end_time_from_ultrastar(ultrastar_class: UltrastarTxtValue, pos: int) -> float:
    """Calculates the end time from the Ultrastar txt"""

    end_time = get_end_time(ultrastar_class.gap, ultrastar_class.bpm, ultrastar_class.UltrastarNoteLines[pos].startBeat,
                            ultrastar_class.UltrastarNoteLines[pos].duration)
    return end_time


def get_end_time(gap: str, ultrastar_bpm: str, startBeat: float, duration: float) -> float:
    """Calculates the end time from the Ultrastar txt"""

    gap = __convert_gap(gap)
    real_bpm = __convert_bpm(ultrastar_bpm)
    end_time = (
            beat_to_second(
                int(startBeat) + int(duration),
                real_bpm,
            )
            + gap
    )
    return end_time


def __convert_gap(gap: str) -> float:
    gap = float(gap.replace(",", ".")) / 1000
    return gap


def __convert_bpm(ultrastar_bpm: str) -> float:
    real_bpm = ultrastar_bpm_to_real_bpm(float(ultrastar_bpm.replace(",", ".")))
    return real_bpm


def ultrastar_to_midi_segments(ultrastar_txt: UltrastarTxtValue) -> list[MidiSegment]:
    """Converts an Ultrastar txt to Midi segments"""
    midi_segments = []
    for i, data in enumerate(ultrastar_txt.UltrastarNoteLines):
        start_time = get_start_time_from_ultrastar(ultrastar_txt, i)
        end_time = get_end_time_from_ultrastar(ultrastar_txt, i)
        midi_segments.append(
            MidiSegment(librosa.midi_to_note(ultrastar_note_to_midi_note(data.pitch)),
                        start_time,
                        end_time,
                        data.word,
                        )
        )
    return midi_segments


def map_to_datapoints(
    ultrastar_class: UltrastarTxtValue, step_size: int = 10
) -> list[int]:
    gap = float(ultrastar_class.gap.replace(",", "."))

    data = []

    previous_step = -step_size
    for pos, note_line in enumerate(ultrastar_class.UltrastarNoteLines):
        # TODO: does this make sense?
        if note_line.noteType == UltrastarTxtNoteTypeTag.FREESTYLE:
            continue

        start_time = int(get_start_time_from_ultrastar(ultrastar_class, pos) * 1000 + gap)
        end_time = int(get_end_time_from_ultrastar(ultrastar_class, pos) * 1000 + gap)

        start_nearest_step = (start_time + step_size - 1) // step_size * step_size
        end_nearest_step = (end_time + step_size - 1) // step_size * step_size

        if previous_step == start_nearest_step:
            start_nearest_step += step_size

        duration = end_nearest_step - start_nearest_step

        if duration < 10:
            continue

        # pad gaps between pitches with empty datapoints
        gap_steps_count = (start_nearest_step - previous_step - step_size) // step_size
        data += [NO_PITCH] * gap_steps_count

        pitch_steps_count = duration // step_size

        if note_line.noteType == UltrastarTxtNoteTypeTag.FREESTYLE:
            data += [FREESTYLE] * pitch_steps_count
        else:
            data += [int(note_line.pitch)] * pitch_steps_count

        previous_step = end_nearest_step

    return data


def compare_pitches(input_ultrastar_class, output_ultrastar_class) -> tuple[float, float, dict[int, float], dict[int, float], float, float]:
    step_size = 10

    input_datapoints = map_to_datapoints(input_ultrastar_class, step_size)
    output_datapoints = map_to_datapoints(output_ultrastar_class, step_size)

    longest = max(len(input_datapoints), len(output_datapoints))
    for datapoints in [input_datapoints, output_datapoints]:
        length = len(datapoints)
        if length < longest:
            gap_steps_count = longest - length
            # pad gaps between pitches with empty datapoints
            datapoints += [NO_PITCH] * gap_steps_count

    input_pitched_datapoints = len([x for x in input_datapoints if x != NO_PITCH])
    output_pitched_datapoints = len([x for x in output_datapoints if x != NO_PITCH])

    matches = 0
    pitch_shift_matches = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    pitch_where_should_be_no_pitch = 0
    no_pitch_where_should_be_pitch = 0
    for index, _ in enumerate(input_datapoints):
        input_pitch = input_datapoints[index]
        output_pitch = output_datapoints[index]
        if input_pitch == NO_PITCH and output_pitch == NO_PITCH:
            continue

        if input_pitch == output_pitch or (input_pitch == FREESTYLE and output_pitch != NO_PITCH):
            matches += 1
        elif input_pitch == NO_PITCH:
            pitch_where_should_be_no_pitch += 1
        elif output_pitch == NO_PITCH:
            no_pitch_where_should_be_pitch += 1
        else:
            _, input_pitch_remainder = divmod(input_pitch, 12)
            _, output_pitch_remainder = divmod(output_pitch, 12)
            pitch_difference = abs(input_pitch_remainder - output_pitch_remainder)
            pitch_shift_matches[pitch_difference] += 1

    input_match_ratio = matches / input_pitched_datapoints
    output_match_ratio = matches / output_pitched_datapoints

    input_pitch_shift_match_ratios = {}
    output_pitch_shift_match_ratios = {}
    for index, pitch_shift_matches_item in enumerate(pitch_shift_matches):
        pitch_shift_matches_count = pitch_shift_matches_item
        if index == 0:
            pitch_shift_matches_count += matches
        input_pitch_shift_match_ratios[index] = pitch_shift_matches_item / input_pitched_datapoints
        output_pitch_shift_match_ratios[index] = pitch_shift_matches_item / output_pitched_datapoints

    output_pitch_where_should_be_no_pitch_ratio = pitch_where_should_be_no_pitch / output_pitched_datapoints
    output_no_pitch_where_should_be_pitch_ratio = no_pitch_where_should_be_pitch / input_pitched_datapoints

    return (input_match_ratio,
            output_match_ratio,
            input_pitch_shift_match_ratios,
            output_pitch_shift_match_ratios,
            output_pitch_where_should_be_no_pitch_ratio,
            output_no_pitch_where_should_be_pitch_ratio
            )


def determine_nearest_end_step(input_ultrastar_class, step_size) -> int:
    pitches_count = len(input_ultrastar_class.pitches) - 1
    end_time = int(
        get_end_time_from_ultrastar(input_ultrastar_class, pitches_count) * 1000
    ) + int(input_ultrastar_class.gap)
    return (end_time + step_size - 1) // step_size * step_size
