"""Plot transcribed data"""
import os
from dataclasses import dataclass
from re import sub

import librosa
import numpy
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from modules.console_colors import ULTRASINGER_HEAD
from modules.Pitcher.pitched_data import PitchedData
from modules.Pitcher.pitcher import get_pitched_data_with_high_confidence
from modules.Speech_Recognition.TranscribedData import TranscribedData

@dataclass
class PlottedNote:
    """Plotted note"""

    note: str
    frequency: float
    frequency_log_10: float
    octave: int


NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
OCTAVES = [0, 1, 2, 3, 4, 5, 6, 7, 8]
X_TICK_SIZE = 5


def get_frequency_range(midi_note: str) -> float:
    """Get frequency range"""
    midi = librosa.note_to_midi(midi_note)
    frequency_range = librosa.midi_to_hz(midi + 1) - librosa.midi_to_hz(midi)
    return frequency_range


def create_plot_notes(notes: list[str], octaves: list[int]) -> list[PlottedNote]:
    """Create list of notes for plot y axis"""
    plotted_notes = []
    for octave in octaves:
        for note in notes:
            note_with_octave = note + str(octave)
            frequency = librosa.note_to_hz(note_with_octave)
            frequency_log_10 = numpy.log10([frequency])[0]
            plotted_notes.append(
                PlottedNote(note_with_octave, frequency, frequency_log_10, octave)
            )

    return plotted_notes


PLOTTED_NOTES = create_plot_notes(NOTES, OCTAVES)


def plot(
    pitched_data: PitchedData,
    output_path: str,
    transcribed_data: list[TranscribedData] = None,
    midi_notes: list[str] = None,
    title: str = None,
) -> None:
    """Plot transcribed data"""

    # determine time between to datapoints if there is no gap (this is the step size crepe ran with)
    step_size = pitched_data.times[1]
    pitched_data = get_pitched_data_with_high_confidence(pitched_data)

    if len(pitched_data.frequencies) < 2:
        print(f"{ULTRASINGER_HEAD} Plot can't be created; too few datapoints")
        return

    print(
        f"{ULTRASINGER_HEAD} Creating plot{': ' + title if title is not None else ''}"
    )

    # map each frequency to logarithm with base 10 for a linear progression of values between the musical notes
    # see http://www.phon.ox.ac.uk/jcoleman/LOGARITH.htm
    frequencies_log_10 = numpy.log10(pitched_data.frequencies)

    # add 'nan' where there are gaps for frequency values so the graph is only continuous where it should be
    pitched_data_with_gaps = create_gaps(pitched_data, step_size)
    frequencies_log_10_with_gaps = numpy.log10(pitched_data_with_gaps.frequencies)

    # dynamically set the minimum and maximum values for x and y axes based on data
    y_lower_bound, y_upper_bound = determine_bounds(frequencies_log_10)
    ymin = max(0, y_lower_bound - 0.05)
    ymax = y_upper_bound + 0.05
    plt.ylim(ymin, ymax)
    xmin = min(pitched_data.times)
    xmax = max(pitched_data.times)
    plt.xlim(xmin, xmax)

    plt.xlabel("Time (s)")
    plt.ylabel("log10 of Frequency (Hz)")

    notes_within_range = set_axes_ticks_and_labels(pitched_data.times, ymin, ymax)

    # draw horizontal lines for each note
    for note in notes_within_range:
        color = "b"
        if note.note.startswith("C") and not note.note.startswith("C#"):
            color = "r"
        plt.axhline(y=note.frequency_log_10, color=color, linestyle="-", linewidth=0.2)

    # create line and scatter plot of pitched data
    plt.plot(pitched_data_with_gaps.times, frequencies_log_10_with_gaps, linewidth=0.1)
    scatter_path_collection = plt.scatter(
        pitched_data_with_gaps.times,
        frequencies_log_10_with_gaps,
        s=5,
        c=pitched_data_with_gaps.confidence,
        cmap=plt.colormaps.get_cmap("gray").reversed(),
        vmin=0,
        vmax=1,
    )
    plt.figure(1).colorbar(scatter_path_collection, label="confidence")

    set_figure_dimensions(xmax - xmin, y_upper_bound - y_lower_bound)

    draw_words(transcribed_data, midi_notes)

    if title is not None:
        plt.title(label=title)

    plt.figure(1).tight_layout(h_pad=1.4)

    dpi = 200
    plt.savefig(
        os.path.join(
            output_path, f"plot{'' if title is None else '_' + snake(title)}.svg"
        ),
        dpi=dpi,
    )
    plt.clf()
    plt.cla()


def set_axes_ticks_and_labels(confidence, ymin, ymax):
    """Set ticks and their labels for x and y axes"""
    notes_within_range = [
        x for x in PLOTTED_NOTES if ymin <= x.frequency_log_10 <= ymax
    ]
    plt.yticks(
        [x.frequency_log_10 for x in notes_within_range],
        [x.note for x in notes_within_range],
    )

    first_time = min(confidence)
    min_tick = first_time // X_TICK_SIZE * X_TICK_SIZE + X_TICK_SIZE

    last_time = max(confidence)
    max_tick = last_time // X_TICK_SIZE * X_TICK_SIZE + 0.1
    ticks = numpy.arange(min_tick, max_tick, X_TICK_SIZE, dtype=int).tolist()

    if len(ticks) == 0 or ticks[0] != first_time:
        ticks.insert(0, first_time)

    if len(ticks) == 1 or ticks[-1] != last_time:
        ticks.append(last_time)

    plt.xticks(ticks, [str(x) for x in ticks])
    return notes_within_range


def determine_bounds(frequency_log_10: list[float]) -> tuple[float, float]:
    """Determine bounds based on 1st and 99th percentile of data"""
    lower = numpy.percentile(numpy.array(frequency_log_10), 1)
    upper = numpy.percentile(numpy.array(frequency_log_10), 99)

    return lower, upper


def set_figure_dimensions(time_range, frequency_log_10_range):
    """Dynamically scale the figure dimensions based on the duration/frequency amplitude of the song"""
    height = frequency_log_10_range / 0.06
    width = time_range / 2

    plt.figure(1).set_figwidth(max(6.4, width))
    plt.figure(1).set_figheight(max(4, height))


def create_gaps(pitched_data: PitchedData, step_size: float) -> PitchedData:
    """
    Add 'nan' where there are no high confidence frequency values.
    This way the graph is only continuous where it should be.

    """
    pitched_data_with_gaps = PitchedData([], [], [])

    previous_time = 0
    for i, time in enumerate(pitched_data.times):
        comes_right_after_previous = time - previous_time <= step_size
        previous_frequency_is_not_gap = (
            len(pitched_data_with_gaps.frequencies) > 0
            and str(pitched_data_with_gaps.frequencies[-1]) != "nan"
        )
        if previous_frequency_is_not_gap and not comes_right_after_previous:
            pitched_data_with_gaps.times.append(time)
            pitched_data_with_gaps.frequencies.append(float("nan"))
            pitched_data_with_gaps.confidence.append(pitched_data.confidence[i])

        pitched_data_with_gaps.times.append(time)
        pitched_data_with_gaps.frequencies.append(pitched_data.frequencies[i])
        pitched_data_with_gaps.confidence.append(pitched_data.confidence[i])

        previous_time = time

    return pitched_data_with_gaps


def draw_words(transcribed_data, midi_notes):
    """Draw rectangles for each word"""
    if transcribed_data is not None:
        for i, data in enumerate(transcribed_data):
            note_frequency = librosa.note_to_hz(midi_notes[i])
            frequency_range = get_frequency_range(midi_notes[i])

            half_frequency_range = frequency_range / 2
            height = (
                numpy.log10([note_frequency + half_frequency_range])[0]
                - numpy.log10([note_frequency - half_frequency_range])[0]
            )

            xy_start_pos = (
                data.start,
                numpy.log10([note_frequency - half_frequency_range])[0],
            )
            width = data.end - data.start
            rect = Rectangle(
                xy_start_pos,
                width,
                height,
                edgecolor="none",
                facecolor="red",
                alpha=0.5,
            )
            plt.gca().add_patch(rect)
            plt.text(data.start + width/4, numpy.log10([note_frequency + half_frequency_range])[0], data.word, rotation=90)


def snake(s):
    """Turn any string into a snake case string"""
    return "_".join(
        sub(
            "([A-Z][a-z]+)", r" \1", sub("([A-Z]+)", r" \1", s.replace("-", " "))
        ).split()
    ).lower()


def plot_spectrogram(audio_seperation_path: str,
                     output_path: str,
                     title: str = "Spectrogram",

                     ) -> None:
        """Plot spectrogram of data"""

        print(
            f"{ULTRASINGER_HEAD} Creating plot{': ' + title}"
        )

        audio, sr = librosa.load(audio_seperation_path, sr=None)
        powerSpectrum, frequenciesFound, time, imageAxis = plt.specgram(audio, Fs=sr)
        plt.colorbar()

        if title is not None:
            plt.title(label=title)

        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")

        ymin = 0
        ymax = max(frequenciesFound) + 0.05
        plt.ylim(ymin, ymax)
        xmin = 0
        xmax = max(time)
        plt.xlim(xmin, xmax)

        plt.figure(1).set_figwidth(max(6.4, xmax))
        plt.figure(1).set_figheight(4)

        plt.figure(1).tight_layout(h_pad=1.4)

        dpi = 200
        plt.savefig(
            os.path.join(
                output_path, f"plot{'_' + snake(title)}.svg"
            ),
            dpi=dpi,
        )
        plt.clf()
        plt.cla()