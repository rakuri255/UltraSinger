import csv

import crepe
from scipy.io import wavfile

from moduls.Log import (PRINT_ULTRASTAR, print_blue_highlighted_text,
                        print_red_highlighted_text)
from moduls.Pitcher.pitched_data import PitchedData


def get_pitch_with_crepe_file(filename, step_size, model_capacity):
    print(f"{PRINT_ULTRASTAR} Pitching with {print_blue_highlighted_text('crepe')} and model {print_blue_highlighted_text(model_capacity)} and {print_red_highlighted_text('cpu')} as worker")
    # Todo: add GPU support by using torchcrepe
    sr, audio = wavfile.read(filename)

    pitched_data = PitchedData()
    pitched_data.times, pitched_data.frequencies, pitched_data.confidence, activation = crepe.predict(audio, sr,
                                                                                                      model_capacity,
                                                                                                      step_size=step_size,
                                                                                                      viterbi=True)
    return pitched_data


def get_pitch_with_crepe(y, sr, step_size, model_capacity):
    pitched_data = PitchedData()
    pitched_data.times, pitched_data.frequencies, pitched_data.confidence, activation = crepe.predict(y, sr,
                                                                                                      model_capacity,
                                                                                                      step_size=step_size,
                                                                                                      viterbi=True)
    return pitched_data


def write_lists_to_csv(time, frequency, confidence, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ["time", "frequency", "confidence"]
        writer.writerow(header)
        for i in range(len(time)):
            writer.writerow([time[i], frequency[i], confidence[i]])


def read_data_from_csv(filename):
    csv_data = []
    with open(filename, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for line in csv_reader:
            csv_data.append(line)
    headless_data = csv_data[1:]
    return headless_data


def get_frequency_with_high_confidence(f, c, threshold=0.4):
    conf_f = []
    for i in range(len(c)):
        if c[i] > threshold:
            conf_f.append(f[i])
    if not conf_f:
        conf_f = f
    return conf_f


class Pitcher:
    pass
