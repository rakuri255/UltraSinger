import csv
import crepe
from scipy.io import wavfile


def get_pitch_with_crepe_file(filename, step_size):
    sr, audio = wavfile.read(filename)

    pitch_time, pitch_frequency, pitch_confidence, activation = crepe.predict(audio, sr, 'tiny',
                                                                              step_size=step_size,
                                                                              viterbi=True)

    return pitch_time, pitch_frequency, pitch_confidence


def get_pitch_with_crepe(y, sr, step_size):
    pitch_time, pitch_frequency, pitch_confidence, activation = crepe.predict(y, sr, 'tiny', step_size=step_size,
                                                                              viterbi=True)
    return pitch_time, pitch_frequency, pitch_confidence


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


def get_frequency_with_high_confidance(f, c, threshold=0.4):
    conf_f = []
    for i in range(len(c)):
        if c[i] > threshold:
            conf_f.append(f[i])
    if not conf_f:
        conf_f = f
    return conf_f


class Pitcher:
    pass
