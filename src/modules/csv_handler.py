"""CSV export module"""

import csv

from modules.console_colors import ULTRASINGER_HEAD
from modules.Speech_Recognition.TranscribedData import TranscribedData


def export_transcribed_data_to_csv(transcribed_data: list[TranscribedData], filename: str) -> None:
    """Export transcribed data to csv"""
    print(f"{ULTRASINGER_HEAD} Exporting transcribed data to CSV")

    with open(filename, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        header = ["word", "start", "end", "confidence"]
        writer.writerow(header)
        for i, data in enumerate(transcribed_data):
            writer.writerow(
                [
                    data.word,
                    data.start,
                    data.end,
                    data.confidence,
                ]
            )


def write_lists_to_csv(times, frequencies, confidences, filename: str):
    """Write lists to csv"""
    with open(filename, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        header = ["time", "frequency", "confidence"]
        writer.writerow(header)
        for i in enumerate(times):
            pos = i[0]
            writer.writerow([times[pos], frequencies[pos], confidences[pos]])


def read_data_from_csv(filename: str):
    """Read data from csv"""
    csv_data = []
    with open(filename, "r", encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file)
        for line in csv_reader:
            csv_data.append(line)
    headless_data = csv_data[1:]
    return headless_data
