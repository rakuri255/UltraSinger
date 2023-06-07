"""CSV export module"""

import csv

from modules.console_colors import ULTRASINGER_HEAD


def export_transcribed_data_to_csv(transcribed_data: [], filename: str) -> None:
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
                    data.conf,
                ]
            )
