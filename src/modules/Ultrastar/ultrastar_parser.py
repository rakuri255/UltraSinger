"""Ultrastar txt parser"""

from modules.console_colors import ULTRASINGER_HEAD
from modules.Ultrastar.ultrastar_converter import (
    get_end_time_from_ultrastar,
    get_start_time_from_ultrastar,
)
from modules.Ultrastar.ultrastar_txt import UltrastarTxtValue, UltrastarTxtTag, UltrastarTxtNoteTypeTag, FILE_ENCODING

def parse_ultrastar_txt(input_file: str) -> UltrastarTxtValue:
    """Parse ultrastar txt file to UltrastarTxt class"""
    print(f"{ULTRASINGER_HEAD} Parse ultrastar txt -> {input_file}")

    with open(input_file, "r", encoding=FILE_ENCODING) as file:
        txt = file.readlines()

    ultrastar_class = UltrastarTxtValue()
    count = 0

    # Strips the newline character
    for line in txt:
        count += 1
        if line.startswith("#"):
            if line.startswith(f"#{UltrastarTxtTag.ARTIST}"):
                ultrastar_class.artist = line.split(":")[1].replace("\n", "")
            elif line.startswith(f"#{UltrastarTxtTag.TITLE}"):
                ultrastar_class.title = line.split(":")[1].replace("\n", "")
            elif line.startswith(f"#{UltrastarTxtTag.MP3}"):
                ultrastar_class.mp3 = line.split(":")[1].replace("\n", "")
            elif line.startswith(f"#{UltrastarTxtTag.AUDIO}"):
                ultrastar_class.audio = line.split(":")[1].replace("\n", "")
            elif line.startswith(f"#{UltrastarTxtTag.VIDEO}"):
                ultrastar_class.video = line.split(":")[1].replace("\n", "")
            elif line.startswith(f"#{UltrastarTxtTag.GAP}"):
                ultrastar_class.gap = line.split(":")[1].replace("\n", "")
            elif line.startswith(f"#{UltrastarTxtTag.BPM}"):
                ultrastar_class.bpm = line.split(":")[1].replace("\n", "")
        elif line.startswith((
                f"{UltrastarTxtNoteTypeTag.FREESTYLE} ",
                f"{UltrastarTxtNoteTypeTag.NORMAL} ",
                f"{UltrastarTxtNoteTypeTag.GOLDEN} ",
                f"{UltrastarTxtNoteTypeTag.RAP} ",
                f"{UltrastarTxtNoteTypeTag.RAP_GOLDEN} ")):
            parts = line.split()
            # [0] F : * R G
            # [1] start beat
            # [2] duration
            # [3] pitch
            # [4] word

            ultrastar_class.noteType.append(parts[0])
            ultrastar_class.startBeat.append(parts[1])
            ultrastar_class.durations.append(parts[2])
            ultrastar_class.pitches.append(parts[3])
            ultrastar_class.words.append(parts[4] if len(parts) > 4 else "")

            # do always as last
            pos = len(ultrastar_class.startBeat) - 1
            ultrastar_class.startTimes.append(
                get_start_time_from_ultrastar(ultrastar_class, pos)
            )
            ultrastar_class.endTimes.append(
                get_end_time_from_ultrastar(ultrastar_class, pos)
            )
            # todo: Progress?

    return ultrastar_class
