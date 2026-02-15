"""Ultrastar txt parser"""
import os

from modules import os_helper

from modules.console_colors import ULTRASINGER_HEAD, red_highlighted
from modules.Ultrastar.coverter.ultrastar_converter import (
    get_end_time,
    get_start_time,
)
from modules.Ultrastar.ultrastar_txt import (
    UltrastarTxtValue,
    UltrastarTxtTag,
    UltrastarTxtNoteTypeTag,
    FILE_ENCODING,
    UltrastarNoteLine,
    get_note_type_from_string
)
from modules.os_helper import get_unused_song_output_dir


def parse(input_file: str) -> UltrastarTxtValue:
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
            if line.startswith(f"#{UltrastarTxtTag.ARTIST.value}"):
                ultrastar_class.artist = line.split(":")[1].replace("\n", "")
            elif line.startswith(f"#{UltrastarTxtTag.TITLE.value}"):
                ultrastar_class.title = line.split(":")[1].replace("\n", "")
            elif line.startswith(f"#{UltrastarTxtTag.MP3.value}"):
                ultrastar_class.mp3 = line.split(":")[1].replace("\n", "")
            elif line.startswith(f"#{UltrastarTxtTag.AUDIO.value}"):
                ultrastar_class.audio = line.split(":")[1].replace("\n", "")
            elif line.startswith(f"#{UltrastarTxtTag.VIDEO.value}"):
                ultrastar_class.video = line.split(":")[1].replace("\n", "")
            elif line.startswith(f"#{UltrastarTxtTag.GAP.value}"):
                ultrastar_class.gap = line.split(":")[1].replace("\n", "")
            elif line.startswith(f"#{UltrastarTxtTag.BPM.value}"):
                ultrastar_class.bpm = line.split(":")[1].replace("\n", "")
            elif line.startswith(f"#{UltrastarTxtTag.VIDEOGAP.value}"):
                ultrastar_class.videoGap = line.split(":")[1].replace("\n", "")
            elif line.startswith(f"#{UltrastarTxtTag.COVER.value}"):
                ultrastar_class.cover = line.split(":")[1].replace("\n", "")
            elif line.startswith(f"#{UltrastarTxtTag.BACKGROUND.value}"):
                ultrastar_class.background = line.split(":")[1].replace("\n", "")
        elif line.startswith(
            (
                f"{UltrastarTxtNoteTypeTag.FREESTYLE.value} ",
                f"{UltrastarTxtNoteTypeTag.NORMAL.value} ",
                f"{UltrastarTxtNoteTypeTag.GOLDEN.value} ",
                f"{UltrastarTxtNoteTypeTag.RAP.value} ",
                f"{UltrastarTxtNoteTypeTag.RAP_GOLDEN.value} ",
            )
        ):
            parts = line.split()
            # [0] F : * R G
            # [1] start beat
            # [2] duration
            # [3] pitch
            # [4] word

            ultrastar_note_line = UltrastarNoteLine(noteType=get_note_type_from_string(parts[0]),
                              startBeat=float(parts[1]),
                              duration=float(parts[2]),
                              pitch=int(parts[3]),
                              word=parts[4] if len(parts) > 4 else "",
                            startTime=get_start_time(ultrastar_class.gap, ultrastar_class.bpm, float(parts[1])),
                            endTime=get_end_time(ultrastar_class.gap, ultrastar_class.bpm, float(parts[1]), float(parts[2])))

            ultrastar_class.UltrastarNoteLines.append(ultrastar_note_line)

            # todo: Progress?

    return ultrastar_class


def parse_ultrastar_txt(input_file_path: str, output_folder_path: str) -> tuple[str, str, str, UltrastarTxtValue]:
    """Parse Ultrastar txt"""
    ultrastar_class = parse(input_file_path)

    if ultrastar_class.mp3:
        ultrastar_mp3_name = ultrastar_class.mp3
    elif ultrastar_class.audio:
        ultrastar_mp3_name = ultrastar_class.audio
    else:
        print(
            f"{ULTRASINGER_HEAD} {red_highlighted('Error!')} The provided text file does not have a reference to "
            f"an audio file."
        )
        exit(1)

    song_output = os.path.join(
        output_folder_path,
        ultrastar_class.artist.strip() + " - " + ultrastar_class.title.strip(),
    )

    # todo: get_unused_song_output_dir should be in the runner
    song_output = get_unused_song_output_dir(str(song_output))
    os_helper.create_folder(song_output)

    dirname = os.path.dirname(input_file_path)
    audio_file_path = os.path.join(dirname, ultrastar_mp3_name)
    basename_without_ext = f"{ultrastar_class.artist.strip()} - {ultrastar_class.title.strip()}"
    return (
        basename_without_ext,
        song_output,
        str(audio_file_path),
        ultrastar_class,
    )
