import os
import re

from packaging import version

from modules import os_helper
from modules.Midi.MidiSegment import MidiSegment
from modules.ProcessData import ProcessData, MediaInfo
from modules.Ultrastar.coverter.ultrastar_converter import ultrastar_bpm_to_real_bpm
from modules.Ultrastar.coverter.ultrastar_midi_converter import ultrastar_to_midi_segments, \
    convert_midi_notes_to_ultrastar_notes
from modules.Ultrastar.ultrastar_txt import UltrastarTxtValue, FormatVersion
from modules.Ultrastar.ultrastar_writer import create_repitched_txt_from_ultrastar_data, format_separated_string, \
    create_ultrastar_txt
from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted


def from_ultrastar_txt(ultrastar_txt: UltrastarTxtValue) -> ProcessData:
    """Converts an Ultrastar txt to ProcessData"""
    process_data = ProcessData()
    process_data.parsed_file = ultrastar_txt
    # todo: is this the real bpm? or calculate it from file?
    real_bpm = ultrastar_bpm_to_real_bpm(float(ultrastar_txt.bpm.replace(",", ".")))
    process_data.media_info = MediaInfo(
        title=ultrastar_txt.title,
        artist=ultrastar_txt.artist,
        year=ultrastar_txt.year,
        genre=ultrastar_txt.genre,
        language=ultrastar_txt.language,
        bpm=real_bpm
    )
    process_data.midi_segments = ultrastar_to_midi_segments(ultrastar_txt)

    return process_data


def create_ultrastar_txt_from_midi_segments(
        song_folder_output_path: str,
        input_file_path: str,
        title: str,
        midi_segments: list[MidiSegment],
) -> str:
    """Create Ultrastar txt from Ultrastar data"""
    output_repitched_ultrastar = os.path.join(song_folder_output_path, title + ".txt")
    ultrastar_note_numbers = convert_midi_notes_to_ultrastar_notes(midi_segments)

    create_repitched_txt_from_ultrastar_data(
        input_file_path,
        ultrastar_note_numbers,
        output_repitched_ultrastar,
    )
    return output_repitched_ultrastar


def create_ultrastar_txt_from_automation(
        basename: str,
        song_folder_output_path: str,
        midi_segments: list[MidiSegment],
        media_info: MediaInfo,
        format_version: FormatVersion,
        create_karaoke: bool,
        app_version: str,
):
    """Create Ultrastar txt from automation"""
    print(f"{ULTRASINGER_HEAD} Using UltraStar {blue_highlighted(f'Format Version {format_version.value}')}")

    ultrastar_txt = UltrastarTxtValue()
    ultrastar_txt.version = format_version.value
    ultrastar_txt.mp3 = basename + ".mp3"
    ultrastar_txt.audio = basename + ".mp3"
    ultrastar_txt.vocals = basename + " [Vocals].mp3"
    ultrastar_txt.instrumental = basename + " [Instrumental].mp3"
    ultrastar_txt.video = basename + ".mp4"
    ultrastar_txt.language = media_info.language
    cover = basename + " [CO].jpg"
    ultrastar_txt.cover = (
        cover if os_helper.check_file_exists(os.path.join(song_folder_output_path, cover)) else None
    )
    ultrastar_txt.creator = f"{ultrastar_txt.creator} {app_version}"
    ultrastar_txt.comment = f"{ultrastar_txt.comment} {app_version}"

    # Additional data
    ultrastar_txt.title = media_info.title
    ultrastar_txt.artist = media_info.artist
    if media_info.year is not None:
        ultrastar_txt.year = extract_year(media_info.year)
    if media_info.genre is not None:
        ultrastar_txt.genre = format_separated_string(media_info.genre)
    if media_info.video_url is not None:
        ultrastar_txt.videoUrl = media_info.video_url
    if media_info.cover_url is not None:
        ultrastar_txt.coverUrl = media_info.cover_url

    ultrastar_file_output_path = os.path.join(song_folder_output_path, basename + ".txt")
    create_ultrastar_txt(
        midi_segments,
        ultrastar_file_output_path,
        ultrastar_txt,
        media_info.bpm,
    )
    if create_karaoke and version.parse(format_version.value) < version.parse(FormatVersion.V1_1_0.value):
        title = basename + " [Karaoke]"
        ultrastar_txt.title = title
        ultrastar_txt.mp3 = title + ".mp3"
        karaoke_output_path = os.path.join(song_folder_output_path, title)
        karaoke_txt_output_path = karaoke_output_path + ".txt"
        create_ultrastar_txt(
            midi_segments,
            karaoke_txt_output_path,
            ultrastar_txt,
            media_info.bpm,
        )
    if version.parse(format_version.value) < version.parse(FormatVersion.V1_2_0.value):
        ultrastar_txt.videoUrl = media_info.video_url

    return ultrastar_file_output_path


# Todo: Isnt it MusicBrainz? + Sanitize when parsing UltraStar?
def extract_year(date: str) -> str:
    match = re.search(r'\b\d{4}\b', date)
    if match:
        return match.group(0)
    else:
        return date
