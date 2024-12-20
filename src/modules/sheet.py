import os.path
import re

from music21 import stream, note, duration, environment, metadata, tempo
from modules.Midi.MidiSegment import MidiSegment
from modules.console_colors import ULTRASINGER_HEAD, red_highlighted, blue_highlighted
from modules.os_helper import move
from modules.ProcessData import MediaInfo

def add_metadata_to_stream(stream, artist: str, title: str, bpm: int):
    stream.metadata = metadata.Metadata()
    stream.metadata.title = title
    stream.metadata.composer = artist
    metronome_mark = tempo.MetronomeMark(number=bpm)
    stream.insert(0, metronome_mark)


def add_midi_segments_to_stream(stream, midi_segments: list[MidiSegment]):
    for segment in midi_segments:
        # Convert the note name to a music21 note
        m21_note = note.Note(replace_unsupported_accidentals(segment.note))

        # Calculate the note's duration in quarter lengths
        note_duration = segment.end - segment.start
        note_quarter = round_to_nearest_quarter(note_duration)
        if(note_quarter == 0):
            note_quarter = 0.25

        m21_note.duration = duration.Duration(note_quarter)
        m21_note.lyrics.append(note.Lyric(text=segment.word))
        stream.append(m21_note)


def create_sheet(midi_segments: list[MidiSegment],
                 output_folder_path: str,
                 cache_folder_path: str,
                 musescore_path: str,
                 filename: str,
                 media_info: MediaInfo):
    print(f"{ULTRASINGER_HEAD} Creating music sheet with {blue_highlighted('MuseScore')}")
    success = set_environment_variables(musescore_path)
    if not success:
        return
    
    try:
        s = stream.Stream()
        add_metadata_to_stream(s, media_info.artist, media_info.title, int(media_info.bpm))
        add_midi_segments_to_stream(s, midi_segments)
        export_stream_to_pdf(s, os.path.join(output_folder_path, f"{filename}.pdf"))
        move(os.path.join(output_folder_path, f"{filename}.musicxml"), os.path.join(cache_folder_path, f"{filename}.musicxml"))
    except Exception as e:
        song_error = (f"{media_info.artist} - {media_info.title}")
        print(f"{ULTRASINGER_HEAD} {red_highlighted('Error: Could not create sheet for')} {blue_highlighted(song_error)}")
        print(f"\t{red_highlighted(f'Error: ->{e}')}")
        return

def round_to_nearest_quarter(number: float) -> float:
    return round(number * 4) / 4


def find_musescore_version_in_path(path) -> int:
    pattern = r"MuseScore\s+(\d+)"
    match = None
    try:
        for i, data in enumerate(os.listdir(path)):
            match = re.findall(pattern, data)
            if match:
                break
    except FileNotFoundError:
        return -1

    if match:
        try:
            version = int(match[0])
            return version
        except ValueError:
            # int cast error
            return -1
    else:
        # MuseScore is not found or version is unknown
        return -1

def set_environment_variables(musescorePath=None) -> bool:
    if(musescorePath is None):
        version = find_musescore_version_in_path('C:\\Program Files')
        if(version == -1):
            print(f"{ULTRASINGER_HEAD} {red_highlighted('MuseScore is not installed or version is unknown')}")
            return False
        musescorePath = f'C:\\Program Files\\MuseScore {version}\\bin\\MuseScore{version}.exe'
        print(f"{ULTRASINGER_HEAD} Using MuseScore version {version} in path {musescorePath}")
    env = environment.UserSettings()
    env['musicxmlPath'] = musescorePath
    env['musescoreDirectPNGPath'] = musescorePath
    return True

def export_stream_to_pdf(stream, pdf_path):
    print(f"{ULTRASINGER_HEAD} Creating sheet PDF -> {pdf_path}")
    stream.write('musicxml.pdf', fp=pdf_path)

def replace_unsupported_accidentals(note_name):
    accidental_replacements = {
        "♯": "#",
        "♭": "b",
    }
    for unsupported, supported in accidental_replacements.items():
        note_name = note_name.replace(unsupported, supported)
    return note_name