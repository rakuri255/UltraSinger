from dataclasses import dataclass, field
from modules.Speech_Recognition.TranscribedData import TranscribedData
from modules.Pitcher.pitched_data import PitchedData
from modules.Ultrastar.ultrastar_txt import UltrastarTxtValue
from modules.Ultrastar import ultrastar_converter
from modules.Midi.MidiSegment import MidiSegment
from typing import Optional, List

@dataclass
class ProcessDataPaths:
    # Process data Paths
    processing_audio_path: Optional[str] = ""
    cache_folder_path: Optional[str] = ""
    audio_output_file_path: Optional[str] = "" # Output audio file path
    vocals_audio_file_path: Optional[str] = "" # Separated vocals audio file path
    instrumental_audio_file_path: Optional[str] = "" # Separated instrumental audio file path

@dataclass
class MediaInfo:
    """Media Info"""
    title: str
    artist: str
    bpm: float
    year: Optional[str] = None
    genre: Optional[str] = None
    language: Optional[str] = None
    youtube_thumbnail_url: Optional[str] = None

@dataclass
class ProcessData:
    """Data for processing"""
    process_data_paths: ProcessDataPaths = ProcessDataPaths()
    basename: Optional[str] = None
    media_info: Optional[MediaInfo] = None
    transcribed_data: Optional[List[TranscribedData]] = field(default_factory=list)
    pitched_data: Optional[PitchedData] = None
    midi_segments: Optional[List[MidiSegment]] = field(default_factory=list)
    parsed_file: Optional[UltrastarTxtValue] = None


def from_ultrastar_txt(ultrastar_txt: UltrastarTxtValue) -> ProcessData:
    """Converts an Ultrastar txt to ProcessData"""
    process_data = ProcessData()
    process_data.parsed_file = ultrastar_txt
    # todo: is this the real bpm? or calculate it from file?
    real_bpm = ultrastar_converter.ultrastar_bpm_to_real_bpm(float(ultrastar_txt.bpm.replace(",", ".")))
    process_data.media_info = MediaInfo(
        title=ultrastar_txt.title,
        artist=ultrastar_txt.artist,
        year=ultrastar_txt.year,
        genre=ultrastar_txt.genre,
        language=ultrastar_txt.language,
        bpm=real_bpm
    )
    process_data.midi_segments = ultrastar_converter.ultrastar_to_midi_segments(ultrastar_txt)

    return process_data
