from dataclasses import dataclass, field
from modules.Speech_Recognition.TranscribedData import TranscribedData
from modules.Pitcher.pitched_data import PitchedData
from typing import Optional, List

@dataclass
class ProcessDataPaths:
    # Process data Paths
    processing_audio_path: Optional[str] = ""
    cache_path: Optional[str] = ""
    song_output: Optional[str] = ""
    # ultrastar_audio_input_path: Optional[str] = ""

@dataclass
class MediaInfo:
    """Media Info"""
    title: Optional[str] = None
    artist: Optional[str] = None
    bpm: Optional[float] = None
    year: Optional[str] = None
    genre: Optional[str] = None
    language: Optional[str] = None

@dataclass
class ProcessData:
    """Data for processing"""
    process_data_paths: ProcessDataPaths = ProcessDataPaths()
    basename: Optional[str] = None
    media_info: Optional[MediaInfo] = None
    transcribed_data: Optional[List[TranscribedData]] = field(default_factory=list)
    pitched_data: Optional[PitchedData] = None
