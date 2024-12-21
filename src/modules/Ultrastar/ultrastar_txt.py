"""Ultrastar TXT"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List

FILE_ENCODING = "utf-8"


class UltrastarTxtTag(str, Enum):
    """Tags for Ultrastar TXT files."""

    # 0.2.0
    VERSION = 'VERSION'  # Version of the file format: See https://usdx.eu/format/
    ARTIST = 'ARTIST'
    TITLE = 'TITLE'
    MP3 = 'MP3'  # Removed in v2.0.0
    GAP = 'GAP'
    BPM = 'BPM'
    LANGUAGE = 'LANGUAGE'  # Multi-language support since v1.1.0
    GENRE = 'GENRE'  # Multi-language support since v1.1.0
    YEAR = 'YEAR'  # Multi-language support since v1.1.0
    COVER = 'COVER'  # Path to cover. Should end with `*[CO].jpg`
    BACKGROUND = 'BACKGROUND'  # Path to background. Is shown when there is no video. Should end with `*[BG].jpg`
    CREATOR = 'CREATOR'  # Multi-language support since v1.1.0
    COMMENT = 'COMMENT'
    VIDEO = 'VIDEO'
    VIDEOGAP = 'VIDEOGAP'
    FILE_END = 'E'
    LINEBREAK = '-'

    # 1.1.0
    AUDIO = 'AUDIO'  # Its instead of MP3. Just renamed
    VOCALS = 'VOCALS'  # Vocals only audio
    INSTRUMENTAL = 'INSTRUMENTAL'  # Instrumental only audio
    TAGS = 'TAGS'  # Tags for the song. Can be used for filtering

    # 1.2.0
    VIDEOURL = 'VIDEOURL'  # URL to the video file

    # Unused 0.2.0
    EDITION = 'EDITION'  # Multi-language support since v1.1.0
    START = 'START'
    END = 'END'
    PREVIEWSTART = 'PREVIEWSTART'
    MEDLEYSTARTBEAT = 'MEDLEYSTARTBEAT'  # Removed in 2.0.0
    MEDLEYENDBEAT = 'MEDLEYENDBEAT'  # Removed in v2.0.0
    CALCMEDLEY = 'CALCMEDLEY'
    P1 = 'P1'  # Only for UltraStar Deluxe
    P2 = 'P2'  # Only for UltraStar Deluxe
    DUETSINGERP1 = 'DUETSINGERP1'  # Removed in 1.0.0 (Used by UltraStar WorldParty)
    DUETSINGERP2 = 'DUETSINGERP2'  # Removed in 1.0.0 (Used by UltraStar WorldParty)
    RESOLUTION = 'RESOLUTION'  # Changes the grid resolution of the editor. Only for the editor and nothing for singing. # Removed in 1.0.0
    NOTESGAP = 'NOTESGAP'  # Removed in 1.0.0
    RELATIVE = 'RELATIVE'  # Removed in 1.0.0
    ENCODING = 'ENCODING'  # Removed in 1.0.0

    # (Unused) 1.1.0
    PROVIDEDBY = 'PROVIDEDBY'  # Should the URL from hoster server

    # (Unused) New in (unreleased) 1.2.0
    AUDIOURL = 'AUDIOURL'  # URL to the audio file
    COVERURL = 'COVERURL'  # URL to the cover file
    BACKGROUNDURL = 'BACKGROUNDURL'  # URL to the background file

    # (Unused) New in (unreleased) 2.0.0
    MEDLEYSTART = 'MEDLEYSTART'  # Rename of MEDLEYSTARTBEAT
    MEDLEYEND = 'MEDLEYEND'  # Renmame of MEDLEYENDBEAT
    # These will forced to be in ms only. This will be an braking change from 1.1.0:
    # GAP: 4500
    # VIDEOGAP: 1200
    # START: 21100
    # END: 223250
    # MEDLEYSTART: 67050
    # MEDLEYEND: 960020
    # PREVIEWSTART: 45200


class UltrastarTxtNoteTypeTag(str, Enum):
    """Note types for Ultrastar TXT files."""
    NORMAL = ':'
    RAP = 'R'
    RAP_GOLDEN = 'G'
    FREESTYLE = 'F'
    GOLDEN = '*'


def get_note_type_from_string(note_type_str: str) -> UltrastarTxtNoteTypeTag:
    if note_type_str == UltrastarTxtNoteTypeTag.NORMAL.value:
        return UltrastarTxtNoteTypeTag.NORMAL
    elif note_type_str == UltrastarTxtNoteTypeTag.RAP.value:
        return UltrastarTxtNoteTypeTag.RAP
    elif note_type_str == UltrastarTxtNoteTypeTag.RAP_GOLDEN.value:
        return UltrastarTxtNoteTypeTag.RAP_GOLDEN
    elif note_type_str == UltrastarTxtNoteTypeTag.FREESTYLE.value:
        return UltrastarTxtNoteTypeTag.FREESTYLE
    elif note_type_str == UltrastarTxtNoteTypeTag.GOLDEN.value:
        return UltrastarTxtNoteTypeTag.GOLDEN
    else:
        raise ValueError(f"Unknown NoteType: {note_type_str}")


@dataclass
class UltrastarNoteLine:
    startBeat: float
    startTime: float
    endTime: float
    duration: float
    pitch: int
    word: str
    noteType: UltrastarTxtNoteTypeTag  # F, R, G, *, :


@dataclass
class UltrastarTxtValue:
    """Vaules for Ultrastar TXT files."""

    version = "1.1.0"
    artist = ""
    title = ""
    year = None
    genre = ""
    mp3 = ""
    audio = ""
    video = None
    videoUrl = None
    videoGap = None
    gap = ""
    bpm = ""
    language = None
    cover = None
    coverUrl = None
    background = None
    vocals = None
    instrumental = None
    tags = None
    creator = "UltraSinger [GitHub]"
    comment = "UltraSinger [GitHub]"
    UltrastarNoteLines: List[UltrastarNoteLine] = field(default_factory=list)


class FormatVersion(Enum):
    V0_2_0 = "0.2.0"
    V0_3_0 = "0.3.0"
    V1_0_0 = "1.0.0"
    V1_1_0 = "1.1.0"
    V1_2_0 = "1.2.0" # Not released yet
    V2_0_0 = "2.0.0" # Not released yet
