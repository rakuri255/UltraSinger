"""Ultrastar TXT"""
from enum import Enum

FILE_ENCODING = "utf-8"


class UltrastarTxtTag(str, Enum):
    """Tags for Ultrastar TXT files."""

    ARTIST = 'ARTIST'
    TITLE = 'TITLE'
    MP3 = 'MP3'
    GAP = 'GAP'
    BPM = 'BPM'
    LANGUAGE = 'LANGUAGE'
    COVER = 'COVER'  # Path to cover. Should end with `*[CO].jpg`
    CREATOR = 'CREATOR'
    COMMENT = 'COMMENT'
    VIDEO = 'VIDEO'
    FILE_END = 'E'
    LINEBREAK = '-'

    # Only used in UltraStar World Party
    FIXER = 'FIXER'

    # Unused
    BACKGROUND = 'BACKGROUND'  # Path to background. Is shown when there is no video. Should end with `*[BG].jpg`
    VIDEOGAP = 'VIDEOGAP'
    GENRE = 'GENRE'
    EDITION = 'EDITION'
    YEAR = 'YEAR'
    START = 'START'
    END = 'END'
    RESOLUTION = 'RESOLUTION'  # Changes the grid resolution of the editor. Only for the editor and nothing for singing.
    NOTESGAP = 'NOTESGAP'
    RELATIVE = 'RELATIVE'
    ENCODING = 'ENCODING'
    PREVIEWSTART = 'PREVIEWSTART'
    MEDLEYSTARTBEAT = 'MEDLEYSTARTBEAT'
    MEDLEYENDBEAT = 'MEDLEYENDBEAT'
    CALCMEDLEY = 'CALCMEDLEY'
    DUETSINGERP1 = 'DUETSINGERP1'
    DUETSINGERP2 = 'DUETSINGERP2'
    P1 = 'P1'  # Only for UltraStar Deluxe
    P2 = 'P2'  # Only for UltraStar Deluxe


class UltrastarTxtNoteTypeTag(str, Enum):
    """Note types for Ultrastar TXT files."""
    NORMAL = ':'
    RAP = 'R'
    RAP_GOLDEN = 'G'
    FREESTYLE = 'F'
    GOLDEN = '*'


class UltrastarTxtValue:
    """Vaules for Ultrastar TXT files."""

    artist = ""
    title = ""
    year = ""
    genre = ""
    mp3 = ""
    video = None
    gap = ""
    bpm = ""
    language = None
    cover = None
    creator = "UltraSinger [GitHub]"
    comment = "UltraSinger [GitHub]"
    startBeat = []
    startTimes = []
    endTimes = []
    durations = []
    pitches = []
    words = []
    noteType = []  # F, R, G, *, :

    # Only used in UltraStar World Party
    fixer = "YOUR NAME"
