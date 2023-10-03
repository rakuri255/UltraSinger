"""Ultrastar TXT"""
from enum import Enum

FILE_ENCODING = "utf-8"


class UltrastarTxtTag(str, Enum):
    """Tags for Ultrastar TXT files."""

    VERSION = 'VERSION' # Version of the file format: See https://usdx.eu/format/
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

    # Unused
    BACKGROUND = 'BACKGROUND'  # Path to background. Is shown when there is no video. Should end with `*[BG].jpg`
    VIDEOGAP = 'VIDEOGAP'
    GENRE = 'GENRE'
    EDITION = 'EDITION'
    YEAR = 'YEAR'
    START = 'START'
    END = 'END'
    PREVIEWSTART = 'PREVIEWSTART'
    MEDLEYSTARTBEAT = 'MEDLEYSTARTBEAT'
    MEDLEYENDBEAT = 'MEDLEYENDBEAT'
    CALCMEDLEY = 'CALCMEDLEY'
    P1 = 'P1'  # Only for UltraStar Deluxe
    P2 = 'P2'  # Only for UltraStar Deluxe

    FIXER = 'FIXER' # Only used in UltraStar World Party

    # (Unused) deprecated since 1.0.0
    DUETSINGERP1 = 'DUETSINGERP1'
    DUETSINGERP2 = 'DUETSINGERP2'
    RESOLUTION = 'RESOLUTION'  # Changes the grid resolution of the editor. Only for the editor and nothing for singing.
    NOTESGAP = 'NOTESGAP'
    RELATIVE = 'RELATIVE'
    ENCODING = 'ENCODING'

    # (Unused) New in 1.1.0
    AUDIO = 'AUDIO' # Its instead of MP3. Just renamed
    VOCALS = 'VOCALS' # Vocals only audio
    INSTRUMENTAL = 'INSTRUMENTAL' # Instrumental only audio
    ONLINE = 'ONLINE' # URL of the song/video online
    MUSICDB = 'MUSICDB' # ID or other of an online DB like musicbrainz
    DIFFICULTY = 'DIFFICULTY' # Difficulty of the song
    MEDLEYSTART = 'MEDLEYSTART' # Rename of MEDLEYSTARTBEAT
    MEDLEYEND = 'MEDLEYEND' # Renmame of MEDLEYENDBEAT

class UltrastarTxtNoteTypeTag(str, Enum):
    """Note types for Ultrastar TXT files."""
    NORMAL = ':'
    RAP = 'R'
    RAP_GOLDEN = 'G'
    FREESTYLE = 'F'
    GOLDEN = '*'


class UltrastarTxtValue:
    """Vaules for Ultrastar TXT files."""

    version = "1.0.0"
    artist = ""
    title = ""
    year = None
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
