"""Ultrastar TXT"""

from enum import Enum

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
    CREATOR = 'CREATOR'  # Multi-language support since v1.1.0
    COMMENT = 'COMMENT'
    VIDEO = 'VIDEO'
    FILE_END = 'E'
    LINEBREAK = '-'

    # 1.1.0
    AUDIO = 'AUDIO'  # Its instead of MP3. Just renamed
    VOCALS = 'VOCALS'  # Vocals only audio
    INSTRUMENTAL = 'INSTRUMENTAL'  # Instrumental only audio
    TAGS = 'TAGS'  # Tags for the song. Can be used for filtering

    # Unused 0.2.0
    BACKGROUND = 'BACKGROUND'  # Path to background. Is shown when there is no video. Should end with `*[BG].jpg`
    VIDEOGAP = 'VIDEOGAP'
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
    VIDEOURL = 'VIDEOURL'  # URL to the video file

    # (Unused) New in (unreleased) 2.0.0
    MEDLEYSTART = 'MEDLEYSTART'  # Rename of MEDLEYSTARTBEAT
    MEDLEYEND = 'MEDLEYEND'  # Renmame of MEDLEYENDBEAT


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
    audio = ""
    video = None
    gap = ""
    bpm = ""
    language = None
    cover = None
    vocals = None
    instrumental = None
    tags = None
    creator = "UltraSinger [GitHub]"
    comment = "UltraSinger [GitHub]"
    startBeat = []
    startTimes = []
    endTimes = []
    durations = []
    pitches = []
    words = []
    noteType = []  # F, R, G, *, :
