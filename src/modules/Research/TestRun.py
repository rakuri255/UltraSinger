import datetime

from Settings import Settings
from dataclasses import dataclass, field

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class TestedSong:
    """Tested song"""

    input_path: str
    output_path: str = ""
    success: bool = False
    input_match_ratio: float = 0.0
    output_match_ratio: float = 0.0
    input_pitch_shift_match_ratios: dict[int, float] = field(default_factory=lambda: {})
    output_pitch_shift_match_ratios: dict[int, float] = field(default_factory=lambda: {})
    no_pitch_where_should_be_pitch_ratio: float = 0.0
    pitch_where_should_be_no_pitch_ratio: float = 0.0


@dataclass_json
@dataclass
class TestRun:
    """Test run"""

    settings: Settings = None
    start_time: datetime.datetime = None
    end_time: datetime.datetime = None
    tested_songs: list[TestedSong] = field(default_factory=lambda: [])
