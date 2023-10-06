from dataclasses import dataclass

from dataclasses_json import dataclass_json

from modules.Pitcher.pitched_data import PitchedData


@dataclass_json
@dataclass
class PitchingResult:
    """Pitching result"""

    midi_notes: list[str]
    pitched_data: PitchedData
    ultrastar_note_numbers: list[int]
