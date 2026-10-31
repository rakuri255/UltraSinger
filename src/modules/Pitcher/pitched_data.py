"""Pitched data"""
from dataclasses import dataclass

try:
    from dataclasses_json import dataclass_json
except Exception:
    def dataclass_json(cls):
        return cls


@dataclass_json
@dataclass
class PitchedData:
    """Pitched data from crepe"""

    times: list[float]
    frequencies: list[float]
    confidence: list[float]
