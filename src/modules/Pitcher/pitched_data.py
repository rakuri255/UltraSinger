"""Pitched data"""
from dataclasses import dataclass

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class PitchedData:
    """Pitched data from pitch detection (SwiftF0)"""

    times: list[float]
    frequencies: list[float]
    confidence: list[float]
