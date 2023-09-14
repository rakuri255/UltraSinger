"""Pitched data"""
from dataclasses import dataclass


@dataclass
class PitchedData:
    """Pitched data from crepe"""

    times: list[float]
    frequencies: list[float]
    confidence: list[float]
