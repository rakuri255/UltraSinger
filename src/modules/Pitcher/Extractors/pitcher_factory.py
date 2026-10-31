"""Factory to instantiate Pitch Extractor strategies."""
from modules.Pitcher.Extractors.base_pitcher import BasePitchExtractor
from modules.Pitcher.Extractors.swiftf0_pitcher import SwiftF0Pitcher
from modules.Pitcher.Extractors.crepe_pitcher import CrepePitcher
from modules.Pitcher.Extractors.rmvpe_pitcher import RMVPEPitcher
from modules.Pitcher.Extractors.basic_pitch_extractor import BasicPitchExtractor


class PitcherFactory:
    """Factory for creating pitch extractor instances based on strategy name."""

    _PITCHERS = {
        "swiftf0": SwiftF0Pitcher,
        "swift-f0": SwiftF0Pitcher,
        "crepe": CrepePitcher,
        "rmvpe": RMVPEPitcher,
        "basic-pitch": BasicPitchExtractor,
        "basicpitch": BasicPitchExtractor,
    }

    @classmethod
    def get_pitcher(cls, name: str, **kwargs) -> BasePitchExtractor:
        key = (name or "swiftf0").lower().strip()
        pitcher_class = cls._PITCHERS.get(key)
        if not pitcher_class:
            valid_keys = ", ".join(["swiftf0", "crepe", "rmvpe", "basic-pitch"])
            raise ValueError(
                f"Unknown pitch extractor: '{name}'. Available pitchers: {valid_keys}"
            )
        return pitcher_class(**kwargs)

    @classmethod
    def available_pitchers(cls) -> list[str]:
        return ["swiftf0", "crepe", "rmvpe", "basic-pitch"]
