"""Factory to instantiate Vocal Separator strategies."""
from modules.Audio.Separators.base_separator import BaseVocalSeparator
from modules.Audio.Separators.demucs_separator import DemucsSeparator
from modules.Audio.Separators.bs_roformer_separator import BSRoformerSeparator
from modules.Audio.Separators.mel_roformer_separator import MelRoformerSeparator
from modules.Audio.Separators.scnet_separator import SCNetSeparator
from modules.Audio.Separators.htdemucs_separator import HTDemucsSeparator


class SeparatorFactory:
    """Factory for creating vocal separator instances based on strategy name."""

    _SEPARATORS = {
        "demucs": DemucsSeparator,
        "bs_roformer": BSRoformerSeparator,
        "bs-roformer": BSRoformerSeparator,
        "roformer": BSRoformerSeparator,
        "mel_roformer": MelRoformerSeparator,
        "mel-roformer": MelRoformerSeparator,
        "mel_band_roformer": MelRoformerSeparator,
        "mel-band-roformer": MelRoformerSeparator,
        "melband": MelRoformerSeparator,
        "scnet": SCNetSeparator,
        "scnet_large": SCNetSeparator,
        "htdemucs": HTDemucsSeparator,
        "htdemucs_ft": HTDemucsSeparator,
        "htdemucs_6s": HTDemucsSeparator,
        "kim-vocal": BSRoformerSeparator,
    }

    @classmethod
    def get_separator(cls, name: str) -> BaseVocalSeparator:
        key = (name or "bs_roformer").lower().strip()
        separator_class = cls._SEPARATORS.get(key)
        if not separator_class:
            valid_keys = ", ".join(cls.available_separators())
            raise ValueError(
                f"Unknown vocal separator: '{name}'. Available separators: {valid_keys}"
            )
        return separator_class()

    @classmethod
    def available_separators(cls) -> list[str]:
        return ["bs_roformer", "mel_roformer", "htdemucs", "scnet", "demucs"]

