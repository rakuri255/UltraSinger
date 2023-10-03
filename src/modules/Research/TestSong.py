from dataclasses import dataclass

from modules.Ultrastar.ultrastar_txt import UltrastarTxtValue


@dataclass
class TestSong:
    """Test song"""

    txt: str
    audio: float
    ultrastar_class: UltrastarTxtValue