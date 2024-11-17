from dataclasses import dataclass

from modules.Ultrastar.ultrastar_txt import UltrastarTxtValue


@dataclass
class TestSong:
    """Test song"""

    input_txt: str
    input_folder: str
    input_ultrastar_class: UltrastarTxtValue
