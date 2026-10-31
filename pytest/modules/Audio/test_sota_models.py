"""Unit tests for SOTA Vocal Separators and Pitch Extractors (Factory & Strategy patterns)."""
import unittest

from modules.Audio.Separators import (
    BaseVocalSeparator,
    DemucsSeparator,
    BSRoformerSeparator,
    SeparatorFactory,
)
from modules.Pitcher.Extractors import (
    BasePitchExtractor,
    SwiftF0Pitcher,
    CrepePitcher,
    RMVPEPitcher,
    PitcherFactory,
)
from modules.Pitcher.pitched_data import PitchedData


class TestSOTAFactoriesAndStrategies(unittest.TestCase):
    def test_separator_factory_demucs(self):
        sep = SeparatorFactory.get_separator("demucs")
        self.assertIsInstance(sep, BaseVocalSeparator)
        self.assertIsInstance(sep, DemucsSeparator)

    def test_separator_factory_bs_roformer(self):
        sep = SeparatorFactory.get_separator("bs-roformer")
        self.assertIsInstance(sep, BaseVocalSeparator)
        self.assertIsInstance(sep, BSRoformerSeparator)

    def test_separator_factory_invalid(self):
        with self.assertRaises(ValueError):
            SeparatorFactory.get_separator("unknown_separator")

    def test_pitcher_factory_swiftf0(self):
        pitcher = PitcherFactory.get_pitcher("swiftf0")
        self.assertIsInstance(pitcher, BasePitchExtractor)
        self.assertIsInstance(pitcher, SwiftF0Pitcher)

    def test_pitcher_factory_crepe(self):
        pitcher = PitcherFactory.get_pitcher("crepe", model_capacity="tiny")
        self.assertIsInstance(pitcher, BasePitchExtractor)
        self.assertIsInstance(pitcher, CrepePitcher)
        self.assertEqual(pitcher.model_capacity, "tiny")

    def test_pitcher_factory_rmvpe(self):
        pitcher = PitcherFactory.get_pitcher("rmvpe")
        self.assertIsInstance(pitcher, BasePitchExtractor)
        self.assertIsInstance(pitcher, RMVPEPitcher)

    def test_pitcher_factory_invalid(self):
        with self.assertRaises(ValueError):
            PitcherFactory.get_pitcher("unknown_pitcher")

    def test_pitched_data_interface_contract(self):
        times = [0.0, 0.01, 0.02]
        frequencies = [440.0, 442.0, 440.0]
        confidence = [0.95, 0.98, 0.94]
        pd = PitchedData(times, frequencies, confidence)
        self.assertEqual(len(pd.times), 3)
        self.assertEqual(len(pd.frequencies), 3)
        self.assertEqual(len(pd.confidence), 3)
        json_str = pd.to_json()
        pd_restored = PitchedData.from_json(json_str)
        self.assertEqual(pd_restored.frequencies, frequencies)

    def test_cli_settings_parsing(self):
        from UltraSinger import init_settings
        s_default = init_settings(["-i", "/input/song.mp3", "-o", "/output"])
        self.assertEqual(s_default.separator, "demucs")
        self.assertEqual(s_default.pitcher, "swiftf0")

        s_sota = init_settings([
            "-i", "/input/song.mp3",
            "-o", "/output",
            "--separator", "bs-roformer",
            "--pitcher", "crepe"
        ])
        self.assertEqual(s_sota.separator, "bs-roformer")
        self.assertEqual(s_sota.pitcher, "crepe")

        s_rmvpe = init_settings([
            "-i", "/input/song.mp3",
            "-o", "/output",
            "--pitcher", "rmvpe"
        ])
        self.assertEqual(s_rmvpe.pitcher, "rmvpe")

    def test_separator_factory_mel_band_roformer(self):
        sep = SeparatorFactory.get_separator("mel-band-roformer")
        self.assertIsInstance(sep, BaseVocalSeparator)
        self.assertIsInstance(sep, BSRoformerSeparator)

    def test_mel_band_model_resolution(self):
        sep = BSRoformerSeparator()
        self.assertEqual(
            sep.MODEL_ALIASES["mel_band_roformer"],
            "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt",
        )
        self.assertEqual(
            sep.MODEL_ALIASES["kim_vocal_2"],
            "Kim_Vocal_2.onnx",
        )

    def test_ultrastar_freestyle_and_headers_spec(self):
        from modules.Ultrastar.ultrastar_writer import create_ultrastar_txt
        from modules.Ultrastar.ultrastar_txt import UltrastarTxtValue
        from modules.Midi.MidiSegment import MidiSegment

        seg = MidiSegment(
            word="spoken_word",
            start=1.0,
            end=1.5,
            note="C4",
            is_freestyle=True,
        )

        txt_val = UltrastarTxtValue()
        txt_val.artist = "Test Artist"
        txt_val.title = "Test Title"

        import tempfile
        import os
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".txt") as tmp:
            tmp_path = tmp.name

        try:
            create_ultrastar_txt(
                midi_segments=[seg],
                ultrastar_file_output=tmp_path,
                ultrastar_class=txt_val,
                real_bpm=120.0,
            )
            with open(tmp_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("#ENCODING:UTF8", content)
            self.assertIn("#RESOLUTION:4", content)
            self.assertIn("F ", content)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
