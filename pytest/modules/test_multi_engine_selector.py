"""Unit tests for UltraSinger Multi-Engine Selectable Architecture."""
import os
import unittest
from unittest.mock import MagicMock, patch

from Settings import Settings
import UltraSinger
from modules.Speech_Recognition.Whisper import WhisperModel
from modules.Audio.Separators import (
    SeparatorFactory,
    BSRoformerSeparator,
    MelRoformerSeparator,
    HTDemucsSeparator,
    SCNetSeparator,
    DemucsSeparator,
    apply_dereverb_filter,
)
from modules.Midi.Extractors import (
    MidiExtractorFactory,
    BasicPitchMidiExtractor,
    SomeMidiExtractor,
    OmnizartMidiExtractor,
    apply_magenta_humanize,
)
from modules.Midi.MidiSegment import MidiSegment
from modules.Audio.duet_separator import separate_duet_phrases


class TestMultiEngineSelector(unittest.TestCase):
    """Test suite validating Strategy/Factory dispatch, CLI arguments, and fallback resilience."""

    def test_cli_parsing_separator_engine(self):
        """Verifies --separator-engine and --separator flags parse correctly."""
        test_args = ["-i", "dummy.mp3", "--separator-engine", "mel_roformer", "--dereverb", "uvr_dereverb"]
        parsed = UltraSinger.init_settings(test_args)
        self.assertEqual(parsed.separator_engine, "mel_roformer")
        self.assertEqual(parsed.separator, "mel_roformer")
        self.assertEqual(parsed.dereverb, "uvr_dereverb")

    def test_cli_parsing_asr_and_aligner_engine(self):
        """Verifies --asr-engine and --aligner-engine flags configure Whisper and aligner properly."""
        test_args = ["-i", "dummy.mp3", "--asr-engine", "whisper_turbo", "--aligner-engine", "mms_fa"]
        parsed = UltraSinger.init_settings(test_args)
        self.assertEqual(parsed.asr_engine, "whisper_turbo")
        self.assertEqual(parsed.whisper_model, WhisperModel.LARGE_V3_TURBO)
        self.assertEqual(parsed.aligner_engine, "mms_fa")

    def test_cli_parsing_pitch_and_midi_engine(self):
        """Verifies --pitch-engine, --midi-engine, and --magenta-humanize flags."""
        test_args = [
            "-i", "dummy.mp3",
            "--pitch-engine", "rmvpe",
            "--midi-engine", "some",
            "--magenta-humanize",
            "--duet-engine", "ecapa_embeddings",
        ]
        parsed = UltraSinger.init_settings(test_args)
        self.assertEqual(parsed.pitch_engine, "rmvpe")
        self.assertEqual(parsed.pitcher, "rmvpe")
        self.assertEqual(parsed.midi_engine, "some")
        self.assertTrue(parsed.magenta_humanize)
        self.assertEqual(parsed.duet_engine, "ecapa_embeddings")

    def test_separator_factory_instantiation(self):
        """Verifies SeparatorFactory instantiates all supported concrete separator strategies."""
        self.assertIsInstance(SeparatorFactory.get_separator("bs_roformer"), BSRoformerSeparator)
        self.assertIsInstance(SeparatorFactory.get_separator("mel_roformer"), MelRoformerSeparator)
        self.assertIsInstance(SeparatorFactory.get_separator("htdemucs"), HTDemucsSeparator)
        self.assertIsInstance(SeparatorFactory.get_separator("scnet"), SCNetSeparator)
        self.assertIsInstance(SeparatorFactory.get_separator("demucs"), DemucsSeparator)

        available = SeparatorFactory.available_separators()
        self.assertIn("bs_roformer", available)
        self.assertIn("mel_roformer", available)
        self.assertIn("htdemucs", available)
        self.assertIn("scnet", available)

    def test_midi_extractor_factory_instantiation(self):
        """Verifies MidiExtractorFactory instantiates all supported MIDI strategies."""
        self.assertIsInstance(MidiExtractorFactory.get_extractor("basic_pitch"), BasicPitchMidiExtractor)
        self.assertIsInstance(MidiExtractorFactory.get_extractor("some"), SomeMidiExtractor)
        self.assertIsInstance(MidiExtractorFactory.get_extractor("omnizart"), OmnizartMidiExtractor)

    def test_magenta_humanizer_timing_and_monotonicity(self):
        """Verifies Magenta humanization applies micro-timing while preserving monotonicity and minimum duration floor."""
        segments = [
            MidiSegment(start=1.0, end=1.5, note="C4", word="Hello "),
            MidiSegment(start=1.6, end=2.2, note="E4", word="world!"),
        ]
        humanized = apply_magenta_humanize(segments, audio_duration=10.0, seed=123)
        self.assertEqual(len(humanized), 2)
        # Check non-overlapping order
        self.assertLess(humanized[0].start, humanized[0].end)
        self.assertLessEqual(humanized[0].end, humanized[1].start)
        self.assertLess(humanized[1].start, humanized[1].end)
        # Check minimum duration floor >= 45ms
        for seg in humanized:
            self.assertGreaterEqual(seg.end - seg.start, 0.045)

    def test_dereverb_filter_none_noop(self):
        """Verifies apply_dereverb_filter with 'none' cleanly bypasses without errors."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_in, \
             tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_out:
            f_in.write(b"RIFFdummydata")
            in_path = f_in.name
            out_path = f_out.name

        try:
            res = apply_dereverb_filter(in_path, out_path, dereverb_engine="none")
            self.assertEqual(res, out_path)
            self.assertTrue(os.path.isfile(out_path))
        finally:
            if os.path.exists(in_path):
                os.remove(in_path)
            if os.path.exists(out_path):
                os.remove(out_path)

    def test_duet_separator_with_mfcc_engine(self):
        """Verifies duet phrase classification assigns player_ids under mfcc_kmeans."""
        segments = [
            MidiSegment(start=0.5, end=1.2, note="C4", word="First "),
            MidiSegment(start=1.3, end=2.0, note="D4", word="phrase "),
            MidiSegment(start=4.0, end=4.8, note="G4", word="Second "),
            MidiSegment(start=4.9, end=5.5, note="A4", word="phrase "),
        ]
        p_ids, phrases, p1, p2 = separate_duet_phrases(
            vocal_audio_path="non_existent.wav",
            midi_segments=segments,
            silence_split_duration=0.5,
            engine="mfcc_kmeans",
        )
        self.assertEqual(len(phrases), 2)
        self.assertEqual(p_ids[0], 1)  # P1 always starts first
        for seg in phrases[0]:
            self.assertEqual(seg.player_id, 1)


if __name__ == "__main__":
    unittest.main()

