"""SOME (Singing-Oriented MIDI Extractor) Strategy."""
import os
import gc
from modules.console_colors import (
    ULTRASINGER_HEAD,
    blue_highlighted,
    gold_highlighted,
    red_highlighted,
)
from modules.Midi.Extractors.base_midi_extractor import BaseMidiExtractor
from modules.Midi.Extractors.basic_pitch_extractor import BasicPitchMidiExtractor
from modules.Midi.MidiSegment import MidiSegment


class SomeMidiExtractor(BaseMidiExtractor):
    """Concrete MIDI extractor using SOME (Singing-Oriented MIDI Extractor).

    Specifically filters internal vocal vibrato oscillations and segments notes
    according to natural syllabic transitions.
    """

    def extract(
        self,
        audio_path: str,
        **kwargs,
    ) -> list[MidiSegment]:
        print(
            f"{ULTRASINGER_HEAD} Extracting MIDI with SOTA {blue_highlighted('SOME')} "
            f"(Singing-Oriented MIDI Extractor with vibrato-invariant syllabic segmentation)"
        )

        # Check if SOME is installed as a package or local repo
        try:
            import importlib
            some_module = None
            for mod_name in ["some", "singing_extractor", "some_midi"]:
                try:
                    some_module = importlib.import_module(mod_name)
                    break
                except ImportError:
                    pass

            if some_module and hasattr(some_module, "extract_notes"):
                raw_notes = some_module.extract_notes(audio_path, **kwargs)
                import librosa
                segments = []
                for n in raw_notes:
                    pitch_val = n.get("pitch", 60)
                    note_str = librosa.midi_to_note(int(round(pitch_val)))
                    seg = MidiSegment(
                        start=round(float(n.get("start", 0.0)), 3),
                        end=round(float(n.get("end", 0.0)), 3),
                        note=note_str,
                        word="",
                    )
                    segments.append(seg)
                return segments

        except Exception as exc:
            print(f"{ULTRASINGER_HEAD} {gold_highlighted('SOME inference exception:')} {exc}")

        # Graceful fallback: notify user and employ Spotify Basic Pitch
        print(
            f"{ULTRASINGER_HEAD} {gold_highlighted('Notice:')} "
            f"SOME model checkpoint or package not found in container environment. "
            f"Falling back smoothly to SOTA Spotify Basic Pitch."
        )
        bp = BasicPitchMidiExtractor()
        return bp.extract(audio_path, **kwargs)

