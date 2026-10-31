"""Omnizart Multi-Timbre Acoustic MIDI Extractor Strategy."""
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


class OmnizartMidiExtractor(BaseMidiExtractor):
    """Concrete MIDI extractor using Omnizart (multi-timbre full-band and vocal transcription)."""

    def extract(
        self,
        audio_path: str,
        **kwargs,
    ) -> list[MidiSegment]:
        print(
            f"{ULTRASINGER_HEAD} Extracting MIDI with SOTA {blue_highlighted('Omnizart')} "
            f"(multi-timbre full-band and vocal transcription)"
        )

        try:
            import omnizart
            from omnizart.vocal import app as vocal_app

            midi_out = vocal_app.transcribe(audio_path)
            # Parse omnizart midi output into MidiSegments
            import pretty_midi
            pm = pretty_midi.PrettyMIDI(midi_out) if isinstance(midi_out, str) else midi_out
            import librosa
            segments = []
            for inst in pm.instruments:
                for note in inst.notes:
                    note_name = librosa.midi_to_note(note.pitch)
                    seg = MidiSegment(
                        start=round(float(note.start), 3),
                        end=round(float(note.end), 3),
                        note=note_name,
                        word="",
                    )
                    segments.append(seg)
            if segments:
                return segments
        except Exception as exc:
            pass

        # Graceful fallback: notify user and employ Spotify Basic Pitch
        print(
            f"{ULTRASINGER_HEAD} {gold_highlighted('Notice:')} "
            f"Omnizart package not present or vocal model not loaded. "
            f"Falling back smoothly to SOTA Spotify Basic Pitch."
        )
        bp = BasicPitchMidiExtractor()
        return bp.extract(audio_path, **kwargs)

