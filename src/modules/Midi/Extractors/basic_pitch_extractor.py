"""Spotify Basic Pitch MIDI Extraction Strategy."""
import os
import gc
from modules.console_colors import (
    ULTRASINGER_HEAD,
    blue_highlighted,
    red_highlighted,
    green_highlighted,
)
from modules.Midi.Extractors.base_midi_extractor import BaseMidiExtractor
from modules.Midi.MidiSegment import MidiSegment


class BasicPitchMidiExtractor(BaseMidiExtractor):
    """Concrete MIDI extractor using Spotify's Basic Pitch (neural pitch & onset tracking)."""

    def extract(
        self,
        audio_path: str,
        **kwargs,
    ) -> list[MidiSegment]:
        print(
            f"{ULTRASINGER_HEAD} Extracting MIDI with SOTA {blue_highlighted('Spotify Basic Pitch')} "
            f"(neural pitch & note event tracking)"
        )

        try:
            from basic_pitch.inference import predict
            from basic_pitch import ICASSP_2022_MODEL_PATH
            import librosa

            model_output, midi_data, note_events = predict(
                audio_path,
                onset_threshold=kwargs.get("onset_threshold", 0.5),
                frame_threshold=kwargs.get("frame_threshold", 0.3),
                minimum_note_length=kwargs.get("minimum_note_length", 58),
                minimum_frequency=kwargs.get("minimum_frequency", None),
                maximum_frequency=kwargs.get("maximum_frequency", None),
            )

            segments = []
            for start, end, pitch, velocity, pitch_bends in note_events:
                note_name = librosa.midi_to_note(int(round(pitch)))
                seg = MidiSegment(
                    start=round(float(start), 3),
                    end=round(float(end), 3),
                    note=note_name,
                    word="",
                )
                segments.append(seg)

            return segments
        except Exception as exc:
            print(
                f"{ULTRASINGER_HEAD} Basic Pitch direct extraction notice: {exc}."
            )
            return []
        finally:
            gc.collect()

