"""Spotify Basic Pitch Extraction Strategy."""
import gc
import numpy as np

from modules.console_colors import (
    ULTRASINGER_HEAD,
    blue_highlighted,
    red_highlighted,
)
from modules.Pitcher.Extractors.base_pitcher import BasePitchExtractor
from modules.Pitcher.pitched_data import PitchedData


class BasicPitchExtractor(BasePitchExtractor):
    """Spotify Basic Pitch Neural Network Extractor (ICASSP 2022).

    Extracts fundamental pitch (F0) using Spotify's lightweight AMT architecture,
    converting polyphonic note activations and continuous pitch contours into PitchedData.
    """

    SAMPLE_RATE = 22050
    FFT_HOP = 256

    def extract_pitch(self, audio_path: str, device: str = "cpu") -> PitchedData:
        print(
            f"{ULTRASINGER_HEAD} Pitching with SOTA {blue_highlighted('Spotify Basic Pitch')} "
            f"(worker: {red_highlighted(device)})"
        )

        try:
            from basic_pitch.inference import predict
        except ImportError as exc:
            raise ImportError(
                "The 'basic-pitch' and 'mir-eval' packages are required for Basic Pitch detection. "
                "Install them using: uv pip install mir-eval && uv pip install --no-deps basic-pitch==0.4.0"
            ) from exc

        model_output, midi_data, note_events = predict(audio_path)

        note_matrix = model_output.get("note", None)
        hop_seconds = self.FFT_HOP / float(self.SAMPLE_RATE)

        if note_matrix is not None and len(note_matrix) > 0:
            num_frames = len(note_matrix)
            times = [float(i * hop_seconds) for i in range(num_frames)]
            frequencies = []
            confidence = []

            # Sort note_events chronologically by start time for fast linear indexing
            sorted_events = sorted(note_events, key=lambda ev: ev[0]) if note_events else []
            ev_idx = 0
            num_events = len(sorted_events)

            for i in range(num_frames):
                t = times[i]
                # Advance event pointer while event has already passed
                while ev_idx < num_events and sorted_events[ev_idx][1] < t:
                    ev_idx += 1

                active_note = None
                if ev_idx < num_events and sorted_events[ev_idx][0] <= t <= sorted_events[ev_idx][1]:
                    active_note = sorted_events[ev_idx]

                if active_note is not None:
                    midi_pitch = float(active_note[2])
                    conf = float(active_note[3])
                    freq = 440.0 * (2.0 ** ((midi_pitch - 69.0) / 12.0))
                else:
                    frame_probs = note_matrix[i]
                    best_idx = int(np.argmax(frame_probs))
                    conf = float(frame_probs[best_idx])
                    midi_pitch = float(best_idx + 21)
                    freq = 440.0 * (2.0 ** ((midi_pitch - 69.0) / 12.0))

                frequencies.append(freq)
                confidence.append(conf)
        else:
            times = []
            frequencies = []
            confidence = []

        del model_output
        del midi_data
        gc.collect()

        return PitchedData(times, frequencies, confidence)
