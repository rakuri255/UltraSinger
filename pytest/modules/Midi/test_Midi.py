"""Tests for midi_creator.py"""

import unittest
import src.modules.Midi.midi_creator as test_subject
from src.modules.Speech_Recognition.TranscribedData import TranscribedData
from src.modules.Speech_Recognition.Whisper import convert_to_transcribed_data
from helper import *
import librosa

def n(note: str) -> float:
    return librosa.note_to_hz(note)
def m(note: str) -> float:
    return librosa.note_to_midi(note)

class MidiCreatorTest(unittest.TestCase):
    def test_create_midi_notes_from_pitched_data(self):
        
        # Arrange
        pitch_data = []
        pitch_data.append(flat_pitch(5, n('A2')))
        pitch_data.append(steady_pitch_change(1, n('A2'), n('B2')))
        pitch_data.append(flat_pitch(5, n('B2')))
        pitch_data.append(steady_pitch_change(1, n('B2'), n('A2')))
        pitch_data.append(flat_pitch(5, n('A2')))

        merged_pitch_data = merge_pitch_data(pitch_data)

        ideal_midis = []
        ideal_midis.append(expect_midi('A2', 0, 5))
        ideal_midis.append(expect_midi('B2', 6, 5))
        ideal_midis.append(expect_midi('A2', 12, 5))
        
        ideal_result = merge_expected_midis_to_expected_output(ideal_midis)

        # Act
        result = test_subject.create_midi_note_from_pitched_data(0, merged_pitch_data.times[-1], merged_pitch_data)

        # Assert
        print(merged_pitch_data)
        print(ideal_result)
        print(result)
        print("end")

if __name__ == "__main__":
    unittest.main()
