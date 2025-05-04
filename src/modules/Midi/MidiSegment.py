from dataclasses import dataclass


@dataclass
class MidiSegment:
  note: str
  start: float
  end: float
  word: str
  midi_note: int | None = None # Add field for MIDI note number
