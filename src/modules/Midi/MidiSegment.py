from dataclasses import dataclass


@dataclass
class MidiSegment:
  note: str
  start: float
  end: float
  word: str
  is_word_end: bool = True
  is_line_end: bool = False
  is_freestyle: bool = False
