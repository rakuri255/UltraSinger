from modules.Pitcher import pitched_data
import librosa

def merge_pitch_data(datas: list[pitched_data.PitchedData], step_size: float = 0.1) -> pitched_data.PitchedData:
  merged_data = pitched_data.PitchedData()
  merged_data.times = []
  merged_data.frequencies = []
  merged_data.confidence = []
  
  offset = float(0)
  for data in datas:
    if (len(data.times) > 0):
      merged_data.times.extend(map(apply_offset(offset), data.times))
      merged_data.frequencies.extend(data.frequencies)
      merged_data.confidence.extend(data.frequencies)
      offset = merged_data.times[-1] + step_size

  return merged_data

def apply_offset(offset: float) -> pitched_data.PitchedData:
  def with_offset(x: float):
    return x + offset
  
  return with_offset

def flat_pitch(duration: float, frequency: float, step_size: float = 0.1,  confidence: float = 1) -> pitched_data.PitchedData:
  times = []
  frequencies = []
  confidences = []

  step_size_decimal_points = len(str(step_size).split(".")[1])
  steps = int(duration / step_size) + 1

  for step in range(int(steps)):
    times.append(round(step * step_size, step_size_decimal_points))
    frequencies.append(frequency)
    confidences.append(confidence)

  data = pitched_data.PitchedData()
  data.times = times
  data.frequencies = frequencies
  data.confidence = confidences
  
  return data

def steady_pitch_change(duration: float, start_frequency: float, end_frequency: float, step_size: float = 0.1,  confidence: float = 1) -> pitched_data.PitchedData:
  times = []
  frequencies = []
  confidences = []

  step_size_decimal_points = len(str(step_size).split(".")[1])
  
  steps = int(duration / step_size) + 1

  frequency_step_size = abs(start_frequency - end_frequency) / (steps - 1)
  # does the frequency increase or decrease?
  frequency_shift_factor = -1 if start_frequency > end_frequency else 1

  for step in range(int(steps)):
    times.append(round(step * step_size, step_size_decimal_points))
    frequencies.append(start_frequency + (frequency_shift_factor * round(step * frequency_step_size, 1)))
    confidences.append(confidence)

  data = pitched_data.PitchedData()
  data.times = times
  data.frequencies = frequencies
  data.confidence = confidences
  
  return data

def expect_midi(note: str, start_time: float, duration: float, step_size: float = 0.01) -> tuple[int, float, float]:
  return note, start_time, start_time + duration - step_size

def merge_expected_midis_to_expected_output(expected_midis: list[tuple[str, float, float]]) -> tuple[list[str], list[float], list[float]]:
  notes = []
  start_times = []
  end_times = []

  for expected_midi in expected_midis:
    notes.append(expected_midi[0])
    start_times.append(expected_midi[1])
    end_times.append(expected_midi[2])

  return notes, start_times, end_times

# @dataclass
# class ScoringResult:
#   score: float
#   relative_match: float
#   relative_miss: float
#   absolute_match: int
#   possible_match: int
#   absolute_miss: int
#   possible_miss: int

# def scoring(ideal_notes, actual_notes) -> ScoringResult:
#   scoring_step_size = 0.05

#   last_compared_index = 0
#   for index, ideal in enumerate(ideal_notes):
#     start = ideal[1]
#     end = ideal[2]
#     between_start_and_end = find_overlapping_notes(start, end, actual_notes[last_compared_index:])

# def find_overlapping_notes(start: float, end: float, to_compare: tuple[list[str], list[float], list[float]]) -> tuple[list[str], list[float], list[float]]