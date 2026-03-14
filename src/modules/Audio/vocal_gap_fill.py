"""Fill gaps between transcribed words with placeholder notes.

When WhisperX fails to transcribe vocalizations (ad-libs, melismas,
extended vowels, humming), those time ranges produce no notes.  This
module detects gaps where SwiftF0 still reports confident pitch and
inserts placeholder ``TranscribedData`` entries so that the rest of
the pipeline creates notes for them.
"""

from __future__ import annotations

from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted
from modules.Pitcher.pitched_data import PitchedData
from modules.Pitcher.pitched_data_helper import get_frequencies_with_high_confidence
from modules.Speech_Recognition.TranscribedData import TranscribedData

# Minimum gap duration to consider (seconds)
_MIN_GAP_S = 0.15
# Minimum fraction of pitch frames in a gap that must have high confidence
_MIN_VOCAL_FRACTION = 0.3
# Confidence threshold for detecting vocal content
_CONFIDENCE_THRESHOLD = 0.5


def fill_vocal_gaps(
    transcribed_data: list[TranscribedData],
    pitched_data: PitchedData,
    min_gap_s: float = _MIN_GAP_S,
    min_vocal_fraction: float = _MIN_VOCAL_FRACTION,
    confidence_threshold: float = _CONFIDENCE_THRESHOLD,
) -> list[TranscribedData]:
    """Insert placeholder entries for un-transcribed vocal segments.

    Scans gaps between consecutive ``TranscribedData`` entries.  For each
    gap longer than *min_gap_s*, checks the SwiftF0 pitch confidence in
    that time range.  If at least *min_vocal_fraction* of the frames
    exceed *confidence_threshold*, a placeholder ``TranscribedData`` with
    ``word="~"`` is inserted.

    The returned list is sorted by start time.

    Args:
        transcribed_data: Existing transcribed segments (sorted by start).
        pitched_data: Full pitch data from SwiftF0.
        min_gap_s: Minimum gap duration to check (seconds).
        min_vocal_fraction: Minimum fraction of confident frames to fill.
        confidence_threshold: Pitch confidence threshold (0.0-1.0).

    Returns:
        Augmented list with gap-fill entries inserted.
    """
    if not transcribed_data or not pitched_data.times:
        return transcribed_data

    gaps = _find_gaps(transcribed_data, min_gap_s)
    if not gaps:
        return transcribed_data

    fill_count = 0
    new_entries: list[TranscribedData] = []

    for gap_start, gap_end in gaps:
        if _has_vocal_content(
            pitched_data, gap_start, gap_end,
            confidence_threshold, min_vocal_fraction,
        ):
            entry = TranscribedData()
            entry.word = "~ "
            entry.start = gap_start
            entry.end = gap_end
            entry.is_hyphen = False
            entry.is_word_end = True
            entry.confidence = 0.0
            new_entries.append(entry)
            fill_count += 1

    if fill_count > 0:
        print(
            f"{ULTRASINGER_HEAD} Filled {blue_highlighted(str(fill_count))} "
            f"vocal gap(s) with placeholder notes"
        )
        result = list(transcribed_data) + new_entries
        result.sort(key=lambda td: td.start)
        return result

    return transcribed_data


def _find_gaps(
    transcribed_data: list[TranscribedData],
    min_gap_s: float,
) -> list[tuple[float, float]]:
    """Find time gaps between consecutive transcribed segments.

    Returns a list of ``(gap_start, gap_end)`` pairs where the gap
    duration is at least *min_gap_s*.
    """
    gaps: list[tuple[float, float]] = []

    for i in range(len(transcribed_data) - 1):
        gap_start = transcribed_data[i].end
        gap_end = transcribed_data[i + 1].start

        if gap_end - gap_start >= min_gap_s:
            gaps.append((gap_start, gap_end))

    return gaps


def _has_vocal_content(
    pitched_data: PitchedData,
    start: float,
    end: float,
    confidence_threshold: float,
    min_fraction: float,
) -> bool:
    """Check if a time range has enough confident pitch frames.

    Returns ``True`` if at least *min_fraction* of the pitch frames
    in ``[start, end]`` have confidence above *confidence_threshold*.
    """
    # Find frame indices covering [start, end]
    times = pitched_data.times
    start_idx = _find_first_ge(times, start)
    end_idx = _find_first_ge(times, end)

    if start_idx >= end_idx or start_idx >= len(times):
        return False

    total_frames = end_idx - start_idx
    if total_frames == 0:
        return False

    confident_frames = sum(
        1 for i in range(start_idx, end_idx)
        if pitched_data.confidence[i] > confidence_threshold
    )

    return (confident_frames / total_frames) >= min_fraction


def _find_first_ge(sorted_list: list[float], value: float) -> int:
    """Binary search for the first index where ``sorted_list[i] >= value``."""
    lo, hi = 0, len(sorted_list)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_list[mid] < value:
            lo = mid + 1
        else:
            hi = mid
    return lo
