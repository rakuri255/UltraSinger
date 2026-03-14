"""Tests for onset_correction.py — onset-based timing correction."""

import unittest

import numpy as np

from src.modules.Speech_Recognition.TranscribedData import TranscribedData
from src.modules.Audio.onset_correction import snap_to_onsets


class TestSnapToOnsets(unittest.TestCase):
    """Tests for snap_to_onsets()."""

    @staticmethod
    def _make_data(start: float, end: float, word: str = "x ") -> TranscribedData:
        data = TranscribedData()
        data.start = start
        data.end = end
        data.word = word
        return data

    # -- edge cases -----------------------------------------------------------

    def test_empty_data_returns_empty(self):
        result = snap_to_onsets([], np.array([1.0, 2.0]))
        self.assertEqual(result, [])

    def test_empty_onsets_returns_unchanged(self):
        data = [self._make_data(1.0, 2.0)]
        result = snap_to_onsets(data, np.array([]))
        self.assertAlmostEqual(result[0].start, 1.0)

    # -- snapping to nearby onset ---------------------------------------------

    def test_snap_to_earlier_onset(self):
        """Note at 1.05s, onset at 1.00s → snap to 1.00s (50ms < 80ms)."""
        data = [self._make_data(1.05, 2.0)]
        onsets = np.array([1.0])
        result = snap_to_onsets(data, onsets, max_snap_ms=80.0)
        self.assertAlmostEqual(result[0].start, 1.0)

    def test_snap_to_later_onset(self):
        """Note at 1.00s, onset at 1.06s → snap to 1.06s (60ms < 80ms)."""
        data = [self._make_data(1.0, 2.0)]
        onsets = np.array([1.06])
        result = snap_to_onsets(data, onsets, max_snap_ms=80.0)
        self.assertAlmostEqual(result[0].start, 1.06)

    def test_snap_picks_nearest_onset(self):
        """When two onsets are within range, snap to the nearest one."""
        data = [self._make_data(1.05, 2.0)]
        onsets = np.array([0.98, 1.02])  # 70ms before, 30ms before
        result = snap_to_onsets(data, onsets, max_snap_ms=80.0)
        self.assertAlmostEqual(result[0].start, 1.02)

    # -- no snap when too far -------------------------------------------------

    def test_no_snap_when_too_far(self):
        """Note at 1.15s, onset at 1.00s → 150ms > 80ms → no snap."""
        data = [self._make_data(1.15, 2.0)]
        onsets = np.array([1.0])
        result = snap_to_onsets(data, onsets, max_snap_ms=80.0)
        self.assertAlmostEqual(result[0].start, 1.15)

    def test_just_within_threshold(self):
        """Distance just under max_snap_ms → should snap."""
        data = [self._make_data(1.079, 2.0)]
        onsets = np.array([1.0])
        result = snap_to_onsets(data, onsets, max_snap_ms=80.0)
        self.assertAlmostEqual(result[0].start, 1.0)

    # -- safety: don't create zero-duration notes -----------------------------

    def test_no_snap_when_would_exceed_end(self):
        """Onset at 1.10s, note end at 1.10s → would create 0ms note → no snap."""
        data = [self._make_data(1.05, 1.10)]
        onsets = np.array([1.10])
        result = snap_to_onsets(data, onsets, max_snap_ms=80.0)
        # Should NOT snap because 1.10 is not < 1.10 - 0.01
        self.assertAlmostEqual(result[0].start, 1.05)

    def test_snap_preserves_minimum_duration(self):
        """Snap should leave at least 10ms of duration."""
        data = [self._make_data(1.00, 1.05)]
        onsets = np.array([1.03])
        result = snap_to_onsets(data, onsets, max_snap_ms=80.0)
        # 1.03 < 1.05 - 0.01 = 1.04 → OK to snap
        self.assertAlmostEqual(result[0].start, 1.03)

    # -- multiple notes -------------------------------------------------------

    def test_multiple_notes_independent(self):
        """Each note snaps independently to its nearest onset."""
        data = [
            self._make_data(1.05, 1.5, "first "),
            self._make_data(2.03, 2.5, "second "),
            self._make_data(3.15, 3.5, "third "),  # too far from any onset
        ]
        onsets = np.array([1.0, 2.0])
        result = snap_to_onsets(data, onsets, max_snap_ms=80.0)
        self.assertAlmostEqual(result[0].start, 1.0)   # snapped
        self.assertAlmostEqual(result[1].start, 2.0)   # snapped
        self.assertAlmostEqual(result[2].start, 3.15)   # too far, unchanged

    # -- preserves word and end time ------------------------------------------

    def test_preserves_word_and_end(self):
        """Snapping should only modify start, not word or end."""
        data = [self._make_data(1.05, 2.0, "hello ")]
        onsets = np.array([1.0])
        result = snap_to_onsets(data, onsets, max_snap_ms=80.0)
        self.assertEqual(result[0].word, "hello ")
        self.assertAlmostEqual(result[0].end, 2.0)

    # -- overlap prevention ----------------------------------------------------

    def test_no_snap_before_previous_note_end(self):
        """Snapping must not move a note before the previous note's end."""
        data = [
            self._make_data(1.00, 1.15, "first "),
            self._make_data(1.16, 1.30, "second "),
        ]
        # second note could wrongly snap to 1.05 (before first note ends at 1.15)
        onsets = np.array([1.00, 1.05])
        result = snap_to_onsets(data, onsets, max_snap_ms=80.0)
        # First note snaps to 1.00 — fine
        self.assertAlmostEqual(result[0].start, 1.00)
        # Second note must NOT snap to 1.05 (which is before first.end=1.15)
        self.assertGreaterEqual(result[1].start, result[0].end)

    def test_overlap_prevention_with_valid_later_onset(self):
        """When the nearest onset would cause overlap, but a valid position
        exists after clamping to prev_end, the note should still snap."""
        data = [
            self._make_data(1.00, 1.10, "first "),
            self._make_data(1.13, 1.50, "second "),
        ]
        # onset at 1.05 would overlap, but clamped to prev_end=1.10
        # 1.10 < 1.50 - 0.01 = 1.49 → valid
        onsets = np.array([1.00, 1.05])
        result = snap_to_onsets(data, onsets, max_snap_ms=80.0)
        self.assertAlmostEqual(result[1].start, 1.10)

    # -- custom max_snap_ms ---------------------------------------------------

    def test_custom_max_snap(self):
        """Custom max_snap_ms should be respected."""
        data = [self._make_data(1.05, 2.0)]
        onsets = np.array([1.0])
        # 50ms distance, max_snap=30ms → too far
        result = snap_to_onsets(data, onsets, max_snap_ms=30.0)
        self.assertAlmostEqual(result[0].start, 1.05)

    def test_large_max_snap(self):
        """Larger max_snap_ms catches more distant onsets."""
        data = [self._make_data(1.15, 2.0)]
        onsets = np.array([1.0])
        # 150ms distance, max_snap=200ms → snaps
        result = snap_to_onsets(data, onsets, max_snap_ms=200.0)
        self.assertAlmostEqual(result[0].start, 1.0)


if __name__ == "__main__":
    unittest.main()
