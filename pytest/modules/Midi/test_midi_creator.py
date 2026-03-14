"""Tests for midi_creator.py — global and local octave correction."""

import unittest

import librosa
import numpy as np

from src.modules.Midi.MidiSegment import MidiSegment
from src.modules.Midi.midi_creator import (
    apply_octave_shift,
    correct_global_octave,
    correct_octave_outliers,
    confidence_weighted_median_note,
)


class TestCorrectGlobalOctave(unittest.TestCase):
    """Tests for correct_global_octave()."""

    # -- edge cases ----------------------------------------------------------

    def test_empty_list_returns_empty(self):
        result = correct_global_octave([])
        self.assertEqual(result, [])

    def test_single_note_in_range_unchanged(self):
        seg = MidiSegment(note="C4", start=0, end=1, word="hello ")
        result = correct_global_octave([seg])
        self.assertEqual(result[0].note, "C4")

    # -- already in range (no shift) -----------------------------------------

    def test_notes_in_range_unchanged(self):
        """Notes whose median is inside 48-84 should not be shifted."""
        segs = [
            MidiSegment(note="C4", start=0, end=1, word="one "),   # MIDI 60
            MidiSegment(note="E4", start=1, end=2, word="two "),   # MIDI 64
            MidiSegment(note="G4", start=2, end=3, word="three "), # MIDI 67
        ]
        result = correct_global_octave(segs)
        notes = [s.note for s in result]
        self.assertEqual(notes, ["C4", "E4", "G4"])

    # -- shift up (sub-harmonic detection) ------------------------------------

    def test_all_notes_two_octaves_low_shifted_up(self):
        """Median MIDI ~36 (C2) -> needs +24 shift to reach 48+ range."""
        segs = [
            MidiSegment(note="C2", start=0, end=1, word="a "),   # MIDI 36
            MidiSegment(note="D2", start=1, end=2, word="b "),   # MIDI 38
            MidiSegment(note="E2", start=2, end=3, word="c "),   # MIDI 40
        ]
        result = correct_global_octave(segs)
        # All should be shifted up by +12 (one octave brings median to 48-50)
        midi_values = [librosa.note_to_midi(s.note) for s in result]
        for val in midi_values:
            self.assertGreaterEqual(val, 48)
            self.assertLessEqual(val, 84)

    def test_all_notes_four_octaves_low_shifted_up(self):
        """Simulate a real-world sub-harmonic detection problem: MIDI 21-26.

        The function shifts all notes so the *median* lands inside the
        vocal range.  Individual notes may still be below ``low`` if
        they were already the lowest in the set.
        """
        segs = [
            MidiSegment(note="D1", start=0, end=1, word="wake "),   # MIDI 26
            MidiSegment(note="A0", start=1, end=2, word="me "),     # MIDI 21
            MidiSegment(note="C1", start=2, end=3, word="up "),     # MIDI 24
            MidiSegment(note="B0", start=3, end=4, word="in "),     # MIDI 23
            MidiSegment(note="D1", start=4, end=5, word="side "),   # MIDI 26
        ]
        result = correct_global_octave(segs)
        midi_values = [librosa.note_to_midi(s.note) for s in result]
        median_after = float(np.median(midi_values))
        # Median must be within the default vocal range
        self.assertGreaterEqual(median_after, 48)
        self.assertLessEqual(median_after, 84)
        # All notes must have been shifted up (original median was 24)
        for val in midi_values:
            self.assertGreater(val, 24)

    # -- shift down (over-detection) ------------------------------------------

    def test_all_notes_above_range_shifted_down(self):
        """Median MIDI ~96 (C7) -> needs shift down into range."""
        segs = [
            MidiSegment(note="C7", start=0, end=1, word="high "),   # MIDI 96
            MidiSegment(note="D7", start=1, end=2, word="note "),   # MIDI 98
            MidiSegment(note="E7", start=2, end=3, word="test "),   # MIDI 100
        ]
        result = correct_global_octave(segs)
        midi_values = [librosa.note_to_midi(s.note) for s in result]
        for val in midi_values:
            self.assertLessEqual(val, 84)

    # -- shift is always a multiple of 12 ------------------------------------

    def test_shift_is_multiple_of_12(self):
        """The shift must be a whole number of octaves (multiple of 12)."""
        segs = [
            MidiSegment(note="C2", start=0, end=1, word="lo "),  # MIDI 36
            MidiSegment(note="G2", start=1, end=2, word="w "),   # MIDI 43
        ]
        original_midis = [librosa.note_to_midi(s.note) for s in segs]
        result = correct_global_octave(segs)
        shifted_midis = [librosa.note_to_midi(s.note) for s in result]

        for orig, shifted in zip(original_midis, shifted_midis, strict=True):
            self.assertEqual((shifted - orig) % 12, 0)

    # -- custom range ---------------------------------------------------------

    def test_custom_range(self):
        """Custom low/high should be respected."""
        segs = [
            MidiSegment(note="C4", start=0, end=1, word="mid "),  # MIDI 60
        ]
        # With a very narrow high range, C4 (60) is above high=55
        result = correct_global_octave(segs, low=40, high=55)
        midi_val = librosa.note_to_midi(result[0].note)
        self.assertLessEqual(midi_val, 55)

    # -- preserves relative intervals -----------------------------------------

    def test_relative_intervals_preserved(self):
        """After shift, the intervals between notes must be identical."""
        segs = [
            MidiSegment(note="C2", start=0, end=1, word="do "),   # MIDI 36
            MidiSegment(note="E2", start=1, end=2, word="mi "),   # MIDI 40
            MidiSegment(note="G2", start=2, end=3, word="sol "),  # MIDI 43
        ]
        original_midis = [librosa.note_to_midi(s.note) for s in segs]
        original_intervals = [
            original_midis[i + 1] - original_midis[i]
            for i in range(len(original_midis) - 1)
        ]

        result = correct_global_octave(segs)
        shifted_midis = [librosa.note_to_midi(s.note) for s in result]
        shifted_intervals = [
            shifted_midis[i + 1] - shifted_midis[i]
            for i in range(len(shifted_midis) - 1)
        ]

        self.assertEqual(original_intervals, shifted_intervals)


class TestCorrectOctaveOutliers(unittest.TestCase):
    """Tests for correct_octave_outliers() — single and multi-pass."""

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _make_segs(notes: list[str]) -> list[MidiSegment]:
        """Create MidiSegments from a list of note names."""
        return [
            MidiSegment(note=n, start=float(i), end=float(i + 1), word=f"w{i} ")
            for i, n in enumerate(notes)
        ]

    @staticmethod
    def _get_notes(segs: list[MidiSegment]) -> list[str]:
        return [s.note for s in segs]

    @staticmethod
    def _get_midis(segs: list[MidiSegment]) -> list[int]:
        return [librosa.note_to_midi(s.note) for s in segs]

    # -- edge cases -----------------------------------------------------------

    def test_empty_list_returns_empty(self):
        result = correct_octave_outliers([])
        self.assertEqual(result, [])

    def test_two_notes_unchanged(self):
        """Less than 3 notes should be returned unchanged."""
        segs = self._make_segs(["C4", "E4"])
        result = correct_octave_outliers(segs)
        self.assertEqual(self._get_notes(result), ["C4", "E4"])

    # -- single outlier (works with one pass) ---------------------------------

    def test_single_outlier_corrected(self):
        """One note an octave too low among correct neighbours."""
        # C4=60, C4, C2=36(!), C4, C4
        notes = ["C4", "C4", "C2", "C4", "C4"]
        segs = self._make_segs(notes)
        result = correct_octave_outliers(segs, passes=1)
        midis = self._get_midis(result)
        # C2 (36) should be corrected to C4 (60)
        self.assertEqual(midis[2], 60)

    def test_single_outlier_above_corrected(self):
        """One note an octave too high among correct neighbours."""
        # C4=60, C4, C6=84(!), C4, C4
        notes = ["C4", "C4", "C6", "C4", "C4"]
        segs = self._make_segs(notes)
        result = correct_octave_outliers(segs, passes=1)
        midis = self._get_midis(result)
        # C6 (84) should be corrected to C4 (60)
        self.assertEqual(midis[2], 60)

    # -- cluster of wrong-octave notes (needs multiple passes) ----------------

    def test_cluster_of_wrong_octave_notes_corrected(self):
        """A cluster of 8 sub-harmonic notes surrounded by correct notes.

        Single pass cannot fix the cluster because the local median
        within the cluster is also wrong.  Two passes should correct
        all notes.
        """
        # 6x C4, then 8x C3 (one octave too low), then 6x C4
        correct = ["C4"] * 6
        cluster = ["C3"] * 8
        notes = correct + cluster + correct
        segs = self._make_segs(notes)

        result = correct_octave_outliers(segs, passes=2)
        midis = self._get_midis(result)

        # All notes should now be C4 (MIDI 60)
        for i, midi_val in enumerate(midis):
            self.assertEqual(
                midi_val, 60,
                f"Note {i} is MIDI {midi_val}, expected 60 (C4)"
            )

    def test_two_passes_fixes_more_than_one(self):
        """Verify that a second pass corrects notes left behind by pass 1.

        With window=3, a cluster of 5 wrong notes surrounded by 4 correct
        on each side cannot be fully fixed in one pass.
        """
        correct = ["E4"] * 4  # MIDI 64
        cluster = ["E3"] * 5  # MIDI 52, one octave low
        notes = correct + cluster + correct
        segs = self._make_segs(notes)

        # One pass with small window - may not fix the inner cluster notes
        result_1pass = correct_octave_outliers(
            self._make_segs(notes), passes=1, window=3
        )
        midis_1pass = self._get_midis(result_1pass)
        unfixed_1pass = sum(1 for v in midis_1pass if v == 52)

        # Two passes should fix more (or all)
        result_2pass = correct_octave_outliers(segs, passes=2, window=3)
        midis_2pass = self._get_midis(result_2pass)
        unfixed_2pass = sum(1 for v in midis_2pass if v == 52)

        self.assertLessEqual(
            unfixed_2pass, unfixed_1pass,
            "Two passes should fix at least as many outliers as one pass"
        )

    # -- passes=1 backward compatibility --------------------------------------

    def test_passes_1_backward_compatible(self):
        """passes=1 should behave identically to the old single-pass code."""
        notes = ["C4", "C4", "C2", "C4", "C4"]
        segs = self._make_segs(notes)
        result = correct_octave_outliers(segs, passes=1)
        midis = self._get_midis(result)
        self.assertEqual(midis[2], 60)  # C2 -> C4

    # -- global median tie-breaker --------------------------------------------

    def test_global_median_tiebreaker(self):
        """When multiple octave shifts are valid, prefer the one closest
        to the global median.

        All notes are around C4 (60) except one outlier at C2 (36).
        Both C4 (60) and C6 (84) could bring C2 within 11 ST of the
        local median.  C4 is closer to the global median (~60), so
        C4 should be chosen.
        """
        notes = ["C4", "D4", "E4", "C2", "D4", "E4", "C4"]
        segs = self._make_segs(notes)
        result = correct_octave_outliers(segs, passes=1)
        midis = self._get_midis(result)
        # C2 (36) should become C4 (60), not C6 (84)
        self.assertEqual(midis[3], 60)

    # -- early termination when no corrections --------------------------------

    def test_early_termination_no_changes(self):
        """If all notes are already correct, no corrections should happen
        and the function should terminate early.
        """
        notes = ["C4", "D4", "E4", "F4", "G4"]
        segs = self._make_segs(notes)
        result = correct_octave_outliers(segs, passes=5)
        self.assertEqual(self._get_notes(result), notes)

    # -- preserves non-outlier notes ------------------------------------------

    def test_correct_notes_unchanged(self):
        """Notes that are not outliers should remain exactly unchanged."""
        notes = ["C4", "E4", "G4", "C2", "E4", "G4", "C4"]
        segs = self._make_segs(notes)
        original_correct = [60, 64, 67, None, 64, 67, 60]

        result = correct_octave_outliers(segs)
        midis = self._get_midis(result)

        for i, expected in enumerate(original_correct):
            if expected is not None:
                self.assertEqual(
                    midis[i], expected,
                    f"Note {i} changed from {expected} to {midis[i]}"
                )

    # -- valid minority melody note preserved ---------------------------------

    def test_valid_melody_note_preserved_by_consensus(self):
        """G4 (MIDI 67) is 7 ST from global median C4 (60).

        Phase 2 must NOT shift it — it is a valid melody interval,
        not an octave error.  Only notes ≥ 12 ST away should be touched.
        """
        # 6x C4, 1x G4, 6x C4  → global median ≈ 60
        notes = ["C4"] * 6 + ["G4"] + ["C4"] * 6
        segs = self._make_segs(notes)
        result = correct_octave_outliers(segs, passes=2)
        midis = self._get_midis(result)
        # G4 (67) must remain unchanged
        self.assertEqual(midis[6], 67, "G4 was incorrectly shifted by consensus")

    def test_phase1_and_phase2_cooperate(self):
        """Phase 1 (local) and Phase 2 (global consensus) each fix
        a different kind of outlier.

        Setup: 6x C4, 8x C3 (cluster — too large for local fix),
               1x C2 (isolated — fixed by Phase 1), 6x C4.
        Phase 1 corrects C2 → C4 (24 ST from local median, clear outlier).
        Phase 2 corrects the C3 cluster → C4 (12 ST from global median=60,
        local windows are contaminated so Phase 1 can't help).
        """
        notes = ["C4"] * 6 + ["C3"] * 8 + ["C2"] + ["C4"] * 6
        segs = self._make_segs(notes)
        result = correct_octave_outliers(segs, passes=2)
        midis = self._get_midis(result)
        # C2 (index 14) should be fixed by Phase 1
        self.assertEqual(midis[14], 60, "Isolated C2 not fixed by Phase 1")
        # C3 cluster (indices 6-13) should be fixed by Phase 2
        for i in range(6, 14):
            self.assertEqual(
                midis[i], 60,
                f"Cluster note {i} is MIDI {midis[i]}, expected 60 (C4)"
            )
        # Boundary C4s must remain unchanged
        for i in list(range(0, 6)) + list(range(15, 21)):
            self.assertEqual(
                midis[i], 60,
                f"Boundary note {i} is MIDI {midis[i]}, expected 60 (C4)"
            )

    # -- shift is always a multiple of 12 ------------------------------------

    def test_shifts_are_octave_multiples(self):
        """Any correction must be an exact octave shift (multiple of 12)."""
        notes = ["C4", "C4", "C2", "C4", "C4", "C6", "C4", "C4"]
        original = [60, 60, 36, 60, 60, 84, 60, 60]
        segs = self._make_segs(notes)
        result = correct_octave_outliers(segs)
        midis = self._get_midis(result)

        for i, (orig, new) in enumerate(zip(original, midis)):
            self.assertEqual(
                (new - orig) % 12, 0,
                f"Note {i}: shift {new - orig} is not a multiple of 12"
            )


class TestApplyOctaveShift(unittest.TestCase):
    """Tests for apply_octave_shift()."""

    def _make_seg(self, note, start=0, end=1, word="x "):
        return MidiSegment(note=note, start=start, end=end, word=word)

    def _get_midis(self, segs):
        return [librosa.note_to_midi(s.note) for s in segs]

    # -- edge cases ----------------------------------------------------------

    def test_empty_list_returns_empty(self):
        result = apply_octave_shift([], 1)
        self.assertEqual(result, [])

    def test_zero_shift_unchanged(self):
        segs = [self._make_seg("C4"), self._make_seg("E4")]
        result = apply_octave_shift(segs, 0)
        self.assertEqual([s.note for s in result], ["C4", "E4"])

    def test_none_octaves_unchanged(self):
        """Calling with octaves=0 should be a no-op."""
        segs = [self._make_seg("C4")]
        original_note = segs[0].note
        apply_octave_shift(segs, 0)
        self.assertEqual(segs[0].note, original_note)

    # -- shift up ------------------------------------------------------------

    def test_shift_up_one_octave(self):
        segs = [self._make_seg("C3"), self._make_seg("E3"), self._make_seg("G3")]
        result = apply_octave_shift(segs, 1)
        self.assertEqual([s.note for s in result], ["C4", "E4", "G4"])

    def test_shift_up_two_octaves(self):
        segs = [self._make_seg("C2")]
        result = apply_octave_shift(segs, 2)
        self.assertEqual(result[0].note, "C4")

    # -- shift down ----------------------------------------------------------

    def test_shift_down_one_octave(self):
        segs = [self._make_seg("C5"), self._make_seg("E5")]
        result = apply_octave_shift(segs, -1)
        self.assertEqual([s.note for s in result], ["C4", "E4"])

    def test_shift_down_two_octaves(self):
        segs = [self._make_seg("C6")]
        result = apply_octave_shift(segs, -2)
        self.assertEqual(result[0].note, "C4")

    # -- MIDI range clamping -------------------------------------------------

    def test_shift_up_beyond_127_skipped(self):
        """Notes that would exceed MIDI 127 should be left unchanged."""
        segs = [self._make_seg("G9")]  # MIDI 127
        result = apply_octave_shift(segs, 1)
        # G9 + 12 = MIDI 139 > 127, should stay G9
        self.assertEqual(result[0].note, "G9")

    def test_shift_down_below_0_skipped(self):
        """Notes that would go below MIDI 0 should be left unchanged."""
        segs = [self._make_seg("C0")]  # MIDI ~12
        result = apply_octave_shift(segs, -2)
        # C0 - 24 = MIDI -12 < 0, should stay C0
        self.assertEqual(result[0].note, "C0")

    # -- modifies in-place ---------------------------------------------------

    def test_modifies_in_place(self):
        segs = [self._make_seg("C3")]
        apply_octave_shift(segs, 1)
        self.assertEqual(segs[0].note, "C4")

    # -- preserves word and timing -------------------------------------------

    def test_preserves_word_and_timing(self):
        seg = self._make_seg("C3", start=1.5, end=2.5, word="hello ")
        apply_octave_shift([seg], 1)
        self.assertEqual(seg.word, "hello ")
        self.assertEqual(seg.start, 1.5)
        self.assertEqual(seg.end, 2.5)

    # -- mixed notes ---------------------------------------------------------

    def test_shift_preserves_intervals(self):
        """Shifting should preserve all intervals between notes."""
        segs = [self._make_seg("C3"), self._make_seg("E3"), self._make_seg("G3")]
        midis_before = self._get_midis(segs)
        intervals_before = [midis_before[i+1] - midis_before[i] for i in range(len(midis_before)-1)]

        apply_octave_shift(segs, 1)

        midis_after = self._get_midis(segs)
        intervals_after = [midis_after[i+1] - midis_after[i] for i in range(len(midis_after)-1)]
        self.assertEqual(intervals_before, intervals_after)


class TestConfidenceWeightedMedianNote(unittest.TestCase):
    """Tests for confidence_weighted_median_note()."""

    def test_single_frequency(self):
        note = confidence_weighted_median_note([440.0], [1.0])
        self.assertEqual(note, "A4")

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            confidence_weighted_median_note([], [])

    def test_mismatched_lengths_raises(self):
        with self.assertRaises(ValueError):
            confidence_weighted_median_note([440.0, 880.0], [1.0])

    def test_zero_weights_raises(self):
        with self.assertRaises(ValueError):
            confidence_weighted_median_note([440.0], [0.0])

    def test_high_confidence_dominates(self):
        """A single high-confidence frame should dominate low-confidence noise."""
        freqs = [440.0, 220.0, 220.0, 220.0]
        weights = [0.95, 0.1, 0.1, 0.1]
        note = confidence_weighted_median_note(freqs, weights)
        # The high-weight A4 (440 Hz) should win
        self.assertEqual(note, "A4")


if __name__ == "__main__":
    unittest.main()
