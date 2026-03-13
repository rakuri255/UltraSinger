"""Tests for bpm.py -- tempo-ratio correction."""

import unittest

from src.modules.Audio.bpm import _TEMPO_RATIOS, _pick_best_tempo


class TestPickBestTempo(unittest.TestCase):
    """Tests for _pick_best_tempo().

    Default range is [50, 500] with target=None (no bias toward any
    particular tempo).  Values inside the range pass through unchanged;
    out-of-range values are corrected using the candidate closest to
    the detected value.
    """

    # -- edge cases -----------------------------------------------------------

    def test_zero_returns_fallback(self):
        self.assertEqual(_pick_best_tempo(0.0), 120.0)

    def test_negative_returns_fallback(self):
        self.assertEqual(_pick_best_tempo(-50.0), 120.0)

    def test_custom_target_does_not_affect_zero_fallback(self):
        """Zero input always returns 120.0 regardless of target."""
        self.assertEqual(_pick_best_tempo(0.0, target=100.0), 120.0)

    # -- identity (already in range) ------------------------------------------

    def test_in_range_returns_identity(self):
        """120 BPM is in [50, 500] -- returned unchanged."""
        self.assertEqual(_pick_best_tempo(120.0), 120.0)

    def test_moderate_tempo_unchanged(self):
        """100 BPM is in range -- returned unchanged."""
        self.assertEqual(_pick_best_tempo(100.0), 100.0)

    def test_fast_tempo_in_range_unchanged(self):
        """Fast tempos inside the range must not be rescaled."""
        for bpm in [150.0, 170.0, 180.0, 195.0, 200.0, 300.0, 400.0, 500.0]:
            self.assertEqual(
                _pick_best_tempo(bpm), bpm,
                f"{bpm} BPM is in range but was rescaled",
            )

    def test_slow_tempo_in_range_unchanged(self):
        """Slow tempos inside the range must not be rescaled."""
        for bpm in [50.0, 55.0, 60.0, 65.0, 70.0, 80.0]:
            self.assertEqual(
                _pick_best_tempo(bpm), bpm,
                f"{bpm} BPM is in range but was rescaled",
            )

    # -- values that were previously "corrected" but now pass through ---------

    def test_240_bpm_in_range_unchanged(self):
        """240 BPM is inside [50, 500] -- no correction needed."""
        self.assertEqual(_pick_best_tempo(240.0), 240.0)

    def test_220_bpm_in_range_unchanged(self):
        """220 BPM is inside [50, 500] -- no correction needed."""
        self.assertEqual(_pick_best_tempo(220.0), 220.0)

    def test_360_bpm_in_range_unchanged(self):
        """360 BPM is inside [50, 500] -- no correction needed."""
        self.assertEqual(_pick_best_tempo(360.0), 360.0)

    def test_390_bpm_in_range_unchanged(self):
        """390 BPM is inside [50, 500] -- no correction needed."""
        self.assertEqual(_pick_best_tempo(390.0), 390.0)

    def test_480_bpm_in_range_unchanged(self):
        """480 BPM is inside [50, 500] -- no correction needed."""
        self.assertEqual(_pick_best_tempo(480.0), 480.0)

    def test_270_bpm_in_range_unchanged(self):
        """270 BPM is inside [50, 500] -- no correction needed."""
        self.assertEqual(_pick_best_tempo(270.0), 270.0)

    # -- core regression test: the original motivating case -------------------

    def test_bpm_409_passes_through(self):
        """409.66 BPM must pass through unchanged (UltraStar timing resolution).

        This was the original problem: librosa detected 409.66 BPM but the
        old range [60, 200] with target=120 incorrectly corrected it to
        136.55 (one-third), breaking timing alignment.
        """
        self.assertAlmostEqual(_pick_best_tempo(409.66), 409.66)

    # -- out-of-range correction (values outside [50, 500]) -------------------

    def test_very_high_bpm_corrected_to_half(self):
        """800 BPM is above 500. Half = 400, in range and closest to 800."""
        self.assertEqual(_pick_best_tempo(800.0), 400.0)

    def test_very_low_bpm_corrected(self):
        """25 BPM is below 50. Double = 50, in range and closest to 25."""
        self.assertEqual(_pick_best_tempo(25.0), 50.0)

    def test_extremely_high_bpm_uses_quarter(self):
        """2000 BPM. Quarter = 500, in range and closest to 2000."""
        self.assertEqual(_pick_best_tempo(2000.0), 500.0)

    def test_very_low_bpm_prefers_closest_to_original(self):
        """10 BPM: candidates 10, 5, 20, 3.3, 30, 2.5, 40, 6.7, 15, 7.5.

        None in [50, 500] -- returns primary as-is.
        """
        self.assertEqual(_pick_best_tempo(10.0), 10.0)

    # -- target=None uses primary as reference --------------------------------

    def test_target_none_picks_closest_to_primary(self):
        """With target=None, out-of-range correction picks the candidate
        closest to the detected value, not closest to 120.

        600 BPM -> candidates in [50, 500]:
          half = 300 (distance 300), three-quarters = 450 (distance 150),
          third = 200 (distance 400), quarter = 150 (distance 450).
        450 wins (closest to 600).
        """
        self.assertEqual(_pick_best_tempo(600.0), 450.0)

    def test_explicit_target_biases_selection(self):
        """With an explicit target, the candidate closest to that target
        is selected instead.

        600 BPM with target=120: half = 300 (dist 180), third = 200
        (dist 80), quarter = 150 (dist 30).  150 wins (closest to 120).
        """
        self.assertEqual(
            _pick_best_tempo(600.0, target=120.0), 150.0
        )

    # -- simpler ratios win ties ----------------------------------------------

    def test_simpler_ratio_wins_tie(self):
        """When two candidates are equally close to the reference, the simpler
        ratio (earlier in _TEMPO_RATIOS) should win.

        600 BPM with target=350:
          half  = 300 (ratio 0.5, distance |300-350| = 50)
          2/3   = 400 (ratio 2/3, distance |400-350| = 50)
        Both are in range and equidistant from target.  Half (0.5) is
        listed before two-thirds (2/3) in _TEMPO_RATIOS, so 300 wins.
        """
        self.assertEqual(
            _pick_best_tempo(600.0, target=350.0), 300.0
        )

    # -- custom range ---------------------------------------------------------

    def test_custom_range_respected(self):
        """Custom low/high/target boundaries should be used."""
        # 300 BPM with range 200-400 and target 300: identity wins
        self.assertEqual(
            _pick_best_tempo(300.0, low=200.0, high=400.0, target=300.0),
            300.0,
        )

    def test_custom_narrow_range(self):
        """Narrow range forces specific candidate selection."""
        # 240 BPM with range 115-125: half = 120 is in range
        self.assertEqual(
            _pick_best_tempo(240.0, low=115.0, high=125.0), 120.0
        )

    # -- result is always in range when candidates exist ----------------------

    def test_result_in_range(self):
        """For typical inputs, the result should be within the default range."""
        test_values = [75.0, 90.0, 110.0, 130.0, 150.0, 170.0, 200.0,
                       250.0, 300.0, 400.0, 500.0, 600.0, 800.0]
        for bpm in test_values:
            result = _pick_best_tempo(bpm)
            # Check that result is in range (if any candidate was in range)
            candidates = [bpm * r for r in _TEMPO_RATIOS]
            has_candidate_in_range = any(50 <= c <= 500 for c in candidates)
            if has_candidate_in_range:
                self.assertGreaterEqual(
                    result, 50.0, f"Result {result} below range for input {bpm}"
                )
                self.assertLessEqual(
                    result, 500.0, f"Result {result} above range for input {bpm}"
                )

    # -- 180 BPM in-range check (unchanged from before) -----------------------

    def test_180_bpm_in_range_not_rescaled(self):
        """180 BPM is in range -- must NOT be rescaled.

        Legitimate fast songs should keep their detected tempo.
        """
        self.assertEqual(_pick_best_tempo(180.0), 180.0)


if __name__ == "__main__":
    unittest.main()
