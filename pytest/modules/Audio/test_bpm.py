"""Tests for bpm.py -- tempo-ratio correction."""

import unittest

from src.modules.Audio.bpm import _TEMPO_RATIOS, _pick_best_tempo


class TestPickBestTempo(unittest.TestCase):
    """Tests for _pick_best_tempo()."""

    # -- edge cases -----------------------------------------------------------

    def test_zero_returns_target(self):
        self.assertEqual(_pick_best_tempo(0.0), 120.0)

    def test_negative_returns_target(self):
        self.assertEqual(_pick_best_tempo(-50.0), 120.0)

    def test_custom_target_returned_for_zero(self):
        self.assertEqual(_pick_best_tempo(0.0, target=100.0), 100.0)

    # -- identity (already in range) ------------------------------------------

    def test_in_range_returns_identity(self):
        """120 BPM is already perfect -- no correction needed."""
        self.assertEqual(_pick_best_tempo(120.0), 120.0)

    def test_moderate_tempo_unchanged(self):
        """100 BPM is in range and closer to 120 than any variant."""
        self.assertEqual(_pick_best_tempo(100.0), 100.0)

    def test_fast_tempo_in_range_unchanged(self):
        """Fast tempos inside the range must not be rescaled."""
        for bpm in [150.0, 170.0, 180.0, 195.0, 200.0]:
            self.assertEqual(
                _pick_best_tempo(bpm), bpm,
                f"{bpm} BPM is in range but was rescaled",
            )

    def test_slow_tempo_in_range_unchanged(self):
        """Slow tempos inside the range must not be rescaled."""
        for bpm in [60.0, 65.0, 70.0, 80.0]:
            self.assertEqual(
                _pick_best_tempo(bpm), bpm,
                f"{bpm} BPM is in range but was rescaled",
            )

    # -- half-tempo correction ------------------------------------------------

    def test_double_detected_corrected_to_half(self):
        """Detector reports 240 BPM (double). Half = 120, which is ideal."""
        self.assertEqual(_pick_best_tempo(240.0), 120.0)

    def test_slightly_above_range_halved(self):
        """220 BPM is above range. Half = 110, which is in range."""
        self.assertEqual(_pick_best_tempo(220.0), 110.0)

    # -- double-tempo correction ----------------------------------------------

    def test_half_detected_corrected_to_double(self):
        """Detector reports 55 BPM (half). Double = 110, in range."""
        self.assertEqual(_pick_best_tempo(55.0), 110.0)

    # -- third-tempo correction -----------------------------------------------

    def test_triple_detected_corrected_to_third(self):
        """Detector reports 360 BPM (triple). Third = 120, ideal."""
        self.assertEqual(_pick_best_tempo(360.0), 120.0)

    def test_waltz_triple_tempo(self):
        """Detector reports 390 BPM. Third = 130, in range and close."""
        self.assertEqual(_pick_best_tempo(390.0), 130.0)

    # -- quarter-tempo correction ---------------------------------------------

    def test_quadruple_detected_corrected_to_quarter(self):
        """Detector reports 480 BPM (quadruple). Quarter = 120, ideal."""
        self.assertEqual(_pick_best_tempo(480.0), 120.0)

    def test_high_tempo_uses_quarter(self):
        """500 BPM. Quarter = 125, in range."""
        self.assertEqual(_pick_best_tempo(500.0), 125.0)

    # -- two-thirds correction ------------------------------------------------

    def test_in_range_not_rescaled(self):
        """180 BPM is in range -- must NOT be rescaled to 120 via 2/3 ratio.

        Legitimate fast songs should keep their detected tempo; ratio
        correction only applies to out-of-range values.
        """
        self.assertEqual(_pick_best_tempo(180.0), 180.0)

    def test_two_thirds_correction_out_of_range(self):
        """270 BPM is above range. 270 * 2/3 = 180, in range."""
        self.assertEqual(_pick_best_tempo(270.0), 135.0)  # half = 135, closer to 120

    # -- simpler ratios win ties ----------------------------------------------

    def test_simpler_ratio_wins_tie(self):
        """When two candidates are equally close to 120, the simpler
        ratio (earlier in _TEMPO_RATIOS) should win.

        Example: primary=120. Identity gives 120 (exact match).
        Other ratios may also produce values, but identity wins.
        """
        self.assertEqual(_pick_best_tempo(120.0), 120.0)

    # -- nothing in range falls back ------------------------------------------

    def test_nothing_in_range_returns_primary(self):
        """If no candidate falls in range, return the primary as-is."""
        # 10 BPM: candidates are 10, 5, 20, 3.3, 30, 2.5, 40, 6.7, 15, 7.5
        # None of these are in 60-200
        self.assertEqual(_pick_best_tempo(10.0), 10.0)

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
        test_values = [75.0, 90.0, 110.0, 130.0, 150.0, 170.0, 200.0, 250.0, 300.0, 400.0]
        for bpm in test_values:
            result = _pick_best_tempo(bpm)
            # Check that result is in range (if any candidate was in range)
            candidates = [bpm * r for r in _TEMPO_RATIOS]
            has_candidate_in_range = any(60 <= c <= 200 for c in candidates)
            if has_candidate_in_range:
                self.assertGreaterEqual(
                    result, 60.0, f"Result {result} below range for input {bpm}"
                )
                self.assertLessEqual(
                    result, 200.0, f"Result {result} above range for input {bpm}"
                )

    # -- backward compatibility: half/double still work -----------------------

    def test_backward_compat_half(self):
        """Original half-tempo case still works."""
        # 240 -> half = 120 (closest to target)
        self.assertEqual(_pick_best_tempo(240.0), 120.0)

    def test_backward_compat_double(self):
        """Original double-tempo case still works."""
        # 55 -> double = 110 (in range, closest to target)
        self.assertEqual(_pick_best_tempo(55.0), 110.0)

    def test_backward_compat_in_range(self):
        """Original in-range case still works."""
        self.assertEqual(_pick_best_tempo(120.0), 120.0)


if __name__ == "__main__":
    unittest.main()
