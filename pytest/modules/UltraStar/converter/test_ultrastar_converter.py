"""Tests for ultrastar_converter.py"""

from modules.Ultrastar.coverter.ultrastar_converter import real_bpm_to_ultrastar_bpm


def test_real_bpm_to_ultrastar_bpm():
    """Test case for real_bpm_to_ultrastar_bpm function."""
    # Real BPM is a multiple of 4
    assert real_bpm_to_ultrastar_bpm(120) == 30

    # Real BPM is not a multiple of 4
    assert real_bpm_to_ultrastar_bpm(130) == 32.5

    # Real BPM is zero
    assert real_bpm_to_ultrastar_bpm(0) == 0

    # Real BPM is a decimal number
    assert real_bpm_to_ultrastar_bpm(123.45) == 30.8625

    # Real BPM is a large number
    assert real_bpm_to_ultrastar_bpm(1000000) == 250000

    # Real BPM is a small number
    assert real_bpm_to_ultrastar_bpm(0.001) == 0.00025
