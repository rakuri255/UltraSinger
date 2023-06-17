"""Tests for hyphenation.py"""

from src.modules.Speech_Recognition.hyphenation import hyphenation


def test_hypenation():
    """Test case for hyphenation function."""
    # test
    assert hyphenation("begegnen", "de_DE") == ["be", "geg", "nen"]
    assert hyphenation(".b,e~g'eg*nen, ", "de_DE") == [".b,e", "~g'eg", "*nen, "]
    assert hyphenation("Abend, ", "de_AT") == None
    assert hyphenation("Abend.", "de_AT") == None



